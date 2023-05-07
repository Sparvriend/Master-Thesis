import copy
import captum
from captum.attr import visualization as viz
import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models
import torchvision.transforms as T
from tqdm import tqdm
import warnings


# Temproary imports for RBF DUQ
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchmetrics import Accuracy, F1Score
from torch import nn, optim

from utils import cutoff_date, get_data_loaders, get_transforms

from datasets import NTZFilterDataset, CIFAR10Dataset, TinyImageNet200Dataset


class RBF_model(nn.Module):
    """RBF layer definition taken from Joost van Amersfoort's implementation of DUQ:
    https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/resnet_duq.py
    and inspired by Matias Valdenegro Toro implementation:
    https://github.com/mvaldenegro/keras-uncertainty/blob/master/keras_uncertainty/layers/rbf_layers.py

    Variable explanations:
    kernels = ?. Shape = [Fe out features, classes, fe out features]
    N = ?. Shape = [Classes]
    m = ?. Shape = [fe out features, classes]
    # For MobileNetV2, fe out features is 1280

    The essence of DUQ is that it learns a set of centroids for each class,
    which it can then compare to new inputs during inference time. The
    distance to the closest centroid is the uncertainty score.
    """
    def __init__(self, fe, in_features, out_features, device):
        super(RBF_model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = 0.1
        self.gamma = 0.999
        self.fe = fe
        self.device = device

        # Initializing kernels and centroids
        self.kernels = nn.Parameter(torch.Tensor(in_features, out_features, in_features)).to(device)
        self.N = (torch.zeros(out_features) + 13).to(device) # (Why did Joost set this value to 13?)
        self.m = torch.zeros(in_features, out_features).to(device)
        self.m *= self.N

        nn.init.normal_(self.m, 0.05, 1)
        nn.init.kaiming_normal_(self.kernels, nonlinearity = 'relu')


    def forward(self, x):
        # Getting feature output from fe and then applying kernels
        z = self.fe(x)
        z = torch.einsum("ij, mnj->imn", z, self.kernels).to(self.device)

        # Getting embedded centres
        c = (self.m / self.N.unsqueeze(0)).unsqueeze(0).to(self.device)

        # Getting distances to each centroid
        distances = ((z - c) ** 2).mean(1) / (2 * self.sigma ** 2)

        # With Gaussian distribution
        distances = torch.exp(-1 * distances)
        return distances


    def update_centres(self, inputs, labels):
        # Defining update function
        update_f = lambda x, y: self.gamma * x + (1 - self.gamma) * y

        # Summing labels for updating N
        unique, counts = torch.unique(labels, return_counts = True)
        labels_sum = torch.zeros(self.out_features, dtype = torch.long).to(self.device)
        labels_sum[unique] = counts
 
        # Update N
        self.N = update_f(self.N, labels_sum)

        # Calculating centroid sum
        x = self.fe(inputs)
        x = torch.einsum("ij, mnj->imn", x, self.kernels)
        labels = labels.unsqueeze(1).cpu()
        x = x.type(torch.LongTensor)
        centroid_sum = torch.einsum("ijk, il->jk", x, labels).to(self.device)

        # Update m here
        self.m = update_f(self.m, centroid_sum)


def visualize_explainability(img_data: torch.Tensor, img_paths: list, img_destination: str):
    """Function that visualizes an explainability result.
    It combines the original image with image data that explains
    the decision making of the model.

    Args:
        img_data: torch tensor with image data.
        img_paths: list of paths to the original images.
        img_destination: path to folder to save the images in.
    """
    # Custom transform for only resizing and then cropping to center
    transform = T.Compose([T.Resize(256, max_size = 320), T.CenterCrop(224)])

    # Setting experiment name
    experiment_name = os.path.normpath(img_destination).split(os.sep)[-1]

    # Creating custom colormaps
    bw_cmap = LinearSegmentedColormap.from_list("custom bw",
                                                [(0, "#ffffff"),
                                                 (0.25, "#000000"),
                                                 (1, "#000000")],
                                                N = 256)
    
    # Iterating over all image data and saving 
    # them in combination with the original image
    for i, img in enumerate(img_data):
        # Retrieving the image, with the label printed on it
        img_name = os.path.normpath(img_paths[i]).split(os.sep)[-1]
        img_path = os.path.join("Results", "Test-Predictions",  experiment_name, img_name) 
        norm_img = Image.open(img_path)
        norm_img = transform(norm_img)
        exp_img = np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0))
        fig, _ = viz.visualize_image_attr_multiple(exp_img,
                                                   np.asarray(norm_img),
                                                   methods = ["original_image", "heat_map"],
                                                   signs = ["all", "positive"],
                                                   cmap = bw_cmap,
                                                   show_colorbar = True)

        img_name = os.path.normpath(img_paths[i]).split(os.sep)[-1]
        fig.savefig(os.path.join(img_destination, img_name.replace(".bmp", ".png")))
        plt.close()


def explainability_setup(model: torchvision.models, img_paths: list, option: str,
                         device: torch.device, input_concat: torch.Tensor,
                         predicted_labels: list, experiment_folder: str):
    """Function that generates function arguments for explainability.
    Since Captum has a very similar way of running the explainability functions
    Doing it like this is a nice option.

    Args:
        model: model to explain.
        img_paths: list of paths to the original images.
        option: name of the explainability method.
        device: device to run the model on.
        input_concat: concatenated input tensor.
        predicted_labels: list of predicted labels.
        experiment_folder: path to the experiment folder that was tested on.
    """
    # Getting experiment_name and creating the folder to paste the images in
    experiment_name = cutoff_date(experiment_folder)
    img_desintation = os.path.join("Results", "Explainability-Results", experiment_name)
    if not os.path.exists(img_desintation):
        os.mkdir(img_desintation)

    # Ignoring UserWarnings, since they are not helpful
    # 1st warning is about requireing grad which it then sets
    # 2nd warning is about setting backward hooks for ReLu activations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Based on which explainability option is selected
        # Arguments are created and passed to the explainability function
        args = {"inputs": input_concat.to(device), "target": predicted_labels} 
        if option == "integrated_gradients":
            args["internal_batch_size"] = len(predicted_labels)
            args["n_steps"] = 200
            print("Running integrated gradients for model explanation")
            gen_model_explainability(captum.attr.IntegratedGradients, model,
                                     img_paths, args, img_desintation)
        elif option == "saliency_map":
            print("Running saliency map for model explanation")
            gen_model_explainability(captum.attr.Saliency, model,
                                     img_paths, args, img_desintation)
        elif option == "deeplift":
            print("Running deeplift for model explanation")
            gen_model_explainability(captum.attr.DeepLift, model,
                                     img_paths, args, img_desintation)
        elif option == "guided_backpropagation":
            print("Running guided backpropagation for model explanation")
            gen_model_explainability(captum.attr.GuidedBackprop, model,
                                     img_paths, args, img_desintation)
        else:
            print("Explainability option not valid")


def gen_model_explainability(explain_func, model: torchvision.models,
                             img_paths: list, args: list, img_desintation: str):
    """Function that generates an explainability object,
    which is basically a significance value per pixel for each image,
    and passes it to the visualization function
    
    Args:
        explain_func: function that generates the explainability object.
        model: model to explain.
        img_paths: list of paths to the original images.
        args: list of arguments for the explainability function.
        img_desintation: path to the folder where the images are saved.
    """
    # Creating explainability object via Captum
    explainability_obj = explain_func(model)
    # Unpacking dictionary arguments with **
    explainability_attr = explainability_obj.attribute(**args)
    visualize_explainability(explainability_attr, img_paths, img_desintation)


def rbf_uncertainty():
    """Function that calculates deep uncertainty based on
    radial basis function (RBF). It calculates uncertainty
    based on the distance to average centroids. It is a
    seperate testing module, since it predicts its own labels.
    """
    # Setting up the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Loading model and defining experiment name
    feature_extractor = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
    feature_extractor.classifier = nn.Sequential(nn.Dropout(p = 0.2))
    model = RBF_model(feature_extractor, 1280, 4, device)
    model.to(device)

    #data_loaders = get_data_loaders(transform = get_transforms("categorical"))
    #data_loaders = get_data_loaders(transform = get_transforms("rand_augment"))
    data_loaders = get_data_loaders(transform = get_transforms("no_augment"))

    # Optimizer params taken from DUQ paper
    optimizer = optim.SGD(model.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 0.0005)
    acc_metric = Accuracy(task = "multiclass", num_classes = 4).to(device)
    f1_metric = F1Score(task = "multiclass", num_classes = 4).to(device)
    criterion = nn.CrossEntropyLoss()

    # Training loop:
    for i in range(50):
        print("Epoch = " + str(i))
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                print("Training phase")
            else:
                model.eval()
                print("Validation phase")
            # Set model metrics to 0 and starting model timer
            loss_over_epoch = 0
            acc = 0
            f1_score = 0
            combined_labels = []
            combined_labels_pred = []
            for inputs, labels in tqdm(data_loaders[phase]):
                with torch.set_grad_enabled(phase == "train"):
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    inputs.requires_grad_(True)

                    # Forward pass
                    model_output = model(inputs)
                    predicted_labels = model_output.argmax(dim = 1)
                    
                    # Calculating loss and metrics
                    loss = criterion(model_output, labels)
                    acc += acc_metric(predicted_labels, labels).item()
                    f1_score += f1_metric(predicted_labels, labels).item()

                    if phase == "train":
                        # Updating model weights
                        loss.backward()
                        optimizer.step()
                        # Updating RBF centres
                        inputs.requires_grad_(False)
                        with torch.no_grad():
                            model.eval()
                            model.update_centres(inputs, labels)
    
                    loss_over_epoch += loss.item()
                    combined_labels.append(labels)
                    combined_labels_pred.append(predicted_labels)

            # Printing performance metrics
            mean_accuracy = (acc / len(data_loaders[phase]))
            mean_f1_score = (f1_score / len(data_loaders[phase]))
            print("Loss = " + str(round(loss_over_epoch, 2)))
            print("Accuracy = " + str(round(mean_accuracy, 2)))
            print("F1 score = " + str(round(mean_f1_score, 2)))


def deep_ensemble_uncertainty():
   # Hardest to implement? -> No clear library available
   # Testing module, but can not import test_model due to circular imports
    print("Not yet implemented")


if __name__ == '__main__':
    rbf_uncertainty()