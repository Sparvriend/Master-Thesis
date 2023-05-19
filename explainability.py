import argparse
import captum
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as T
from types import SimpleNamespace
import warnings

from train import train_model
from test import setup_testing, test_model
from utils import get_transforms, get_data_loaders, setup_tensorboard, \
                  setup_hyp_file, set_classification_layer, \
                  add_confusion_matrix, setup_hyp_dict, merge_experiments, \
                  save_test_predicts


class RBF_model(nn.Module):
    """RBF layer definition based on Joost van Amersfoort's implementation
    of Determenistic Uncertainty Quantification (DUQ):
    https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/resnet_duq.py
    and further inspired by Matias Valdenegro Toro implementation:
    https://github.com/mvaldenegro/keras-uncertainty/blob/master/keras_uncertainty/layers/rbf_layers.py

    Variable explanations:
    out_features is the amount of classes of the model.
    in_features is the amount of features inserted into an RBF layer by the model.

    [kernels] holds the representation of conversion to a feature space.
    Any time an output of the feature extractor is calculated it is first
    matrix multiplied (einsum) with the kernels to get a feature space
    representation of the feature extractor output. A parameter of the
    model, hence updated every backwards pass.
    Shape = [in features, classes, in features]

    [N] holds the label counts multiplied by the constant gamma. In essence
    it holds the frequency of each label relative to the other labels.
    Shape = [classes]

    [m] holds the centroid sum multiplied by the constant gamma. The centroid sum
    consists of feature extractor output, combined through matrix multiplication
    (einsum) with the kernels. The result is then again combined (einsum) with 
    the labels to get a sum of the feature extractor output for each label;
    the centroid sum.
    Shape = [in features, classes]

    [m / N] Gives the centroids, it applies the relative label frequency
    of N to m.
    Shape = [in features, classes]

    The essence of DUQ is that it learns a set of centroids for each class,
    which it can then compare to new inputs during inference time. The
    distance to the closest centroid is the uncertainty metric.
    """
    def __init__(self, fe, in_features, out_features, device):
        super(RBF_model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = 0.1
        self.gamma = 0.999
        self.fe = fe
        self.device = device

        # Initializing kernels centroid embedding
        self.kernels = nn.Parameter(torch.Tensor(in_features, out_features,
                                                 in_features))
        self.N = (torch.ones(out_features)).to(device)
        self.m = torch.zeros(in_features, out_features).to(device)

        nn.init.normal_(self.m, 0.05, 1)
        nn.init.kaiming_normal_(self.kernels, nonlinearity = 'relu')
        self.m *= self.N


    def forward(self, x):
        # Getting feature output from fe and then applying kernels
        z = self.fe(x)
        z = torch.einsum("ij, mnj->imn", z, self.kernels)

        # Getting embedded centroids
        c = (self.m / self.N.unsqueeze(0)).unsqueeze(0).to(self.device)

        # Getting distances to each centroid
        distances = ((z - c) ** 2).mean(1) / (2 * self.sigma ** 2)

        # With Gaussian distribution
        distances = torch.exp(-1 * distances)
        return distances


    def update_centroids(self, inputs, labels):
        # Defining update function
        update_f = lambda x, y: self.gamma * x + (1 - self.gamma) * y

        # Summing labels for updating N
        unique, counts = torch.unique(labels, return_counts = True)
        labels_sum = torch.zeros(self.out_features,
                                 dtype = torch.long).to(self.device)
        labels_sum[unique] = counts
 
        # Update N
        self.N = update_f(self.N, labels_sum)

        # Calculating centroid sum
        z = self.fe(inputs)
        z = torch.einsum("ij, mnj->imn", z, self.kernels)
        labels = labels.unsqueeze(1).cpu()
        z = z.type(torch.LongTensor)
        centroid_sum = torch.einsum("ijk, il->jk", z, labels).to(self.device)

        # Update m
        self.m = update_f(self.m, centroid_sum)


    def get_gradients(self, inputs, model_output):
        """Function that calculates a gradients for model inputs,
        given the predicted output.

        Args:
            inputs: Model inputs.
            model_output: Predicted labels given input.
        """
        gradients = torch.autograd.grad(outputs = model_output, inputs = inputs,
                                        grad_outputs = torch.ones_like(model_output),
                                        create_graph = True)[0]
        return gradients.flatten(start_dim = 1)


    def get_grad_pen(self, inputs, model_output):
        """Function that calculates the gradient penalty
        based on the gradients of the inputs, its L2 norm,
        applying the two sided penalty and the gradient
        penalty constant. Taken from Joost van Amersfoort
        paper on DUQ (2020).

        Args:
            inputs: Model inputs.
            model_output: Predicted labels given input.
        """
        # Gradient penalty constant, taken from DUQ paper
        gp_const = 0.5

        # First getting gradients
        gradients = self.get_gradients(inputs, model_output)

        # Then computing L2 norm (2 sided)
        L2_norm = gradients.norm(2, dim = 1)

        # Applying the 2 sided penalty
        grad_pen = ((L2_norm - 1) ** 2).mean()

        return grad_pen * gp_const


def cutoff_date(folder_name: str):
    """This function takes a folder in the form of a string.
    It cuts off the date and time from the end of the string.
    This function expects the folder name to not be in a folder.

    Args:
        folder_name: Name of the folder.
    """
    return os.path.normpath(folder_name).split(os.sep)[-1][:len(folder_name)-17]


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


def captum_explainability(model: torchvision.models, img_paths: list, option: str,
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


def deep_ensemble_uncertainty(experiment_name, results_path, ensemble_n: int = 5):
    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Defining the train transforms
    transform = get_transforms(args.dataset, args.augmentation)
    # Retrieving data loaders
    data_loaders = get_data_loaders(args.batch_size, args.shuffle, args.num_workers,
                                    transform, args.dataset)

    # Setting up tensorboard writers and writing hyperparameters
    tensorboard_writers, experiment_path = setup_tensorboard(experiment_name, "Experiment-Results")
    setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)

    print("Starting training phase")
    for n in range(ensemble_n):
        print("On ensemble run " + str(n))
        # Retrieving hyperparameter dictionary
        hyp_dict = setup_hyp_dict(experiment_name)
        args = SimpleNamespace(**hyp_dict)

        # Replacing the output classification layer with a N class version
        # And transferring model to device
        model = args.model
        classes = data_loaders["train"].dataset.n_classes
        model = set_classification_layer(model, classes, args.RBF_flag, device)
        model.to(device)

        model, c_labels, c_labels_pred = train_model(model, device, 
                                                     args.criterion, 
                                                     args.optimizer, 
                                                     args.scheduler,
                                                     data_loaders, 
                                                     tensorboard_writers,
                                                     args.epochs, 
                                                     args.PFM_flag,
                                                     args.RBF_flag, 
                                                     args.early_limit,
                                                     args.replacement_limit,
                                                     experiment_path,
                                                     classes)
        torch.save(model, os.path.join(experiment_path, "model.pth"))

        # Adding the confusion matrix of the last epoch to the tensorboard
        add_confusion_matrix(c_labels, c_labels_pred, 
                             tensorboard_writers["hyp"],
                             data_loaders["train"].dataset.label_map)

        # Closing tensorboard writers
        for _, writer in tensorboard_writers.items():
            writer.close()

    merge_experiments([sys.argv[1]], results_path)
    experiment_folders = os.listdir(os.path.join(results_path, experiment_name))
    test_loader = data_loaders["test"]
    predictions_per_model = []

    print("Starting testing phase")
    for experiment_folder in experiment_folders:
        print("On ensemble run " + str(n))

        # Loading the model from an experiment directory
        model = torch.load(os.path.join("Results", "Experiment-Results", 
                                        experiment_folder, "model.pth"), 
                           map_location = torch.device(device))
        img_destination = os.path.join("Results", "Test-Predictions", experiment_folder)

        # Getting test predictions
        prediction_list, _, _ = test_model(model, 
                                           device,
                                           test_loader,
                                           img_destination,
                                           args.rbf_flag)
        predictions_per_model.append(prediction_list)
    
    # TODO: predicted_labels needs to be combined into one list,
    #       with most frequent values being the final prediction.

    # TODO: predicted_uncertaitny needs to be the proportion of
    #       models that predict the final label for a datapoint.

    # After ensemble training and testing, save the test predictions
    # The labels are a combination of predicted labels by each model
    # The uncertainty is a measure of the agreement that exists
    # on the predicted label between the models
    img_destination = os.path.join("Results", "Explainability-Results",
                                   "ENS" + str(ensemble_n) + experiment_name)
    save_test_predicts(predicted_labels, img_paths, img_destination,
                       data_loaders["train"].dataset, predicted_uncertainty)


if __name__ == '__main__':
    # Explainability.py is runnable for either Captum explainability or
    # Deep Ensemble Uncertainty. The experiment argument takes both
    # the experiment folder if using Captum, or the experiment itself
    # if using DEU.
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type = str)
    parser.add_argument("explainability_method", choices = ["Captum", "DEU"], type = str)
    parser.add_argument("--n_ensembles", type = int, default = 5)
    parser.add_argument("--explainability_variant", type = str, default = "integrated_gradients",
                         choices = ["integrated_gradients", "saliency_map", "deeplift",
                                    "guided_backpropagation"])
    args = parser.parse_args() 
    
    if args.experiment == "Captum":
        if os.path.exists(os.path.join("Results", "Experiment-Results", args.experiment)):
            model, predicted_labels, img_paths, input_concat = setup_testing(args.experiment)
            # Cutting off part of the data, since too much causes CUDA memory issues
            if len(img_paths) > 100:
                img_paths = img_paths[:100]
                input_concat = input_concat[:100]
                predicted_labels = predicted_labels[:100]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Getting explainability results
            captum_explainability(model, img_paths, args.explainability_variant,
                                  device, input_concat, predicted_labels, 
                                  args.experiment)
    elif args.experiment == "DEU":
        if os.path.exists(os.path.join("Experiments", args.experiment)):
            results_path = os.path.join("Results", "Experiment-Results")
            deep_ensemble_uncertainty(args.experiment, results_path, args.n_ensembles) 
        else:
            print("Experiment not found, exiting ...")