import captum
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as T
import warnings


class RBF_model(nn.Module):
    """RBF layer definition taken from Joost van Amersfoort's implementation of DUQ:
    https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/resnet_duq.py
    and inspired by Matias Valdenegro Toro implementation:
    https://github.com/mvaldenegro/keras-uncertainty/blob/master/keras_uncertainty/layers/rbf_layers.py

    Variable explanations:
    kernels = ?. Shape = [Fe out features, classes, fe out features]
    N = ?. Shape = [Classes]
    m = ?. Shape = [fe out features, classes]

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
        self.kernels = nn.Parameter(torch.Tensor(in_features, out_features, in_features))
        self.N = (torch.zeros(out_features) + 13).to(device) # (Why did Joost set this value to 13?)
        self.m = torch.zeros(in_features, out_features).to(device)
        self.m *= self.N

        nn.init.normal_(self.m, 0.05, 1)
        nn.init.kaiming_normal_(self.kernels, nonlinearity = 'relu')


    def forward(self, x):
        # Getting feature output from fe and then applying kernels
        z = self.fe(x)
        z = torch.einsum("ij, mnj->imn", z, self.kernels)

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
        z = self.fe(inputs)
        z = torch.einsum("ij, mnj->imn", z, self.kernels)
        labels = labels.unsqueeze(1).cpu()
        z = z.type(torch.LongTensor)
        centroid_sum = torch.einsum("ijk, il->jk", z, labels).to(self.device)

        # Update m here
        self.m = update_f(self.m, centroid_sum)


def get_gradients(inputs, model_output):
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


def get_grad_pen(inputs, model_output):
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
    gradients = get_gradients(inputs, model_output)

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


def deep_ensemble_uncertainty():
   # Hardest to implement? -> No clear library available
   # Testing module, but can not import test_model due to circular imports
    print("Not yet implemented")