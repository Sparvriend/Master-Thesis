import captum
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models
import torchvision.transforms as T
import warnings


def visualize_explainability(img_data: torch.Tensor, img_paths: list):
    """Function that visualizes an explainability result.
    It combines the original image with image data that explains
    the decision making of the model.

    Args:
        img_data: torch tensor with image data.
        img_paths: list of paths to the original images.
    """
    # Custom transform for only resizing and then cropping to center
    transform = T.Compose([T.Resize(256, max_size = 320), T.CenterCrop(224)])

    # Creating custom colormap
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')],
                                                     N = 256)
    #rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#2E2B5F', '#8B00FF']
    #default_cmap = ListedColormap(rainbow_colors)
    
    # Iterating over all image data and saving 
    # them in combination with the original image
    for i, img in enumerate(img_data):
        # Retrieving the image, with the label printed on it
        img_path = os.path.join("data", "test_predictions",
                                 os.path.normpath(img_paths[i]).split(os.sep)[-1]) 
        norm_img = Image.open(img_path)
        norm_img = transform(norm_img)
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                   np.asarray(norm_img),
                                                   methods = ["original_image", "heat_map"],
                                                   signs = ["all", "positive"],
                                                   cmap = default_cmap,
                                                   show_colorbar = True)
        img_name = os.path.normpath(img_paths[i]).split(os.sep)[-1]
        fig.savefig(os.path.join("Explainability-Results", img_name.replace(".bmp", ".png")))
        plt.close()


def explainability_setup(model: torchvision.models, img_paths: list, option: str,
                         device: torch.device, input_concat: torch.Tensor,
                         predicted_labels: list):
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
    """
    # Ignoring UserWarnings, since they are not helpful
    # 1st warning is about requireing grad which it then sets
    # 2nd warning is about setting backward hooks for ReLu activations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Based on which explainability option is selected
        # Arguments are created and passed to the explainability function
        if option == "integrated_gradients":
            args = {"inputs": input_concat.to(device), "target": predicted_labels,
                    "internal_batch_size": len(predicted_labels), "n_steps": 200}
            print("Running integrated gradients for model explanation")
            gen_model_explainability(captum.attr.IntegratedGradients, model,
                                    img_paths, args)
        elif option == "saliency_map":
            args = {"inputs": input_concat.to(device), "target": predicted_labels}
            print("Running saliency map for model explanation")
            gen_model_explainability(captum.attr.Saliency, model,
                                    img_paths, args)
        elif option == "deeplift":
            args = {"inputs": input_concat.to(device), "target": predicted_labels}
            print("Running deeplift for model explanation")
            gen_model_explainability(captum.attr.DeepLift, model,
                                    img_paths, args)
        elif option == "guided_backpropagation":
            args = {"inputs": input_concat.to(device), "target": predicted_labels}
            print("Running guided backpropagation for model explanation")
            gen_model_explainability(captum.attr.GuidedBackprop, model,
                                    img_paths, args)
        else:
            print("Explainability option not valid")


def gen_model_explainability(explain_func, model: torchvision.models,
                             img_paths: list, args: list):
    """Function that generates an explainability object,
    which is basically a significance value per pixel for each image,
    and passes it to the visualization function
    
    Args:
        explain_func: function that generates the explainability object.
        model: model to explain.
        img_paths: list of paths to the original images.
        args: list of arguments for the explainability function.
    """
    # Creating explainability object via Captum
    explainability_obj = explain_func(model)
    # Unpacking dictionary arguments with **
    explainability_attr = explainability_obj.attribute(**args)
    visualize_explainability(explainability_attr, img_paths)


def deep_uncertainty_quantification():
    # Average centroids need to be calculated during training
    # The function here can then use these centroids to calculate
    # Uncertainty by distance.
    print("Not yet implemented")


def deep_ensemble_uncertainty():
   # Hardest to implement? -> No clear library available
   # During training time/testing time.
    print("Not yet implemented")