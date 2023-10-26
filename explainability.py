import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.models
import torchvision.transforms as T
import warnings

from PIL import Image
from captum.attr import IntegratedGradients, Saliency, DeepLift, GuidedBackprop
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset
from types import SimpleNamespace

from test import setup_testing, test_model
from train import train_model
from train_rbf import RBF_model
from utils import get_transforms, get_data_loaders, setup_tensorboard, \
                  setup_hyp_file, set_classification_layer, \
                  add_confusion_matrix, setup_hyp_dict, merge_experiments, \
                  save_test_predicts, remove_predicts, get_device, \
                  draw_uncertainty_bar, draw_label, get_text_loc, \
                  convert_to_list


def visualize_explainability(img_data: torch.Tensor, img_paths: list, img_destination: str,
                             predictions: dict, dataset: Dataset):
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

    # Creating custom colormaps
    bw_cmap = LinearSegmentedColormap.from_list("custom bw",
                                                [(0, "#ffffff"),
                                                 (0.25, "#000000"),
                                                 (1, "#000000")],
                                                N = 256)
    
    # Converting dict to variables
    predicted_labels = predictions["label_list"]
    predicted_uncertainty = predictions["uncertainty_list"]

    if len(predicted_uncertainty) != 0 and type(predicted_uncertainty[0]) == torch.Tensor:
        predicted_uncertainty = convert_to_list(predicted_uncertainty)

    # Setting label text size and text location
    ex_width, ex_height = Image.open(img_paths[0]).size
    text_size = int(((ex_height + ex_width) / 2)  / 16)
    text_loc = get_text_loc(dataset)
    
    # Getting the dataset label map
    label_dict = dataset.label_map
    
    # Iterating over all image data and saving 
    # them in combination with the original image
    for i, img in enumerate(img_data):
        # Retrieving the image, transform into 224x224 version
        norm_img = Image.open(img_paths[i])
        norm_img = transform(norm_img)

        # Print the label onto it
        label_name = label_dict[predicted_labels[i]]
        norm_img = draw_label(norm_img, text_loc, text_size, label_name)

        # And add uncertainty if it exists
        if len(predicted_uncertainty) != 0:
            norm_img = draw_uncertainty_bar(norm_img, predicted_uncertainty[i], text_size)

        # Draw images along side each other
        # plt.cm.inferno for the cmap looks a lot better, but it looks worse in the report,
        # hence not using it.
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


def captum_explainability(model: torchvision.models, option: str,
                          device: torch.device, input_concat: torch.Tensor,
                          predicted_labels: list, experiment_folder: str):
    """Function that generates function arguments for explainability.
    Since Captum has a very similar way of running the explainability functions,
    doing it like this is a nice option.

    Args:
        model: model to explain.
        img_paths: list of paths to the original images.
        option: name of the explainability method.
        device: device to run the model on.
        input_concat: concatenated input tensor.
        predicted_labels: list of predicted labels.
        experiment_folder: path to the experiment folder that was tested on.
    Returns:
        explainability object.
    """
    # Getting experiment_name and creating the folder to paste the images in
    img_desintation = os.path.join("Results", "Explainability-Results", experiment_folder)
    if not os.path.exists(img_desintation):
        os.mkdir(img_desintation)

    # Ignoring UserWarnings, since they are not helpful
    # 1st warning is about requireing grad which it then sets
    # 2nd warning is about setting backward hooks for ReLu activations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category = UserWarning)
        # Based on which explainability option is selected
        # Arguments are created and passed to the explainability function
        args = {"inputs": input_concat.to(device), "target": predicted_labels} 
        if option == "integrated_gradients":
            args["internal_batch_size"] = len(predicted_labels)
            args["n_steps"] = 200
            print("Running integrated gradients for model explanation")
            explainability_attr = gen_model_explainability(IntegratedGradients, model, args)
        elif option == "saliency_map":
            print("Running saliency map for model explanation")
            explainability_attr = gen_model_explainability(Saliency, model, args)
        elif option == "deeplift":
            print("Running deeplift for model explanation")
            explainability_attr = gen_model_explainability(DeepLift, model, args)
        elif option == "guided_backpropagation":
            print("Running guided backpropagation for model explanation")
            explainability_attr = gen_model_explainability(GuidedBackprop, model, args)
    return explainability_attr


def gen_model_explainability(explain_func, model: torchvision.models, args: list):
    """Function that generates an explainability object,
    which is basically a significance value per pixel for each image,
    and passes it to the visualization function
    
    Args:
        explain_func: function that generates the explainability object.
        model: model to explain.
        img_paths: list of paths to the original images.
        args: list of arguments for the explainability function.
        img_desintation: path to the folder where the images are saved.
    Returns:
        explainability object.
    """
    # Creating explainability object via Captum
    explainability_obj = explain_func(model)
    # Unpacking dictionary arguments with **
    explainability_attr = explainability_obj.attribute(**args)
    return explainability_attr
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type = str)
    parser.add_argument("--explainability_variant", type = str, default = "integrated_gradients",
                         choices = ["integrated_gradients", "saliency_map", "deeplift",
                                    "guided_backpropagation"])
    args = parser.parse_args() 
    
    if os.path.exists(os.path.join("Results", "Experiment-Results", args.experiment)):
        model, predicted_labels, img_paths, input_concat, predicted_uncertainty, dataset = setup_testing(args.experiment)

        # Cutting off part of the data, since too much causes CUDA memory issues
        max_imgs = 100
        if len(img_paths) > max_imgs:
            img_paths = img_paths[:max_imgs]
            input_concat = input_concat[:max_imgs]
            predicted_labels = predicted_labels[:max_imgs]
            predicted_uncertainty = predicted_uncertainty[:max_imgs]
        predictions = {"label_list": predicted_labels, "uncertainty_list": predicted_uncertainty}
        device = get_device()

        # Getting explainability results
        img_data = captum_explainability(model, args.explainability_variant,
                                         device, input_concat, predicted_labels, 
                                         args.experiment)
        
        # Forming image destination
        img_destination = os.path.join("Results", "Explainability-Results", args.experiment)
        if not os.path.exists(img_destination):
            os.mkdir(img_destination)

        # And visualizing results
        visualize_explainability(img_data, img_paths, img_destination, predictions, dataset)

    else:
        print("Experiment results not found, exiting ...")