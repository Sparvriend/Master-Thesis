import copy
import captum
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
import torch
import torchvision.models
import torchvision.transforms as T
from tqdm import tqdm
import warnings

from utils import cutoff_date, flatten_list, get_data_loaders, \
                  save_test_predicts, remove_predicts, cutoff_classification_layer


def visualize_explainability(img_data: torch.Tensor, img_paths: list, img_desintation: str):
    """Function that visualizes an explainability result.
    It combines the original image with image data that explains
    the decision making of the model.

    Args:
        img_data: torch tensor with image data.
        img_paths: list of paths to the original images.
        img_desintation: path to folder to save the images in.
    """
    # Custom transform for only resizing and then cropping to center
    transform = T.Compose([T.Resize(256, max_size = 320), T.CenterCrop(224)])

    # Creating custom colormaps
    bw_cmap = LinearSegmentedColormap.from_list("custom bw",
                                                [(0, '#ffffff'),
                                                 (0.25, '#000000'),
                                                 (1, '#000000')],
                                                N = 256)
    
    # Iterating over all image data and saving 
    # them in combination with the original image
    for i, img in enumerate(img_data):
        # Retrieving the image, with the label printed on it
        img_path = os.path.join("Results", "Test-Predictions",
                                 os.path.normpath(img_paths[i]).split(os.sep)[-1]) 
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
        fig.savefig(os.path.join(img_desintation, img_name.replace(".bmp", ".png")))
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
    # Setting up the explainability results folder
    if not os.path.exists(os.path.join("Results", "Explainability-Results")):
        os.mkdir(os.path.join("Results", "Explainability-Results"))

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
        if option == "integrated_gradients":
            args = {"inputs": input_concat.to(device), "target": predicted_labels,
                    "internal_batch_size": len(predicted_labels), "n_steps": 200}
            print("Running integrated gradients for model explanation")
            gen_model_explainability(captum.attr.IntegratedGradients, model,
                                    img_paths, args, img_desintation)
        elif option == "saliency_map":
            args = {"inputs": input_concat.to(device), "target": predicted_labels}
            print("Running saliency map for model explanation")
            gen_model_explainability(captum.attr.Saliency, model,
                                    img_paths, args, img_desintation)
        elif option == "deeplift":
            args = {"inputs": input_concat.to(device), "target": predicted_labels}
            print("Running deeplift for model explanation")
            gen_model_explainability(captum.attr.DeepLift, model,
                                    img_paths, args, img_desintation)
        elif option == "guided_backpropagation":
            args = {"inputs": input_concat.to(device), "target": predicted_labels}
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


def compare_feature_maps(feature_map: torch.Tensor, class_centroids: dict,
                         predicted_labels: list, distances: list) -> tuple[list, list]:
    """Function that compares the feature map of an image to the class
    centroids. It calculates the euclidean distance to the closest class
    centroid which is the label prediction, the distance is the uncertainty.
    
    Args:
        feature_map: feature map of the image.
        class_centroids: dictionary with class centroids.
        predicted_labels: list of predicted labels.
        distances: list of distances to the class centroids.
    Returns:
        The list of predicted labels and distances for 
        the feature map comparison
    """
    centroid_dist = []
    for _, class_centroid in class_centroids.items():
        # Convert to cpu, since that is what numpy requires
        feature_map = feature_map.cpu()
        class_centroid = class_centroid.cpu()
        # Calculating the euclidean distance between the two feature maps
        centroid_dist.append(np.linalg.norm(feature_map - class_centroid))
    predicted_labels.append(centroid_dist.index(min(centroid_dist)))
    distances.append(centroid_dist)

    return predicted_labels, distances

def deep_uncertainty_quantification(experiment_name: str):
    """Function that calculates deep uncertainty based on
    radial basis function (RBF). It calculates uncertainty
    based on the distance to average centroids. It is a
    seperate testing module, since it predicts its own labels.

    Args:
        experiment_name: name of the model to run DUQ on.
    """
    # Setting up the device to run the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Setting batch size and retrieving data loaders
    batch_size = 16
    data_loaders = get_data_loaders(batch_size)
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    # Loading model and defining experiment name
    model = torch.load(os.path.join("Results", "Experiment-Results",
                                     experiment_name, "model.pth"), 
                                     map_location = torch.device(device))
    experiment_name = cutoff_date(experiment_name)
    
    # Making a copy of the model with the classification layer cut off
    # To form a model that outputs a feature map instead of a a class
    feature_extractor = copy.deepcopy(model)
    feature_extractor = cutoff_classification_layer(feature_extractor)

    # Set model to evaluating
    model.eval()
    feature_extractor.eval()

    # Creating a class dict for storing feature maps per class
    class_dict = {0: [], 1: [], 2: [], 3: []}

    # Running model/feature extraction prediction on validation set
    print("Calculating average centroids over validation data")
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader):
            inputs = inputs.to(device)

            # Getting model output for the feature maps and the labels
            model_output = model(inputs)
            feature_maps = feature_extractor(inputs)
            predicted_labels = model_output.argmax(dim=1)
            
            # Looping over feature maps and labels
            for idx, feature_map in enumerate(feature_maps):
                class_dict[predicted_labels[idx].item()].append(feature_map)

    # Calculating average feature maps for the predictions
    class_centroids = {0: [], 1: [], 2: [], 3: []}
    for c_label, feature_maps in class_dict.items():
        summed_feature_maps = torch.zeros(feature_maps[0].shape).to(device)
        for feature_map in feature_maps:
            summed_feature_maps += feature_map
        class_centroids[c_label] = summed_feature_maps / len(feature_maps)
    
    # Running model/feature extraction prediction on testing set
    # The closest centroid is the predicted class of the model
    distances = []
    predicted_labels = []
    img_paths = []
    print("Comparing test data to average centroids")
    with torch.no_grad():
        for inputs, paths in tqdm(test_loader):
            inputs = inputs.to(device)

            # Getting feature maps and saving image paths
            feature_maps = feature_extractor(inputs)
            img_paths.append(paths)
            
            # Comparing each feature map to class centroids
            for feature_map in feature_maps:
                predicted_labels, distances = compare_feature_maps(feature_map, 
                                                                   class_centroids,
                                                                   predicted_labels,
                                                                   distances)
    # Basing the maximum distance on a difference of 1 in each feature
    # of the feature map. With the minimum distance being 0
    max_distance = np.prod(feature_map.shape)

    # Calculating the uncertainty, based on the maximum distance
    # This is not a good way to calculate uncertainty,
    # The distances range from 400 when it is not the right class
    # (Even when inserting images that are nothing like the dataset)
    # and 70-140 when it is the right class, but comparing 400 to 
    # 62720 does not give any indicative metric.
    img_destination = os.path.join("Results", "Explainability-Results", "DUQ-" +
                                    experiment_name)
    remove_predicts(img_destination)
    save_test_predicts(predicted_labels, img_paths, img_destination)
    img_paths = flatten_list(img_paths)
    with open(os.path.join(img_destination, "results.txt"), "a") as file:
        for idx, distance_list in enumerate(distances):
            uncertainty = 1 - (min(distance_list) / max_distance)
            file.write("distance list = " + str(distance_list) + "\n")
            file.write("Image = " + img_paths[idx] + "\n")
            file.write("Predicted label = " + str(predicted_labels[idx]) + "\n")
            file.write("Uncertainty = " + str(uncertainty) + "\n\n")
    file.close()


def deep_ensemble_uncertainty():
   # Hardest to implement? -> No clear library available
   # Testing module, but can not import test_model due to circular imports
    print("Not yet implemented")

if __name__ == '__main__':
    # Running DUQ on a model that has been through training phase
    if len(sys.argv) == 2 and os.path.exists(os.path.join("Results", "Experiment-Results", sys.argv[1])):
        deep_uncertainty_quantification(sys.argv[1])
    else:
        print("No valid experiment name given, exiting ...")