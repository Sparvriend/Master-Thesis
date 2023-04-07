import argparse
import json
import os
import sys
import tensorrt as trt
import time
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from utils import get_data_loaders, cutoff_date, save_test_predicts, \
                  remove_predicts    
from explainability import explainability_setup

from datasets import NTZFilterDataset, CIFAR10Dataset, TinyImageNet200Dataset

try:
    from torch2trt import torch2trt
except ModuleNotFoundError:
    print("Could not import torch2trt, model conversion to trt will not work")


def convert_to_trt(model: torchvision.models, data_len: int, batch_size: int):
    """This function takes a PyTorch model and converts it to a TensorRT model.
    It creates a config file with a custom profile since the batch size.

    Args:
        model: Pytorch model to convert
        data_len: The length of the testing set.
        batch_size: The batch size of the testing set.
    Returns:
        TensorRT converted model.
    """
    # Using a trt builder to create a config and a profile
    # Which is necessary because the batch size is variable
    # It can change at the end, if the dataset size is not divisible by the batch size
    norm_shape = (batch_size, 3, 224, 224)
    builder = trt.Builder(trt.Logger())
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    # Creating the profile shape and adding to config
    profile.set_shape("input_tensor", min = (data_len % batch_size, 3, 224, 224),
                       max = norm_shape, opt = norm_shape)
    config.add_optimization_profile(profile)
    
    # Converting model to tensorRT with the builder, config and the profile
    model = torch2trt(model, [torch.ones(norm_shape).cuda()], fp16_mode = True,
                      max_batch_size = batch_size, builder = builder, config = config)
    return model


def test_model(model: torchvision.models, device: torch.device, data_loader: DataLoader,
               img_destination: str):
    """Function that tests the feature model on the test dataset.
    It runs through a forward pass to get the model output and saves the
    output images to appropriate directories through the save_test_predicts
    function.

    Args:
        model: The model to test.
        device: The device which data/model is present on.
        data_loader: The data loader contains the data to test on.
        img_destination: Designated folder to save images to.
    Returns:
        Lists of predicted labels and the image paths.
        Concatenated inputs.
    """
    # Set model to evaluating, set speed measurement variable
    model.eval()
    total_imgs = 0

    # Creating a list of paths and predicted labels
    # and starting the timer
    predicted_labels = []
    img_paths = []
    test_start = time.time()
    input_concat = torch.empty((0,)).to(device)

    print("Testing phase")
    with torch.no_grad():
        for inputs, paths in tqdm(data_loader):
            inputs = inputs.to(device)

            # Getting model output and adding labels/paths to lists
            model_output = model(inputs)
            predicted_labels.append(model_output.argmax(dim=1))
            img_paths.append(paths)
    
            # Counting up total amount of images a prediction was made over
            total_imgs += len(inputs)

            # Concatenating the inputs for later use in explainability.py
            input_concat = torch.cat((input_concat, inputs))

    # Saving the test predictions, getting the testing time and
    # printing the fps
    testing_time = time.time() - test_start
    print("FPS = " + str(round(total_imgs / testing_time, 2)))
    prediction_list, img_paths_list = save_test_predicts(predicted_labels, img_paths,
                                                         img_destination, data_loader.dataset)

    return prediction_list, img_paths_list, input_concat


def setup_testing(experiment_folder: str, convert_trt: bool = False, 
                  explain_model: bool = False):
    """Function that sets up dataloader, transforms and loads in the model
    for testing. 

    Args: 
        experiment_folder: The folder with a model.pth file
        convert_trt: If true, the model is converted to tensorRT.
    Returns:
        The model, the dataloader and the device.
        For usage in explainability.py
    """
    # Setting the device to use and enabling lazy loading
    try: 
        torch.cuda._lazy_init()
    except RuntimeError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Loading the model form a experiment directory
    # Map location because if CPU, it otherwise causes issues
    model = torch.load(os.path.join("Results", "Experiment-Results", 
                                     experiment_folder, "model.pth"), 
                                     map_location = torch.device(device))
    
    # Getting experiment_name and creating the folder to paste the images in
    experiment_name = cutoff_date(experiment_folder)
    img_destination = os.path.join("Results", "Test-Predictions", experiment_name)

    # Removing all old predictions that might be present
    # in the prediction folder
    remove_predicts(img_destination)
    
    # ShuffleNet causes an error with the variable batch size
    # Hence setting it to 1 to fix that
    # A batch size of 1 would more accurately represent the
    # Real assembly line at NTZ
    batch_size = 16
    if model.__class__.__name__ == "ShuffleNetV2":
        batch_size = 1

    # Setting the type of dataset for testing
    experiment_location = os.path.join("Experiments", experiment_name + ".json")
    with open(experiment_location, "r") as file:
        dataset = eval(json.load(file)["Dataset"]) 

    # Creating the dataset and transferring to a DataLoader
    test_loader = get_data_loaders(batch_size, dataset = dataset)["test"]

    # Optionally, port the model to TRT version
    # PyTorch model -> ONNX model -> TensorRT model (Optimized model for GPU)
    # Takes ~30 seconds for MobileNetV2 - ~5 mins for EfficientNetB1
    if convert_trt:
        model = convert_to_trt(model, len(test_loader.dataset), batch_size)

    # Testing the model on testing data
    predicted_labels, img_paths, input_concat = test_model(model, device, test_loader,
                                                           img_destination)

    # Optionally, explain the model using integrated gradients
    # Reduce the amount of images to explain, since doing it with too many
    # Causes CUDA out of memory errors
    if explain_model:
        if len(img_paths) > 100:
            img_paths = img_paths[:100]
            input_concat = input_concat[:100]
            predicted_labels = predicted_labels[:100]
        explainability_setup(model, img_paths, "guided_backpropagation", device,
                             input_concat, predicted_labels, experiment_folder)

if __name__ == '__main__':
    # Checking if required folder exists
    if len(sys.argv) > 1:
        if os.path.exists(os.path.join("Results", "Experiment-Results", sys.argv[1])):
            print("Testing on model from experiment: " + sys.argv[1])
        else:
            print("Experiment not found, exiting ...")

    # Forming argparser that takes input arguments, with optional
    # booleans for convert_trt and explain_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert_trt", action = argparse.BooleanOptionalAction)
    parser.add_argument("--explain_model", action = argparse.BooleanOptionalAction)
    args = parser.parse_args(sys.argv[2:])
    setup_testing(sys.argv[1], args.convert_trt, args.explain_model)