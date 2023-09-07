import argparse
import json
import os
import tensorrt as trt
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from utils import get_data_loaders, save_test_predicts, remove_predicts, \
                  cutoff_date, get_device    
from datasets import NTZFilterDataset, NTZFilterSyntheticDataset, \
                     CIFAR10Dataset, TinyImageNet200Dataset
from train_rbf import RBF_model

TRTFLAG = True
try:
    from torch2trt import torch2trt
except ModuleNotFoundError:
    TRTFLAG = False


def convert_to_trt(model: torchvision.models, data_len: int, batch_size: int, img_size: int):
    """This function takes a PyTorch model and converts it to a TensorRT model.
    It creates a config file with a custom profile since the batch size can be variable

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
    norm_shape = (batch_size, 3, img_size, img_size)
    builder = trt.Builder(trt.Logger())
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    # Creating the profile shape and adding to config
    profile.set_shape("input_tensor", min = (data_len % batch_size, 3, img_size, img_size),
                       max = norm_shape, opt = norm_shape)
    config.add_optimization_profile(profile)
    
    # Converting model to tensorRT with the builder, config and the profile
    model = torch2trt(model, [torch.ones(norm_shape).cuda()], fp16_mode = True,
                      max_batch_size = batch_size, builder = builder, config = config)
    return model


def get_inference_speed(model: torchvision.models, device: torch.device, data_loader: DataLoader, n: int):
    """This function calculates the inference speed of a model by 
    doing n+1 forwards passes and calculating the time it takes
    to classify all images in the dataset.
    
    Args:
        model: The model to test on
        device: The device which data/model is present on.
        data_loader: The data loader contains the data.
        n: Amount of runs to do.
    """
    fps = []
    for i in range(n+1):
        print("Inference run " + str(i))
        test_start = time.time()
        total_imgs = 0
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                model(inputs)
                total_imgs += len(inputs)
        # Skip the first run, since its always slow
        if i != 0:
            fps.append(round(total_imgs / (time.time() - test_start)))
    print("Average inference fps = " + str(np.mean(fps)))
    print("Standard deviation = " + str(np.std(fps)))
    return model


def test_model(model: torchvision.models, device: torch.device, data_loader: DataLoader,
               img_destination: str, rbf_flag: bool):
    """Function that tests the feature model on the test dataset.
    It runs through a forward pass to get the model output and saves the
    output images to appropriate directories through the save_test_predicts
    function.

    Args:
        model: The model to test.
        device: The device which data/model is present on.
        data_loader: The data loader contains the data to test on.
        img_destination: Designated  folder to save images to.
        rbf_flag: If true, the model is a RBF model.
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
    predicted_uncertainty = []
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

            if rbf_flag == True:
                predicted_uncertainty.append(model_output.max(dim=1)[0])
    
            # Counting up total amount of images a prediction was made over
            total_imgs += len(inputs)

            # Concatenating the inputs for later use in explainability.py
            input_concat = torch.cat((input_concat, inputs))

    # Saving the test predictions, getting the testing time and
    # printing the fps
    testing_time = time.time() - test_start
    print("FPS = " + str(round(total_imgs / testing_time, 2)))
    prediction_list, img_paths_list = save_test_predicts(predicted_labels, img_paths,
                                                         img_destination, data_loader.dataset,
                                                         predicted_uncertainty)

    return prediction_list, img_paths_list, input_concat, predicted_uncertainty


def setup_testing(experiment_folder: str, convert_trt: bool = False, calc_speed: bool = False):
    """Function that sets up dataloader, transforms and loads in the model
    for testing. 

    Args: 
        experiment_folder: The folder with a model.pth file
        convert_trt: If true, the model is converted to tensorRT.
        explain_model: Contains a string with the model explanation type
    Returns:
        The model, the dataloader and the device.
        For usage in explainability.py
    """
    # Setting the device to use and enabling lazy loading
    try: 
        torch.cuda._lazy_init()
    except RuntimeError:
        pass
    device = get_device()

    # Loading the model from an experiment directory
    # Map location because if CPU, it otherwise causes issues
    model = torch.load(os.path.join("Results", "Experiment-Results", 
                                     experiment_folder, "model.pth"), 
                                     map_location = torch.device(device))
    
    # Getting experiment_name and creating the folder to paste the images in
    experiment_name = cutoff_date(experiment_folder)
    experiment_name = experiment_name.split("_")[0] + "_" + experiment_name.split("_")[1]
    for json_file in os.listdir("Experiments"):
        if json_file.startswith(experiment_name):
            experiment_name = json_file
    img_destination = os.path.join("Results", "Test-Predictions", experiment_folder)

    # Removing all old predictions that might be present
    # in the prediction folder
    remove_predicts(img_destination)
    
    # ShuffleNet causes an error with the variable batch size
    # Hence setting it to 1 to fix that
    batch_size = 16
    if model.__class__.__name__ == "ShuffleNetV2":
        batch_size = 1
    if calc_speed == True:
        # Batch size of 1 better resembles the pipeline at NTZ
        batch_size = 1

    # Taking default parameters from default experiment
    experiment_location = os.path.join("Experiments", "DEFAULT.json")
    with open(experiment_location, "r") as f:
        def_dict = json.load(f)

    # Setting the type of dataset for testing
    experiment_location = os.path.join("Experiments", experiment_name)
    with open(experiment_location, "r") as f:
        exp_dict = json.load(f)
    
    # Copying default parameters to experiment
    hyp_dict = def_dict.copy()
    hyp_dict.update(exp_dict)

    # Getting dataset and default values
    dataset = eval(hyp_dict["dataset"])
    rbf_flag = eval(hyp_dict["RBF_flag"])

    # Creating the dataset and transferring to a DataLoader
    test_loader = get_data_loaders(batch_size, dataset = dataset)["test"]

    # Depending on the dataset, the sizes of images are variable
    if dataset.__name__ == "NTZFilterDataset" \
    or dataset.__name__ == "NTZFilterSyntheticDataset":
        img_size = 224
    elif dataset.__name__ == "CIFAR10Dataset":
        img_size = 32
    elif dataset.__name__ == "TinyImageNet200Dataset":
        img_size = 64
    elif dataset.__name__ == "ImageNet10Dataset":
        img_size = 256

    # Optionally, port the model to TRT version
    # PyTorch model -> ONNX model -> TensorRT model (Optimized model for GPU)
    # Takes ~30 seconds for MobileNetV2 - ~5 mins for EfficientNetB1
    if convert_trt:
        if TRTFLAG == False:
            print("torch2trt library not on device, skipping trt conversion")
        else:
            print("Converting model to trt model")
            model = convert_to_trt(model, len(test_loader.dataset), batch_size, img_size)

    # Testing the model on testing data
    if calc_speed:
        # The amount of runs is high, since the forward passes can be unstable
        get_inference_speed(model, device, test_loader, 100)
    predicted_labels, img_paths, input_concat, predicted_uncertainty = test_model(model,
                                                                                  device,
                                                                                  test_loader,
                                                                                  img_destination,
                                                                                  rbf_flag)
    return model, predicted_labels, img_paths, input_concat, predicted_uncertainty, test_loader.dataset


if __name__ == '__main__':
    # Forming argparser with experiment folder and optional
    # convert to trt flag
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder", type = str)
    parser.add_argument("--convert_trt", action = "store_true")
    parser.add_argument("--calc_speed", action = "store_true")
    args = parser.parse_args()
    if os.path.exists(os.path.join("Results", "Experiment-Results",
                                    args.experiment_folder)):
        print("Testing on model from experiment: " + args.experiment_folder)
        setup_testing(args.experiment_folder, args.convert_trt, args.calc_speed)
    else:
        print("Experiment not found, exiting ...")