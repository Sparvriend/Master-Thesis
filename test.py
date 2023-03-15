import argparse
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import sys
import time
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from NTZ_filter_dataset import NTZFilterDataset
from train_utils import convert_to_list, flatten_list, get_default_transform
from explainability import integrated_gradients
from torch2trt import torch2trt
import tensorrt as trt


def sep_test_collate(batch: list) -> tuple[torch.stack, list]:
    """Manual collate function for testing dataloader.
    It converts the images to a torch stack and returns the paths.

    Args:
        batch: batch of data items from a dataloader.
    Returns:
        images as torch stack and paths.
    """
    path, images, _ = zip(*batch)

    images = torch.stack(list(images), dim = 0)

    return images, path


def save_test_predicts(predicted_labels: list, paths: list):
    """Function that converts labels to a list and then saves paths and labels
    to appropriate prediction directories.

    Args:
        predicted_labels: list of tensors with predicted labels.
        paths: list of lists with paths (strings) to images.
    
    Returns:
        Prediction list and paths converted to correct format
    """
    prediction_list = convert_to_list(predicted_labels)
    paths = flatten_list(paths)

    # Dictionary for the labels to use in saving
    label_dict = {0: "fail_label_crooked_print", 1: "fail_label_half_printed",
                  2: "fail_label_not_fully_printed", 3: "no_fail"}	
    prediction_dir = os.path.join("data", "test_predictions")

    # Loading necessary information and then drawing on the label on each image.
    for idx, path in enumerate(paths):
        name = os.path.normpath(path).split(os.sep)[-1]
        img = Image.open(path)
        label_name = label_dict[prediction_list[idx]]

        # Drawing the label and saving the image
        # font_loc = os.path.join("C:", "Windows", "Fonts", "arial.ttf")
        font = ImageFont.truetype(os.path.join("data", "arial.ttf"), size = 18)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label_name, font = font, fill = (255, 0, 0))
        img.save(os.path.join(prediction_dir, name))

    return prediction_list, paths


def remove_predicts():
    """Function that removes all old predictions from
    the test_predictions folder.
    """
    # Path information
    path = os.path.join("data", "test_predictions")
    if os.path.exists(path):
        files = [f for f in os.listdir(path) 
                if os.path.isfile(os.path.join(path, f))]
        for old_file in files:
            file_path = os.path.join(path, old_file)
            os.unlink(file_path)
    else:
        os.mkdir(path)


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


def test_model(model: torchvision.models, device: torch.device, data_loader: DataLoader):
    """Function that tests the feature extractor model on the test dataset.
    It runs through a forward pass to get the model output and saves the
    output images to appropriate directories through the save_test_predicts
    function.

    Args:
        model: The model to test.
        device: The device which data/model is present on.
        data_loader: The data loader contains the data to test on.
    Returns:
        Lists of predicted labels and the image paths. 
    """
    # Set model to evaluating, set speed measurement variable
    model.eval()
    total_imgs = 0

    # Creating a list of paths and predicted labels
    # and starting the timer
    predicted_labels = []
    img_paths = []
    test_start = time.time()

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

    # Saving the test predictions, getting the testing time and
    # printing the fps
    testing_time = time.time() - test_start
    print("FPS = " + str(round(total_imgs / testing_time, 2)))
    prediction_list, img_paths_list = save_test_predicts(predicted_labels, img_paths)

    return prediction_list, img_paths_list


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
    # First removing all old predictions that might be present
    # in the prediction folders
    remove_predicts()

    # Setting the device to use and enabling lazy loading
    torch.cuda._lazy_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Setting test path and creating transform. The first entry is a lambda
    # dummy function, because it is cutoff in NTZFilterDataset. Should be
    # fixed in the future.
    test_path = os.path.join("data", "test")
    transform = get_default_transform()
    transform.transforms.insert(0, T.Lambda(lambda x: x))

    # Loading the model form a experiment directory
    model = torch.load(os.path.join("Master-Thesis-Experiments", 
                                     experiment_folder, "model.pth"))
    # ShuffleNet causes an error with the variable batch size
    # Hence setting it to 1 to fix that
    batch_size = 16
    if model.__class__.__name__ == "ShuffleNetV2":
        batch_size = 1

    # Creating the dataset and transferring to a DataLoader
    test_data = NTZFilterDataset(test_path, transform)
    test_loader = DataLoader(test_data, batch_size = batch_size,
                             collate_fn = sep_test_collate, shuffle = True, num_workers = 4)

    # Optionally, port the model to TRT version
    # PyTorch model -> ONNX model -> TensorRT model (Optimized model for GPU)
    # Takes ~30 seconds for MobileNetV2 - ~5 mins for EfficientNetB1
    if convert_trt:
        model = convert_to_trt(model, len(test_data), batch_size)

    # Testing the model on testing data
    predicted_labels, img_paths = test_model(model, device, test_loader)

    # Optionally, explain the model using integrated gradients
    if explain_model:
        integrated_gradients(model, test_loader, predicted_labels, img_paths)

if __name__ == '__main__':
    # Forming argparser that takes input arguments, with optional
    # booleans for convert_trt and explain_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", type = str, required = True)
    parser.add_argument("--convert_trt", action = argparse.BooleanOptionalAction)
    parser.add_argument("--explain_model", action = argparse.BooleanOptionalAction)
    args = parser.parse_args()
    setup_testing(args.exp_folder, args.convert_trt, args.explain_model)