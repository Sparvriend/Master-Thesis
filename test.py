import os
from os import listdir
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
from train_utils import convert_to_list, flatten_list


def sep_test_collate(batch: list) -> tuple[torch.stack, list]:
    """ Manual collate function for testing dataloader.
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


def remove_predicts():
    """Function that removes all old predictions from
    the test_predictions folder.
    """
    # Path information
    path = os.path.join("data", "test_predictions")
    if os.path.exists(path):
        files = [f for f in listdir(path) 
                if os.path.isfile(os.path.join(path, f))]
        for old_file in files:
            file_path = os.path.join(path, old_file)
            os.unlink(file_path)
    else:
        os.mkdir(path)


def test_model(model: torchvision.models, device: torch.device, data_loader: DataLoader):
    """Function that tests the feature extractor model on the test dataset.
    It runs through a forward pass to get the model output and saves the
    output images to appropriate directories through the save_test_predicts
    function.

    Args:
        model: The model to test.
        device: The device which data/model is present on.
        data_loader: The data loader contains the data to test on.
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
    save_test_predicts(predicted_labels, img_paths)


def setup_testing(experiment_folder: str):
    """Function that sets up dataloader, transforms and loads in the model
    for testing. 

    Args: 
        experiment_folder: The folder with a model.pth file
    """
    # First removing all old predictions that might be present
    # in the prediction folders
    remove_predicts()

    # Setting the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Setting test path and creating transform. The first entry is a lambda
    # dummy function, because it is cutoff in NTZFilterDataset. Might be
    # fixed in the future.
    test_path = os.path.join("data", "test")
    transform = T.Compose([
        T.Lambda(lambda x: x),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # Creating the dataset and transferring to a DataLoader
    test_data = NTZFilterDataset(test_path, transform)
    test_loader = DataLoader(test_data, batch_size = 32, collate_fn = sep_test_collate,
                             shuffle = True, num_workers = 4)

    # Loading the model form a experiment directory
    model = torch.load(os.path.join("Master-Thesis-Experiments", 
                                     experiment_folder, "model.pth"))

    # Testing the feature extractor on testing data
    test_model(model, device, test_loader)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if os.path.exists(os.path.join("Master-Thesis-Experiments", sys.argv[1])):
            print("Testing on model from experiment: " + sys.argv[1])
            setup_testing(sys.argv[1])
        else:
            print("Experiment not found, exiting ...")
    else:
        print("No folder name given, exiting...")
        exit()