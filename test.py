import os
from os import listdir
import sys
import time
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from NTZ_filter_dataset import NTZFilterDataset
from train_utils import save_test_predicts, sep_test_collate


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
    # Set model to evaluatingm, set speed measurement variable
    # and starting the timer
    model.eval()
    total_imgs = 0
    validation_start = time.time()

    # Creating a list of paths and predicted labels
    predicted_labels = []
    img_paths = []

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
    save_test_predicts(predicted_labels, img_paths)
    testing_time = time.time() - validation_start
    print("FPS = " + str(round(total_imgs / testing_time, 2)))


def remove_predicts():
    """Function that removes all old predictions from
    the test_predictions folder.
    """
    # Path information
    test_prediction_destination = "data/test_predictions"
    data_classes = ["fail_label_crooked_print", "fail_label_half_printed",
                    "fail_label_not_fully_printed", "no_fail"]

    # Going by each data class and removing old images present in the folders
    for data_class in data_classes:
        path = os.path.join(test_prediction_destination, data_class)
        files = [f for f in listdir(path) 
                 if os.path.isfile(os.path.join(path, f))]
        for old_file in files:
            file_path = os.path.join(path, old_file)
            os.unlink(file_path)


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
    test_path = "data/test"
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
        print("Testing on model from experiment: " + sys.argv[1])
        setup_testing(sys.argv[1])
    else:
        print("No folder name given, exiting...")
        exit()