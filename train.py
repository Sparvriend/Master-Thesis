import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from NTZ_filter_dataset import NTZFilterDataset
from torchmetrics import Accuracy
import copy
from tqdm import tqdm
import time
import os
from train_utils import save_test_predicts, sep_collate, sep_test_collate, get_transforms

# TODO: Use Tensorboard for implementing different experiments (Ratnajit would send tutorial)
# TODO: Use on the fly augmentation instead of fixed augmentations (per epoch) for each image (Ratnajit would send tutorial)
# TODO: Create synthetic data -> for each class, move the filter across the screen and the label across the filter (where applicable)
# TODO: Text detection model for fourth label class?
# TODO: Uncertainty prediction per image
# TODO: Adapt testing function such that it does not print a txt file but rather moves predictions to folders in a prediction folder

# File paths
TRAIN_PATH = "data/train"; VAL_PATH = "data/val"; TEST_PATH = "data/test"

# General parameters for data loaders
BATCH_SIZE = 8
EPOCHS = 2
SHUFFLE = True
NUM_WORKERS = 4

# Classes:
# 0: fail_label_crooked_print
# 1: fail_label_half_printed
# 2: fail_label_not_fully_printed
# 3: no_fail

def test_feature_extractor(model, device, data_loader):
    # Set model to evaluating
    model.eval()

    # Measuring images classified for speed measurement
    total_imgs = 0

    # Starting the validation timer
    validation_start = time.time()

    # Creating a list of paths and predicted labels
    predicted_labels = []
    img_paths = []

    with torch.no_grad():
        for inputs, paths in tqdm(data_loader):
            inputs = inputs.to(device)

            # Getting model output and adding labels/paths to lists
            model_output = model(inputs)
            predicted_labels.append(model_output.argmax(dim=1))
            img_paths.append(paths)
    
            # Counting up total amount of images a prediction was made over
            total_imgs += len(inputs)

    # Saving the test predictions
    save_test_predicts(predicted_labels, img_paths)
    
    # Getting the validation  time
    testing_time = time.time() - validation_start

    # Reporting fps metric
    print("FPS = " + str(round(total_imgs/testing_time, 2)))

def train_feature_extractor(model, device, criterion, optimizer, acc_metric, data_loaders):
    # Setting the preliminary model to be the best model
    best_loss = 1000
    best_model = copy.deepcopy(model)
    early_stop = 0; early_stop_limit = 20

    for i in range(EPOCHS):
        print("On epoch " + str(i))
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                print("Training phase")
            else:
                model.eval()
                print("Validation phase")
            
            # Set model metrics to 0
            loss_over_epoch = 0
            acc = 0
            total_imgs = 0

            # Starting model timer
            start_time = time.time()

            for inputs, labels in tqdm(data_loaders[phase]):
                with torch.set_grad_enabled(phase == "train"):
                    # Moving data to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Getting model output and labels
                    model_output = model(inputs)
                    predicted_labels = model_output.argmax(dim=1)
                    
                    # Computing the loss/accuracy
                    loss = criterion(model_output, labels)
                    acc += acc_metric(predicted_labels, labels).item()
                    
                    # Updating model weights if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Adding the loss over the epoch
                    loss_over_epoch += loss.item()

                    # Counting up total amount of images a prediction was made over
                    total_imgs += len(inputs)
            elapsed_time = time.time() - start_time

            # Reporting metrics over epoch
            mean_accuracy = acc / len(data_loaders[phase])
            print("Loss = " + str(loss_over_epoch))
            print("Accuracy = " + str(mean_accuracy))

            # Reporting fps metric
            print("FPS = " + str(round(total_imgs/elapsed_time, 2)))

            if phase == "val":
                # Change best model to new model if validation loss is better
                if best_loss > loss_over_epoch:
                    best_model = copy.deepcopy(model)
                    best_loss = loss_over_epoch
                    early_stop = 0

                # Change model back to old model if validation loss is worse
                else:
                    model = copy.deepcopy(best_model)
                    early_stop += 1
                    if early_stop > early_stop_limit:
                        print("Early stopping ")
                        return

# Function that defines data loaders based on NTZ_filter_datasets
def setup_data_loaders():
    # Defining the list of transforms
    transform = get_transforms()

    # Creating datasets for validation and training data, based on NTZFilterDataset class
    train_data = NTZFilterDataset(TRAIN_PATH, transform)
    val_data = NTZFilterDataset(VAL_PATH, transform)

    # Dataset for testing class, with labels
    test_data_label = NTZFilterDataset(TEST_PATH, transform)

    # Creating data loaders for validation and training data
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, collate_fn = sep_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, collate_fn = sep_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)

    # Creating data loader for testing data
    test_loader = DataLoader(test_data_label, batch_size = BATCH_SIZE, collate_fn = sep_test_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)

    # Creating a dictionary for the data loaders
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    return data_loaders

# Function that does a setup of all datasets/dataloaders and proceeds to training/validating of the feature extractor
def setup_feature_extractor():
    # First setting the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))

    # Retrieving data loaders
    data_loaders = setup_data_loaders()

    # First using the ready made model from Pytorch
    model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
    # Replacing the output classification layer with a 4 class version
    model.classifier[1] = nn.Linear(in_features=1280, out_features=4)

    # Defining model criterion as well as weight updater (SGD) and transferring the model to the device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    acc_metric = Accuracy(task="multiclass", num_classes=4).to(device)
    model.to(device)

    # Training the feature extractor
    train_feature_extractor(model, device, criterion, optimizer, acc_metric, data_loaders)

    # Testing the feature extractor on testing data
    print("Testing phase")
    test_feature_extractor(model, device, data_loaders["test"])

if __name__ == '__main__':
    setup_feature_extractor()