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
from train_utils import save_test_predicts, sep_collate, sep_test_collate

# TODO:
# - Use Tensorboard for implementing different experiments (Ratnajit would send tutorial)
# - Use on the fly augmentation instead of fixed augmentations (per epoch) for each image (Ratnajit would send tutorial)
# - Create synthetic data -> for each class, move the filter across the screen and the label across the filter (where applicable)
# - Text detection model for fourth label class?
# - Uncertainty prediction per image
# - Combine validation and training functions? -> Move usage of device from start of setup_feature_extractor() to only where it is needed

# File paths
TRAIN_PATH = "data/train"; VAL_PATH = "data/val"; TEST_PATH_LABEL = "data/test"; TEST_PATH_NO_LABEL = "data/test_no_label"

# General parameters for data loaders
BATCH_SIZE = 8
EPOCHS = 10
SHUFFLE = True
NUM_WORKERS = 4

# Classes:
# 0: fail_label_crooked_print
# 1: fail_label_half_printed
# 2: fail_label_not_fully_printed
# 3: no_fail

# Validation function for the feature extractor
def validate_fe(model, device, criterion, acc_metric, data_loader):
    # Set model to evaluating
    model.eval()

    # Set model metrics to 0
    loss_over_epoch = 0
    acc = 0
    total_imgs = 0

    # Starting the validation timer
    validation_start = time.time()

    # Unpacking all inputs and labels to run on the model in batches
    # The model weights should not be updated, hence running it with no_grad()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Getting model output and computing the loss/accuracy
            model_output = model(inputs)
            loss = criterion(model_output, labels)
            predicted_labels = model_output.argmax(dim=1)
            acc += acc_metric(predicted_labels, labels).item()

            # Adding the loss over the epoch
            loss_over_epoch += loss.item()
            
            # Counting up total amount of images a prediction was made over
            total_imgs += len(labels)
    
    # Getting the validation  time
    validation_time = time.time() - validation_start

    # Reporting metrics over epoch
    mean_accuracy = acc / len(data_loader)
    print("Loss = " + str(loss_over_epoch))
    print("Accuracy = " + str(mean_accuracy))

    # Reporting fps metric
    print("FPS = " + str(round(total_imgs/validation_time, 2)))

    return loss_over_epoch

def test_feature_extractor(model, device, data_loader):
    # Set model to evaluating
    model.eval()

    # Measuring images classified for speed measurement
    total_imgs = 0

    # Starting the validation timer
    validation_start = time.time()

    # Clearing file contents of test_predicts.txt if it exists
    open(os.path.join(TEST_PATH_NO_LABEL, "test_predicts.txt"), "w") 

    with torch.no_grad():
        for inputs, paths in tqdm(data_loader):
            inputs = inputs.to(device)

            # Getting model output and labels
            model_output = model(inputs)
            predicted_labels = model_output.argmax(dim=1)
    
            # Counting up total amount of images a prediction was made over
            total_imgs += len(inputs)

            # Saving predictions to a txt file
            save_test_predicts(predicted_labels, paths)

    # Getting the validation  time
    testing_time = time.time() - validation_start

    # Reporting fps metric
    print("FPS = " + str(round(total_imgs/testing_time, 2)))

# Training function for the feature extractor for one epoch
def train_fe_one_epoch(model, device, criterion, optimizer, acc_metric, data_loader):
    # Set model to training phase
    model.train()

    # Set model metrics to 0
    loss_over_epoch = 0
    acc = 0
    
    # Unpacking all inputs and labels to run on the model in batches
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Resetting model weights and getting the model output
        optimizer.zero_grad()
        model_output = model(inputs)

        # Computing the loss/accuracy and updating the model with a backwards pass
        loss = criterion(model_output, labels)
        acc += acc_metric(model_output.argmax(dim=1), labels).item()
        loss.backward()
        optimizer.step()

        # Adding the loss over the epoch
        loss_over_epoch += loss.item()

    # Reporting the metrics over one epoch
    mean_accuracy = acc / len(data_loader)
    print("Training loss = " + str(loss_over_epoch))
    print("Training accuracy = " + str(mean_accuracy))

# Function that defines training of feature extractor over epochs
def train_feature_extractor(model, device, criterion, optimizer, acc_metric, train_loader, val_loader):
    # Setting the preliminary model to be the best model
    best_model = copy.deepcopy(model)
    best_loss = 1000

    # Main epoch loop
    for i in range(EPOCHS):
        print("On epoch " + str(i))
        print("Training phase")
        train_fe_one_epoch(model, device, criterion, optimizer, acc_metric, train_loader)

        print("Validation phase")
        loss = validate_fe(model, device, criterion, acc_metric, val_loader)

        # Change best model to new model if validation loss is better
        if best_loss > loss:
            best_model = copy.deepcopy(model)
            best_loss = loss
        # Change model back to old model if validation loss is worse
        else:
            model = copy.deepcopy(best_model)

# Function that defines data loaders based on NTZ_filter_datasets
def setup_data_loaders():
    # Defining transforms for training data based on information from https://pytorch.org/hub pytorch_vision_mobilenet_v2/
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # Creating datasets for validation and training data, based on NTZFilterDataset class
    train_data = NTZFilterDataset(TRAIN_PATH, transform)
    val_data = NTZFilterDataset(VAL_PATH, transform)

    # Dataset for testing class, with labels
    # test_data_label = NTZFilterDataset(TEST_PATH_LABEL, transform)
    test_data_no_labels = NTZFilterDataset(TEST_PATH_NO_LABEL, transform)

    # Creating data loaders for validation and training data
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, collate_fn = sep_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, collate_fn = sep_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)

    # Creating data loader for testing data
    test_loader = DataLoader(test_data_no_labels, batch_size = BATCH_SIZE, collate_fn = sep_test_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)

    return train_loader, val_loader, test_loader

# Function that does a setup of all datasets/dataloaders and proceeds to training/validating of the feature extractor
def setup_feature_extractor():
    # First setting the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))

    # Retrieving data loaders
    train_loader, val_loader, test_loader = setup_data_loaders()

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
    train_feature_extractor(model, device, criterion, optimizer, acc_metric, train_loader, val_loader)

    # Testing the feature extractor on testing data
    print("Testing phase")
    test_feature_extractor(model, device, test_loader)

if __name__ == '__main__':
    setup_feature_extractor()