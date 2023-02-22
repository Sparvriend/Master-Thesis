import copy
import os
from os import listdir
import sys
import torch
import time
from torch import nn, optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from NTZ_filter_dataset import NTZFilterDataset
from train_utils import save_test_predicts, sep_collate, sep_test_collate, \
                        get_transforms, setup_tensorboard, setup_hyp_file, \
                        setup_hyp_dict 

# TODO: Save experiment setups to a file type which can be loaded in for
#       experiments and saving of experiments with tensorboard.
#       -> Allow for giving the experiment name as an input to train.py
# TODO: Create synthetic data -> for each class, move the filter across
#       the screen and the label across the filter (where applicable)
# TODO: Uncertainty prediction per image
#       -> Uncertainty per layer in the model?
# TODO: Text detection model for fourth label class?
# TODO: Implement classifier setup (multiple different classifiers)

# Classes:
# 0: fail_label_crooked_print
# 1: fail_label_half_printed
# 2: fail_label_not_fully_printed
# 3: no_fail


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


def train_model(model: torchvision.models, device: torch.device,
                criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, 
                acc_metric: Accuracy, data_loaders: dict, tensorboard_writers: dict,
                epochs: int):
    """Function that improves the model through training and validation.
    Includes early stopping, iteration model saving only on improvement,
    performance metrics saving and timing.

    Args:
        model: Pretrained image classification model.
        device: The device which data/model is present on.
        criterion: Cross Entropy Loss function
        optimizer: Stochastic Gradient Descent optimizer (descents in
                   opposite direction of steepest gradient).
        acc_metric: Accuracy measurement between predicted and actual labels.
        data_loaders: Dictionary containing the train, validation and test
                      data loaders.
        tensorboard_writers: Dictionary containing writer elements.
        epochs: Number of epochs to train the model for.
    """
    # Setting the preliminary model to be the best model
    best_loss = 1000
    best_model = copy.deepcopy(model)
    early_stop = 0
    early_stop_limit = 100

    for i in range(epochs):
        print("On epoch " + str(i))
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                print("Training phase")
            else:
                model.eval()
                print("Validation phase")
            
            # Set model metrics to 0 and starting model timer
            loss_over_epoch = 0
            acc = 0
            total_imgs = 0
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

                    # Adding the loss over the epoch and counting
                    # total images a prediction was made over
                    loss_over_epoch += loss.item()
                    total_imgs += len(inputs)
            
            # Measuring elapsed time and reporting metrics over epoch
            elapsed_time = time.time() - start_time
            mean_accuracy = acc / len(data_loaders[phase])
            print("Loss = " + str(round(loss_over_epoch, 2)))
            print("Accuracy = " + str(round(mean_accuracy, 2)))
            print("FPS = " + str(round(total_imgs / elapsed_time, 2)) + "\n")

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
            # Writing results to tensorboard
            writer = tensorboard_writers[phase]
            writer.add_scalar("Loss", loss.item(), i)
            writer.add_scalar("Accuracy", mean_accuracy, i)
    
    # Closing tensorboard writers
    for writer in tensorboard_writers:
        writer.close()


def setup_data_loaders(augmentation_type: str, batch_size: int,
                       shuffle: bool, num_workers: int) -> dict:
    """Function that defines data loaders based on NTZFilterDataset class. It
    combines the data loaders in a dictionary.

    Returns:
        Dictionary of the training, validation and testing data loaders.
    """
    # Defining the list of transforms
    transform = get_transforms(augmentation_type)

    # File paths
    train_path = "data/train"
    val_path = "data/val"
    test_path = "data/test"

    # Creating datasets for training, validation and testing data,
    # based on NTZFilterDataset class.
    train_data = NTZFilterDataset(train_path, transform)
    val_data = NTZFilterDataset(val_path, transform)
    test_data = NTZFilterDataset(test_path, transform)

    # Creating data loaders for training, validation and testing data
    train_loader = DataLoader(train_data, batch_size = batch_size,
                              collate_fn = sep_collate, shuffle = shuffle,
                              num_workers = num_workers)
    val_loader = DataLoader(val_data, batch_size = batch_size,
                            collate_fn = sep_collate, shuffle = shuffle,
                            num_workers = num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size,
                             collate_fn = sep_test_collate, shuffle = shuffle,
                            num_workers = num_workers)

    # Creating a dictionary for the data loaders
    data_loaders = {"train": train_loader, "val": val_loader,
                    "test": test_loader}
    return data_loaders


def run_experiment(experiment_name: str):
    """Function that does a setup of all datasets/dataloaders and proceeds to
    training/validating of the image classification model.

    Args: 
        experiment_name: Name of the experiment to run.
    """
    # Setting the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Retrieving hyperparameter dictionary
    hyp_dict = setup_hyp_dict(experiment_name)

    # Retrieving data loaders
    data_loaders = setup_data_loaders(hyp_dict["Augmentation"], 
                                      hyp_dict["Batch Size"], 
                                      hyp_dict["Shuffle"],
                                      hyp_dict["Num Workers"])

    # Setting up tensorboard writers and writing hyperparameters
    tensorboard_writers = setup_tensorboard(experiment_name)
    setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)

    # Replacing the output classification layer with a 4 class version
    model = hyp_dict["Model"]
    model.classifier[1] = nn.Linear(in_features = 1280, out_features = 4)

    # Defining Accuracy metric and transferring the model to the device
    acc_metric = Accuracy(task="multiclass", num_classes = 4).to(device)
    model.to(device)
    
    # Training the feature extractor
    train_model(model, device, hyp_dict["Criterion"], hyp_dict["Optimizer"],
                acc_metric, data_loaders, tensorboard_writers, hyp_dict["Epochs"])

    # Testing the feature extractor on testing data
    test_model(model, device, data_loaders["test"])


def run_all_experiments():
    """Function that runs all experiments (json files) in the Master-Thesis-Experiments folder.
    """
    path = "Master-Thesis-Experiments"
    files = [f for f in listdir(path) 
             if os.path.isfile(os.path.join(path, f))]
    for file in files:
        if file.endswith(".json"):
            experiment_name = os.path.splitext(file)[0]
            print("Running experiment: " + experiment_name)
            run_experiment(experiment_name)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("Running experiment: " + sys.argv[1])
        run_experiment(sys.argv[1])
    else:
        print("No experiment name given, running all experiments")
        run_all_experiments()
    