import copy
import numpy as np
import os
import torch
from torch import nn, optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torchvision.transforms as T
from train import sep_collate, train_model
from train_utils import get_categorical_transforms, setup_hyp_dict, \
                        setup_tensorboard, setup_hyp_file

from NTZ_filter_dataset import NTZFilterDataset


def get_augment_loaders(augment: T.Compose, batch_size: int,
                        shuffle: bool, num_workers: int) -> dict:
    """Function that creates dataloaders for training and validation,
    based on a specific augmentation.

    Args:
        augment: The augmentation to combine for the transform.
        batch_size: How many samples per batch.
        shuffle: Whether to shuffle the data.
        num_workers: number of workers for dataloaders

    Returns:
        Dictionary with train and validation loaders
    """
    transform = T.Compose([
        augment,
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # File paths
    train_path = "data/train"
    val_path = "data/val"

    # Creating datasets for training, validation and testing data,
    # based on NTZFilterDataset class.
    train_data = NTZFilterDataset(train_path, transform)
    val_data = NTZFilterDataset(val_path, transform)

    # Creating data loaders for training, validation and testing data
    train_loader = DataLoader(train_data, batch_size = batch_size,
                              collate_fn = sep_collate, shuffle = shuffle,
                              num_workers = num_workers)
    val_loader = DataLoader(val_data, batch_size = batch_size,
                            collate_fn = sep_collate, shuffle = shuffle,
                            num_workers = num_workers)
    
    data_loaders = {"train": train_loader, "val": val_loader}
    return data_loaders


def setup_augmentation_testing():
    """This function computes the average accuracy over a number of runs
    for one of the augmentation types defined in train_utils' function
    get_categorical_transforms(). It also saves the seperate run results
    to tensorboard.
    """
    # Setting the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Defining experiment name and retrieving hyperparameter dictionary
    experiment_name = "MobileNetV2-test_augment"
    hyp_dict = setup_hyp_dict(experiment_name)

    # Replacing the output classification layer with a 4 class version.
    # Transferring model to device and making a baseline copy.
    model = hyp_dict["Model"]
    model.classifier[1] = nn.Linear(in_features = 1280, out_features = 4)
    model.to(device)
    def_model = copy.deepcopy(model)

    # Getting all augmentations and defining a number of runs to average over
    augmentation_types, _ = get_categorical_transforms()
    num_runs = 5

    # Defining accuracy metric for multi classification
    acc_metric = Accuracy(task = "multiclass", num_classes = 4).to(device)

    for augment in augmentation_types:
        if isinstance(augment, T.Lambda):
           continue
        elif str(augment).startswith("<train_utils.CustomCorruption"):
            augment_type = augment.corruption_name
        else:
            # Defining experiment folder name and augment type
            augment_type = str(augment).split("(")[0]
        experiment_folder_name = "MobileNetV2-test_augment-" + augment_type

        # Retrieving data loaders for the current augmentation
        data_loaders = get_augment_loaders(augment, 
                                           hyp_dict["Batch Size"],
                                           hyp_dict["Shuffle"],
                                           hyp_dict["Num Workers"])
        
        # Resetting avg accuracy matrix
        acc_list = []

        for i in range(num_runs):
            # Resetting the model on each run
            model.load_state_dict(def_model.state_dict())

            # Setting up tensorboard writers
            tensorboard_writers, _ = setup_tensorboard(os.path.join(experiment_folder_name,
                                                                    ("Run" + str(i) + "-")))
            setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)
            
            # Training model with the current augmentation
            model, c_labels, c_labels_pred = train_model(model, device, 
                                                         hyp_dict["Criterion"], 
                                                         hyp_dict["Optimizer"], 
                                                         data_loaders,
                                                         tensorboard_writers,
                                                         hyp_dict["Epochs"],
                                                         hyp_dict["PFM Flag"],
                                                         hyp_dict["Early Limit"],
                                                         hyp_dict["Replacement Limit"])
            acc_list.append(acc_metric(c_labels_pred[0], c_labels[0]).item())

            # Closing writers for the iteration
            for _, writer in tensorboard_writers.items():
                writer.close()
        
        # Opening txt file and writing average accuracy result to it
        f = open(os.path.join("Master-Thesis-Experiments",
                              experiment_folder_name, "result.txt"), "a")
        f.write("Mean accuracy of last epoch over " + str(num_runs) + " runs for "
                + str(augment_type) + ": " + str(round(np.mean(acc_list), 2)))
        f.write("\nStandard deviation of last epoch over " + str(num_runs) +
                " runs for " + str(augment_type) + ": " +
                str(round(np.std(acc_list), 2)))
        f.close()


if __name__ == '__main__':
    setup_augmentation_testing()