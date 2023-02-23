import copy
import os
from os import listdir
import sys
import torch
import time
from torch import nn, optim
from torchmetrics import Accuracy, ConfusionMatrix
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

from NTZ_filter_dataset import NTZFilterDataset
from train_utils import sep_collate, get_transforms, convert_to_list, \
                        setup_tensorboard, setup_hyp_file, setup_hyp_dict, \
                        add_confusion_matrix

# TODO: Add precision/recall/F1 score to performance metrics
# TODO: Finish add_confusion_matrix in train_utils.py
# TODO: Debloat train_model function
# TODO: Text detection model for fourth label class?
#       -> Just add it in and see what happens?
# TODO: Create synthetic data -> for each class, move the filter across
#       the screen and the label across the filter (where applicable)
# TODO: Uncertainty prediction per image
#       -> Uncertainty per layer in the model?
# TODO: Implement classifier setup (multiple different classifiers)


def train_model(model: torchvision.models, device: torch.device,
                criterion: nn.CrossEntropyLoss, optimizer: optim.SGD, 
                acc_metric: Accuracy, conf_matrix: ConfusionMatrix, data_loaders: dict,
                tensorboard_writers: dict, epochs: int):
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
        conf_matrix: Confusion matrix between predicted and actual labels.
        data_loaders: Dictionary containing the train, validation and test
                      data loaders.
        tensorboard_writers: Dictionary containing writer elements.
        epochs: Number of epochs to train the model for.

    Returns:
        The trained model.
        The confusion matrix calculated for the last epoch.
    """
    # Setting the preliminary model to be the best model
    best_loss = 1000
    best_model = copy.deepcopy(model)
    early_stop = 0
    early_stop_limit = 25

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

            # Setting combined lists of predicted and actual labels
            combined_labels = []
            combined_labels_pred = []

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

                    # Appending prediction and actual labels to combined lists
                    combined_labels.append(labels)
                    combined_labels_pred.append(predicted_labels)
            
            # Measuring elapsed time and reporting metrics over epoch
            elapsed_time = time.time() - start_time
            mean_accuracy = acc / len(data_loaders[phase])
            print("Loss = " + str(round(loss_over_epoch, 2)))
            print("Accuracy = " + str(round(mean_accuracy, 2)))
            print("FPS = " + str(round(total_imgs / elapsed_time, 2)) + "\n")
            
            # Creating confusion matrix, only saved on last epoch
            combined_labels = convert_to_list(combined_labels)
            combined_labels_pred = convert_to_list(combined_labels_pred)
            conf_mat = conf_matrix(torch.tensor(combined_labels_pred),
                                   torch.tensor(combined_labels))
            
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
                        return model, conf_mat
            # Writing results to tensorboard
            writer = tensorboard_writers[phase]
            writer.add_scalar("Loss", loss.item(), i)
            writer.add_scalar("Accuracy", mean_accuracy, i)
    
    return model, conf_mat

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

    # Creating a dictionary for the data loaders
    data_loaders = {"train": train_loader, "val": val_loader}
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
    tensorboard_writers, experiment_path = setup_tensorboard(experiment_name)
    setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)

    # Replacing the output classification layer with a 4 class version
    model = hyp_dict["Model"]
    model.classifier[1] = nn.Linear(in_features = 1280, out_features = 4)

    # Defining Accuracy metric and transferring the model to the device
    acc_metric = Accuracy(task = "multiclass", num_classes = 4).to(device)
    conf_matrix = ConfusionMatrix(task = "multiclass", num_classes = 4)
    model.to(device)
    
    # Training the feature extractor and saving the output model
    model, conf_mat = train_model(model, device, hyp_dict["Criterion"], 
                                  hyp_dict["Optimizer"], acc_metric, conf_matrix,
                                  data_loaders, tensorboard_writers, hyp_dict["Epochs"])
    torch.save(model, os.path.join(experiment_path, "model.pth"))

    # Adding the confusion matrix of the last epoch to the tensorboard
    add_confusion_matrix(conf_mat, tensorboard_writers["hyp"])

    # Closing tensorboard writers
    for _, writer in tensorboard_writers.items():
        writer.close()


def run_all_experiments():
    """Function that runs all experiments (JSON files) in 
    the Master-Thesis-Experiments folder.
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
        print("No experiment name given, running all experiments 5 times")
        for i in range(5):            
            run_all_experiments()