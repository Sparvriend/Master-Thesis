import copy
import os
from os import listdir
import sys
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy, F1Score
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

from NTZ_filter_dataset import NTZFilterDataset
from train_utils import sep_collate, get_transforms, setup_tensorboard, \
                        setup_hyp_file, setup_hyp_dict, add_confusion_matrix, \
                        report_metrics

# To open tensorboard in browser:
# Run the following command in a new terminal:
# tensorboard --logdir=Master-Thesis-Experiments
# TODO: Start thinking about reporting results 
#       -> excel sheet reporting (download data from tensorboard and report final accuracy)
#       -> Report results per experiment, explaining conclusion from each experiment.
#       -> Current exp:
#                       1. Baseline (No augmentation/no pretraining)
#                       2. Pretrained + no augmentation
#                       3. Pretrained + augmentation
#                       4. Pretrained + augmentation without lambda functions
#                       5. Pretrained + different iterations of model replacement (1-5)
#                          + augmentation types
#                       6. Not pretrained + augmentation
#                       7. Pretrained + augmentations seperately (test_augmentations.py)
#                       8. Pretrained + all augmentations on top of each other (PRIORITY)
#                       9. Case Study: Applying setup to tinyImageNet classification.
# TODO: Report GPU memory/Energy
# TODO: In test_augmentations.py, add all results to the same .txt file instead of seperate files
#       -> Implemented, check if it works fine by running test_augmentations.py
# TODO: Implement classifier setup (multiple different classifiers)
#       -> Setup model loading from JSON file in such a way that it is
#       compatible with different final classification layer names/in features.
# TODO: Create synthetic data -> for each class, move the filter across
#       the screen and the label across the filter (where applicable)
# TODO: Uncertainty prediction per image
#       -> Uncertainty per layer in the model?
# TODO: Text detection model for fourth label class?
#       -> Need more information from client about when text is wrong.
#       -> Just add it in and see what happens?    


def train_model(model: torchvision.models, device: torch.device,
                criterion: nn.CrossEntropyLoss, optimizer: optim.SGD,
                scheduler: lr_scheduler.MultiStepLR, data_loaders: dict,
                tensorboard_writers: dict, epochs: int, pfm_flag: bool,
                early_stop_limit: int, model_replacement_limit: int):
    """Function that improves the model through training and validation.
    Includes early stopping, iteration model saving only on improvement,
    performance metrics saving and timing.

    Args:
        model: Pretrained image classification model.
        device: The device which data/model is present on.
        criterion: Cross Entropy Loss function
        optimizer: Stochastic Gradient Descent optimizer (descents in
                   opposite direction of steepest gradient).
        scheduler: Scheduler that decreases the learning rate.
        data_loaders: Dictionary containing the train, validation and test
                      data loaders.
        tensorboard_writers: Dictionary containing writer elements.
        epochs: Number of epochs to train the model for.
        pfm_flag: Boolean deciding on whether to print performance
                  metrics to terminal.
        early_stop_limit: Number of epochs to wait before early stopping.
        model_replacement_limit: Number of epochs to wait before
                                 replacing model.
    Returns:
        The trained model.
        The combined actual and predicted labels per epoch.
    """
    # Setting the preliminary model to be the best model
    best_loss = 1000
    best_model = copy.deepcopy(model)
    early_stop = 0
    model_replacement = 0

    # Setting up performance metrics
    acc_metric = Accuracy(task = "multiclass", num_classes = 4).to(device)
    f1_metric = F1Score(task = "multiclass", num_classes = 4).to(device)

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
            f1_score = 0
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
                    
                    # Computing performance metrics
                    loss = criterion(model_output, labels)
                    acc += acc_metric(predicted_labels, labels).item()
                    f1_score += f1_metric(predicted_labels, labels).item()
                    
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
            
            writer = tensorboard_writers[phase]
            report_metrics(pfm_flag, start_time, len(data_loaders[phase]), acc,
                           f1_score, loss_over_epoch, total_imgs, writer, i)
            
            if phase == "val":
                # Change best model to new model if validation loss is better
                if best_loss > loss_over_epoch:
                    best_model = copy.deepcopy(model)
                    best_loss = loss_over_epoch
                    early_stop = 0
                    model_replacement = 0

                # Change model back to old model if validation loss is worse
                # over a a number of epochs.
                # The early stop limit/model replacement can be set to 0
                # in the JSON file to disable the feature.
                else:
                    model_replacement += 1
                    early_stop += 1
                    if model_replacement >= model_replacement_limit and \
                        model_replacement_limit != 0:
                        print("Replacing model")
                        model.load_state_dict(best_model.state_dict())
                        model_replacement = 0
                    if early_stop > early_stop_limit and early_stop_limit != 0:
                        print("Early stopping ")
                        return model, combined_labels, combined_labels_pred
                # Updating the learning rate if updating scheduler is used
                scheduler.step()

    return model, combined_labels, combined_labels_pred

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
    train_path = os.path.join("data", "train")
    val_path = os.path.join("data", "val")

    # Creating datasets for training and validation
    # based on NTZFilterDataset class.
    train_data = NTZFilterDataset(train_path, transform)
    val_data = NTZFilterDataset(val_path, transform)

    # Creating data loaders for training and validation
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
    # And transferring model to device
    model = hyp_dict["Model"]
    model.classifier[1] = nn.Linear(in_features = 1280, out_features = 4)
    model.to(device)
    
    # Training and saving model
    model, c_labels, c_labels_pred = train_model(model, device, 
                                                 hyp_dict["Criterion"],
                                                 hyp_dict["Optimizer"],
                                                 hyp_dict["Scheduler"],
                                                 data_loaders, 
                                                 tensorboard_writers,
                                                 hyp_dict["Epochs"], 
                                                 hyp_dict["PFM Flag"],
                                                 hyp_dict["Early Limit"],
                                                 hyp_dict["Replacement Limit"])
    torch.save(model, os.path.join(experiment_path, "model.pth"))

    # Adding the confusion matrix of the last epoch to the tensorboard
    add_confusion_matrix(c_labels, c_labels_pred, tensorboard_writers["hyp"])

    # Closing tensorboard writers
    for _, writer in tensorboard_writers.items():
        writer.close()


def run_all_experiments():
    """Function that runs all experiments (JSON files) in 
    the Master-Thesis-Experiments folder.
    """
    experiment_list = ["no_augment", "rand_augment", "categorical",
                       "auto_augment", "not_pretrained", "scheduler",
                       "3_model_replacement", "no_model_replacement"]
    for file in experiment_list:
        experiment_name = "MobileNetV2-" + file
        print("Running experiment: " + experiment_name)
        run_experiment(experiment_name)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if os.path.exists(os.path.join("Master-Thesis-Experiments", sys.argv[1] + ".json")):
            if len(sys.argv) == 2:
                print("Running experiment: " + sys.argv[1])
                run_experiment(sys.argv[1])
            else:
                for i in range(int(sys.argv[2])):
                    print("Running experiment: " + sys.argv[1])
                    run_experiment(sys.argv[1])
        else:
            print("Experiment not found, exiting ...")
    else:
        print("No experiment name given, running all experiments 3 times")
        for i in range(3):            
            run_all_experiments()