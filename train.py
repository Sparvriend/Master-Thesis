import argparse
import copy
import os
import time
import torch
import torchvision

from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm
from types import SimpleNamespace

from utils import get_transforms, setup_tensorboard, setup_hyp_file, \
                  setup_hyp_dict, add_confusion_matrix, get_data_loaders, \
                  report_metrics, set_classification_layer, merge_experiments, \
                  calculate_acc_std, get_device


def train_model(model: torchvision.models, device: torch.device,
                criterion: nn.CrossEntropyLoss, optimizer: optim.SGD,
                scheduler: lr_scheduler.MultiStepLR, data_loaders: dict,
                tensorboard_writers: dict, epochs: int, pfm_flag: bool,
                early_stop_limit: int, model_replacement_limit: int,
                experiment_path: str, classes: int) -> tuple[nn.Module, list, list]:
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
        experiment_path: Path to the experiment folder.
        classes: Number of classes in the dataset.
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
    acc_metric = Accuracy(task = "multiclass", num_classes = classes).to(device)
    f1_metric = F1Score(task = "multiclass", num_classes = classes).to(device)

    for i in range(epochs):
        print("Epoch " + str(i))
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

                    # Removing gradients from previous batch
                    optimizer.zero_grad()

                    # Getting model output and labels
                    model_output = model(inputs)
                    predicted_labels = model_output.argmax(dim = 1)

                    # Computing performance metrics
                    loss = criterion(model_output, labels)
                    acc += acc_metric(predicted_labels, labels).item()
                    f1_score += f1_metric(predicted_labels, labels).item()
                    
                    # Updating model weights if in training phase
                    if phase == "train":
                        # Backwards pass and updating optimizer
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
                           f1_score, loss_over_epoch, total_imgs, writer, i,
                           experiment_path)
            
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


def run_experiment(experiment_name: str):
    """Function that does a setup of all datasets/dataloaders and proceeds to
    training/validating of the image classification model.

    Args: 
        experiment_name: Name of the experiment to run.
    """
    # Setting the device to use
    device = get_device()

    # Retrieving hyperparameter dictionary
    hyp_dict = setup_hyp_dict(experiment_name)
    args = SimpleNamespace(**hyp_dict)

    # Check if RBF, if so, then this setup will not work
    if args.RBF_flag == True:
        print("RBF experiment, exiting ...")
        return

    # Defining the train transforms
    transform = get_transforms(args.dataset, args.augmentation)
    # Retrieving data loaders
    data_loaders = get_data_loaders(args.batch_size, transform, args.dataset)

    # Setting up tensorboard writers and writing hyperparameters
    tensorboard_writers, experiment_path = setup_tensorboard(experiment_name,
                                                             "Experiment-Results")
    setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)

    # Replacing the output classification layer with a N class version
    # And transferring model to device
    model = args.model
    classes = data_loaders["train"].dataset.n_classes
    model = set_classification_layer(model, classes)
    model.to(device)
    
    # Recording total training time
    training_start = time.time()

    # Training and saving model
    model, c_labels, c_labels_pred = train_model(model, device, args.criterion,
                                                 args.optimizer, args.scheduler,
                                                 data_loaders, tensorboard_writers,
                                                 args.epochs, args.PFM_flag,
                                                 args.early_limit,
                                                 args.replacement_limit,
                                                 experiment_path,
                                                 classes)
    
    elapsed_time = time.time() - training_start
    print("Total training time (H/M/S) = ", 
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    torch.save(model, os.path.join(experiment_path, "model.pth"))

    # Adding the confusion matrix of the last epoch to the tensorboard
    add_confusion_matrix(c_labels, c_labels_pred, tensorboard_writers["hyp"],
                         data_loaders["train"].dataset.label_map)

    # Closing tensorboard writers
    for _, writer in tensorboard_writers.items():
        writer.close()          


if __name__ == '__main__':
    # Forming argparser with optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type = str)
    parser.add_argument("--n_runs", type = int, default = 1)
    args = parser.parse_args()
    experiment_path = "Experiments"
    results_path = os.path.join("Results", "Experiment-Results")
    experiment_name = args.experiment_name
    
    if experiment_name != None:
        # An experiment was given, check if it exists
        if os.path.exists(os.path.join(experiment_path, experiment_name + ".json")):
            # Then run the experiment n_runs times
            print("Running experiment: " + experiment_name)
            for _ in range(args.n_runs):
                run_experiment(experiment_name)
            if args.n_runs != 1:
                merge_experiments([experiment_name], results_path)
                calculate_acc_std([experiment_name], results_path) 
        else:
            print("Experiment not found, exiting ...")
