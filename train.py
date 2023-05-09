import copy
import os
import sys
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy, F1Score
import torchvision
from tqdm import tqdm
from types import SimpleNamespace

from explainability import get_grad_pen
from utils import get_transforms, setup_tensorboard, setup_hyp_file, \
                  setup_hyp_dict, add_confusion_matrix, get_data_loaders, \
                  report_metrics, set_classification_layer, merge_experiments, \
                  calculate_acc_std


def train_model(model: torchvision.models, device: torch.device,
                criterion: nn.CrossEntropyLoss, optimizer: optim.SGD,
                scheduler: lr_scheduler.MultiStepLR, data_loaders: dict,
                tensorboard_writers: dict, epochs: int, pfm_flag: bool,
                rbf_flag: bool, early_stop_limit: int,
                model_replacement_limit: int, experiment_path: str,
                classes: int) -> tuple[nn.Module, list, list]:
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
        rbf_flag: Boolean indicating DUQ usage.
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
                    inputs.requires_grad_(True) if rbf_flag else None

                    # Getting model output and labels
                    model_output = model(inputs)
                    predicted_labels = model_output.argmax(dim=1)
                    
                    # Computing performance metrics
                    loss = criterion(model_output, labels)
                    acc += acc_metric(predicted_labels, labels).item()
                    f1_score += f1_metric(predicted_labels, labels).item()
                    
                    # Updating model weights if in training phase
                    if phase == "train":
                        # Optionally, add L2 gradient penalty to RBF loss
                        if rbf_flag == True:
                            grad_pen = get_grad_pen(inputs, model_output)
                            loss += grad_pen

                        loss.backward()
                        optimizer.step()

                        inputs.requires_grad_(False) if rbf_flag else None
                        # Optionally, update RBF centroids
                        if rbf_flag == True:
                            with torch.no_grad():
                                model.eval()
                                model.update_centres(inputs, labels)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Retrieving hyperparameter dictionary
    hyp_dict = setup_hyp_dict(experiment_name)
    args = SimpleNamespace(**hyp_dict)

    # Defining the train transforms
    transform = get_transforms(args.augmentation)
    # Retrieving data loaders
    data_loaders = get_data_loaders(args.batch_size, args.shuffle, args.num_workers,
                                    transform, args.dataset)

    # Setting up tensorboard writers and writing hyperparameters
    tensorboard_writers, experiment_path = setup_tensorboard(experiment_name, "Experiment-Results")
    setup_hyp_file(tensorboard_writers["hyp"], hyp_dict)

    # Replacing the output classification layer with a N class version
    # And transferring model to device
    model = args.model
    classes = data_loaders["train"].dataset.n_classes
    model = set_classification_layer(model, classes, args.RBF_flag, device)
    model.to(device)
    
    # Recording total training time
    training_start = time.time()

    # Training and saving model
    model, c_labels, c_labels_pred = train_model(model, device, args.criterion,
                                                 args.optimizer, args.scheduler,
                                                 data_loaders, tensorboard_writers,
                                                 args.epochs, args.PFM_flag,
                                                 args.RBF_flag, args.early_limit,
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


def setup():
    """This function sets up experimentation. There are three methods of
    running the experiment, this function takes care of running with
    whatever method is specified by the input.
    """
    experiment_path = "Experiments"
    results_path = os.path.join("Results", "Experiment-Results")
    if len(sys.argv) > 1:
        if os.path.exists(os.path.join(experiment_path, sys.argv[1] + ".json")):
            if len(sys.argv) == 2:
                print("Running experiment: " + sys.argv[1])
                run_experiment(sys.argv[1])
            else:
                for _ in range(int(sys.argv[2])):
                    print("Running experiment: " + sys.argv[1])
                    run_experiment(sys.argv[1])
                merge_experiments([sys.argv[1]], results_path)
                calculate_acc_std([sys.argv[1]], results_path)
        else:
            print("Experiment not found, exiting ...")
    else:
        # Listing the experiments, which are the JSON files
        # Without the testaugments file, since that is only meant
        # to be ran by test_augmentations.py
        experiment_list = []
        files = os.listdir(experiment_path)
        for file in files:
            if file.endswith(".json"):
                experiment_list.append(file[:-5])
        experiment_list.remove("TestAugments")
        experiment_list.remove("DEFAULT")

        print("No experiment name given, running all experiments 5 times")
        for _ in range(5):
            for file in experiment_list:
                print("Running experiment: " + file)
                run_experiment(file)
        merge_experiments(experiment_list, results_path)
        calculate_acc_std(experiment_list, results_path)            


if __name__ == '__main__':
    setup()