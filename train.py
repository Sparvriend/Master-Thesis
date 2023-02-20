import copy
import torch
import time
from torch import nn, optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

from NTZ_filter_dataset import NTZFilterDataset
from train_utils import save_test_predicts, sep_collate, sep_test_collate, get_transforms

# TODO: Use Tensorboard for implementing different experiments 
#       -> top1/top5 acc percentage and loss 
#       -> also save model per 2 epochs
# TODO: Create synthetic data -> for each class, move the filter across
#       the screen and the label across the filter (where applicable)
# TODO: Uncertainty prediction per image
# TODO: Text detection model for fourth label class?
# TODO: Implement classifier setup (multiple different classifiers)

# Classes:
# 0: fail_label_crooked_print
# 1: fail_label_half_printed
# 2: fail_label_not_fully_printed
# 3: no_fail

# Setting the amount of training epochs
EPOCHS = 100

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
                acc_metric: Accuracy, data_loaders: dict):
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
    """
    # Setting the preliminary model to be the best model
    best_loss = 1000
    best_model = copy.deepcopy(model)
    early_stop = 0
    early_stop_limit = 20

    for i in range(EPOCHS):
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


def setup_data_loaders() -> dict:
    """Function that defines data loaders based on NTZFilterDataset class. It
    combines the data loaders in a dictionary.

    Returns:
        Dictionary of the training, validation and testing data loaders.
    """
    # Defining the list of transforms
    transform = get_transforms()

    # File paths
    train_path = "data/train"
    val_path = "data/val"
    test_path = "data/test"

    # General parameters for data_loaders
    batch_size = 32
    shuffle = True
    num_workers = 4

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


def setup_model():
    """Function that does a setup of all datasets/dataloaders and proceeds to
    training/validating of the image classification model.
    """

    # Setting the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Retrieving data loaders
    data_loaders = setup_data_loaders()

    # Using MobileNetV2 pretrained model from PyTorch
    model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
    # Replacing the output classification layer with a 4 class version
    model.classifier[1] = nn.Linear(in_features = 1280, out_features = 4)

    # Defining model criterion as well as weight updater (SGD)
    # and transferring the model to the device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, 
                          momentum = 0.9, weight_decay = 0.001)
    acc_metric = Accuracy(task="multiclass", num_classes = 4).to(device)
    model.to(device)

    # Training the feature extractor
    train_model(model, device, criterion, optimizer, acc_metric, data_loaders)

    # Testing the feature extractor on testing data
    test_model(model, device, data_loaders["test"])


if __name__ == '__main__':
    setup_model()