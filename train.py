import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from NTZ_filter_dataset import NTZFilterDataset
from torchmetrics import Accuracy
import copy
from tqdm import tqdm

# TODO:
# - Use Tensorboard for implementing different experiments (Ratnajit would send tutorial)
# - Use on the fly augmentation instead of fixed augmentations (per epoch) for each image (Ratnajit would send tutorial)
# - Combine validation and training functions?

# File paths
TRAIN_PATH = "data/train"; VAL_PATH = "data/val"

# General parameters for training
BATCH_SIZE = 8
EPOCHS = 10
SHUFFLE = True
NUM_WORKERS = 4

def validate_fe(model, device, criterion, acc_metric, data_loader):
    # Set model to evaluating
    model.eval()

    # Set model loss
    loss_over_epoch = 0
    correct = 0

    # Unpacking all inputs and labels to run on the model in batches
    # The model weights should not be updated, hence running it with no_grad()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Getting model output and computing the loss
            model_output = model(inputs)
            loss = criterion(model_output, labels)
            correct += acc_metric(model_output.argmax(dim=1), labels).item()

            # Adding the loss over the epoch
            loss_over_epoch += loss.item()

    accuracy = correct / len(data_loader)
    print("Validation loss = " + str(loss_over_epoch))
    print("Validation accuracy = " + str(accuracy))

def train_fe_one_epoch(model, device, criterion, optimizer, acc_metric, data_loader):
    # Set model to training phase
    model.train()

    # Setting model loss and initializing accuracy
    loss_over_epoch = 0
    correct = 0
    
    # Unpacking all inputs and labels to run on the model in batches
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Resetting model weights and getting the model output
        optimizer.zero_grad()
        model_output = model(inputs)

        # Computing the loss and updating the model with a backwards pass
        loss = criterion(model_output, labels)
        correct += acc_metric(model_output.argmax(dim=1), labels).item()
        loss.backward()
        optimizer.step()

        # Adding the loss over the epoch
        loss_over_epoch += loss.item()

    accuracy = correct / len(data_loader)
    print("Training loss = " + str(loss_over_epoch))
    print("Training accuracy = " + str(accuracy))

def train_feature_extractor(model, device, criterion, optimizer, acc_metric, train_loader, val_loader):

    # Setting the preliminary model to be the best model
    best_model = copy.deepcopy(model)

    for i in range(EPOCHS):
        print("On epoch " + str(i))
        print("Training phase")
        train_fe_one_epoch(model, device, criterion, optimizer, acc_metric, train_loader)

        print("Validation phase")
        validate_fe(model, device, criterion, acc_metric, val_loader)

        # Some metric to check if the validation accuracy is higher than in the previous function
        # Change model back to old model if validation accuracy is worse
        # Change best model to new model if validation accuracy is better


def setup_feature_extractor():
    # First setting the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))

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

    # Creating data loaders for validation and training data
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, collate_fn = sep_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, collate_fn = sep_collate, shuffle = SHUFFLE, num_workers = NUM_WORKERS)

    # analyse_dataset(train_data)

    # First using the ready made model from Pytorch
    model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
    # Replacing the output classification layer with a 4 class version
    model.classifier[1] = nn.Linear(in_features=1280, out_features=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    acc_metric = Accuracy(task="multiclass", num_classes=4)
    model.to(device)

    train_feature_extractor(model, device, criterion, optimizer, acc_metric, train_loader, val_loader)

def sep_collate(batch):
    _, images, labels, _ = zip(*batch)
    tensor_labels = []

    # Labels are ints, which is why they need to be converted to tensors before being entered into a torch stack
    for label in labels:
        tensor_labels.append(torch.tensor(label))

    # Converting both images and label lists of tensors to torch stacks
    images = torch.stack(list(images), dim = 0)
    labels = torch.stack(list(tensor_labels), dim = 0)

    return images, labels

# Function to analyse if the NTZFilterDataset class works properly
def analyse_dataset(dataset):
    print(len(dataset))
    for idx, item in enumerate(dataset):
        print(idx)
        print("name = " + str(item[0])); print("label = " + str(item[2]))
        print("img shape = " + str(item[1].shape)); print("augmentations = " + str(item[3]) + "\n")

if __name__ == '__main__':
    setup_feature_extractor()
    



