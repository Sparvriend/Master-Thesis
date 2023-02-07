import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from NTZ_filter_dataset import NTZFilterDataset
from torch.optim import lr_scheduler

# File paths
TRAIN_PATH = "data/train"; VAL_PATH = "data/val"

# General parameters for training
BATCH_SIZE = 32
EPOCHS = 100
SHUFFLE = True
NUM_WORKERS = 4

def train_fe_one_run():


def train_feature_extractor(model, criterion, optimizer, scheduler, train_loader, val_loader):


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
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = SHUFFLE, num_workers = NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = SHUFFLE, num_workers = NUM_WORKERS)

    #analyse_dataset(train_data)

    # First using the ready made model from Pytorch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    model.to(device)

    train_feature_extractor(model, criterion, optimizer, scheduler, train_loader, val_loader)

# Function to analyse if the NTZFilterDataset class works properly
def analyse_dataset(dataset):
    print(len(dataset))
    for idx, item in enumerate(dataset):
        print(idx)
        print("name = " + str(item[0])); print("label = " + str(item[2]))
        print("img shape = " + str(item[1].shape)); print("augmentations = " + str(item[3]) + "\n")

if __name__ == '__main__':
    setup_feature_extractor()
    



