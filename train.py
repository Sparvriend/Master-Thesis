import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from NTZ_filter_dataset import NTZFilterDataset

# File paths
TRAIN_PATH = "data/train"; VAL_PATH = "data/val"

# General parameters for training
BATCH_SIZE = 32
EPOCHS = 100
SHUFFLE = True
NUM_WORKERS = 4

def train_feature_extractor():
    # First setting the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')
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

# Function to analyse if the NTZFilterDataset class works properly
def analyse_dataset(dataset):
    print(len(dataset))
    for idx, item in enumerate(dataset):
        print("label = " + str(item[1]))
        print(item[0].shape)
        print(idx)
        print("augmentations = " + str(item[2]))
        print("\n\n\n")        
        
    print("Everything works fine")

if __name__ == '__main__':
    train_feature_extractor()
    



