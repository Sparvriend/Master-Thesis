import os
import pandas as pd
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T  

class CIFAR10Dataset(Dataset):
    """CIFAR10Dataset class, to use for any dataset formed out of CIFAR images."""
    def __init__(self, data_path: str, transform: T.Compose):
        self.img_paths = []
        self.img_labels = []
        self.data_type = os.path.normpath(data_path).split(os.sep)[2]
        self.transform = transform

        # Setting the proportion of the dataset to use, since 40k
        # train and 10k validation is way too much
        if self.data_type == "train" or self.data_type == "val":
            self.prop = 0.01
        elif self.data_type == "test":
            self.prop = 0.0003

        # Reading training labels and encoding to integers
        label_df = pd.read_csv(os.path.join("data", "CIFAR10", "trainLabels.csv"))
        unqiue_labels = label_df["label"].unique().tolist()
        self.label_map = {label: i for i, label in enumerate(unqiue_labels)}

        # Setting image paths and training labels
        files = os.listdir(data_path)
        files = random.sample(files, int(len(files) * self.prop))
        for file in files:
            self.img_paths.append(os.path.join(data_path, file))
            if self.data_type == "train" or self.data_type == "val":
                # Cutting off png, retrieving the label and appending encoded version
                file = int(file.removesuffix(".png"))
                label_string = label_df.loc[label_df["id"] == file, "label"].values[0]
                self.img_labels.append(self.label_map[label_string])
        
        # Swapping the label map, since it needs to be saved in the opposite way
        # Not {'frog': 0}, but {'0': 'frog'} for usage in the code.
        self.label_map = {v: k for k, v in self.label_map.items()}

    # Function to return the length of the dataset
    def __len__(self) -> int:
        return len(self.img_paths)
    
    # Function that returns path, image and the label
    # It works by a collate function defined in a pytorch dataloader
    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        path = self.img_paths[idx]
        image = Image.open(path)
        image = self.transform(image)
        if self.data_type == "train" or self.data_type == "val":
            label = self.img_labels[idx]
        else:
            label = None
        return path, image, label