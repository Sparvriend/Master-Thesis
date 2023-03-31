import os
import pandas as pd
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T  

class CIFARDataset(Dataset):
    """CIFARDataset class, to use for any dataset formed out of CIFAR images."""
    def __init__(self, data_path: str, transform: T.Compose):
        self.img_paths = []
        self.img_labels = []
        self.data_type = os.path.normpath(data_path).split(os.sep)[1]
        self.transform = transform
        self.prop = 0.1

        # Reading training labels
        label_df = pd.read_csv(os.path.join(data_path, "trainLabels.csv"))

        # Setting image paths and training labels
        files = os.listdir(data_path)
        files = random.sample(files, int(len(files), self.prop))
        for file in files:
            self.img_paths.append(os.path.join(data_path, file))
            if self.data_type == "train" or self.data_type == "val":
                self.img_labels.append(label_df.loc[label_df["id"] == file, "label"].values[0])

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