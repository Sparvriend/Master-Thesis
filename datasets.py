import os
import pandas as pd
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ProjDataset(Dataset):
    """ProjDataset class, which serves as a parent to any dataset
    class defined for this project. The child classes inherit the common
    arguments as well as the __len__ and __getitem__ functions.
    """
    def __init__(self, data_path: str, transform: T.Compose):
        self.img_paths = []
        self.img_labels = []
        self.data_path = data_path
        self.data_type = os.path.normpath(self.data_path).split(os.sep)[2]
        self.transform = transform

    # Function to return the length of the dataset
    def __len__(self) -> int:
        return len(self.img_paths)
    
    # Function to return attributes per item in the dataset
    # The sep_collate function in train.py ensures that for batches,
    # only the label and images are returned.
    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        path = self.img_paths[idx]
        raw_image = Image.open(path)
        # Converting to RGB if image is grayscale, tinyImageNet has grayscale images
        if raw_image.mode == "L":
            raw_image = raw_image.convert("RGB")

        # Transforming the image, it is augmented if from the training dataset
        image = self.transform(raw_image)
        if self.data_type == "train" or self.data_type == "val":
            # Saving the label if it exists.
            label = self.img_labels[idx]
        else:
            label = None
        return path, image, label


class NTZFilterDataset(ProjDataset):
    """NTZFilterDataset class, to use for any dataset formed out of NTZ filter
    images. The class has different possibilities depending on if the type is
    training, validation or testing data.
    """
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)
        self.label_map = {0: "fail_label_crooked_print",
                          1: "fail_label_half_printed",
                          2: "fail_label_not_fully_printed",
                          3: "no_fail"}
        self.n_classes = len(self.label_map)	

        # syn_prop indicates the proportion of the data
        # that should be taken from the NTZSynthetic dataset
        # 1 - syn_prop is the amount that should be taken from the real NTZ dataset
        # self.data_path is the path to the original NTZ dataset.
        # data_path is the path to the synthetic dataset
        self.syn_prop = 0
        data_path = os.path.join("data", "NTZFilterSynthetic", self.data_type)

        # Setting the paths for each image and a label if it concerns training
        # or validation data, labels are enumerated over
        if self.syn_prop != 1:
            for label, dir_name in enumerate(os.listdir(self.data_path)):
                files = os.listdir(os.path.join(self.data_path, dir_name))
                for file_name in files[:int(len(files) * (1 - self.syn_prop))]:
                    self.img_paths.append(os.path.join(self.data_path, 
                                                       dir_name, file_name))
                    if self.data_type == "train" or self.data_type == "val":
                        self.img_labels.append(label)
        
        # Then add synthetic data if self.syn_prop != 0
        if self.syn_prop != 0:
            for label, dir_name in enumerate(os.listdir(data_path)):
                files = os.listdir(os.path.join(data_path, dir_name))
                for file_name in files[:int(len(files) * self.syn_prop)]:
                    self.img_paths.append(os.path.join(data_path, dir_name, 
                                                       file_name))
                    self.img_labels.append(label)
    

class CIFAR10Dataset(ProjDataset):
    """CIFAR10Dataset class, to use for any dataset formed out of CIFAR images.
    Dataset taken from https://www.kaggle.com/competitions/cifar-10/data?select=train.7z
    With minimal adjustements, located in data_processing.py"""
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)

        # Setting the proportion of the dataset to use, since 40k
        # train and 10k validation is way too much
        if self.data_type == "train" or self.data_type == "val":
            self.prop = 1
        elif self.data_type == "test":
            self.prop = 0.01

        # Reading training labels and encoding to integers
        label_df = pd.read_csv(os.path.join("data", "CIFAR10", "trainLabels.csv"))
        unqiue_labels = label_df["label"].unique().tolist()
        self.label_map = {label: i for i, label in enumerate(unqiue_labels)}
        self.n_classes = len(self.label_map)

        # Setting image paths and training labels
        # Randomly sampling like this does not ensure an equal amount of images
        # If self.prop = 1, then this randomly shuffles the data
        files = os.listdir(self.data_path)
        files = random.sample(files, int(len(files) * self.prop))
        for file in files:
            self.img_paths.append(os.path.join(self.data_path, file))
            if self.data_type == "train" or self.data_type == "val":
                # Cutting off png, retrieving the label and appending encoded version
                file = int(file.removesuffix(".png"))
                label_string = label_df.loc[label_df["id"] == file, "label"].values[0]
                self.img_labels.append(self.label_map[label_string])
        
        # Swapping the label map, since it needs to be saved in the opposite way
        # Not {"frog": 0}, but {0: "frog"} for usage in the code.
        self.label_map = {v: k for k, v in self.label_map.items()}


class TinyImageNet200Dataset(ProjDataset):
    """TinyImageNetDataset class, to use for any dataset formed
    out of TinyImageNet images (64x64). Dataset taken from 
    https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
    The name has been adapted to TinyImageNet200"""
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)

        # Setting proportion of images to use from entire dataset
        self.prop = 1

        # Reading labels if training or validation data
        classes = os.listdir(os.path.join("data", "TinyImageNet200", "train"))
        self.label_map = {label: i for i, label in enumerate(classes)}
        self.n_classes = len(self.label_map)

        # Saving data paths and labels for a train dataset
        if self.data_type == "train":
            for c in classes:
                # Randomly selecting a subset of the data
                images = os.listdir(os.path.join(self.data_path, c, "images"))
                images = random.sample(images, int(len(images) * self.prop))	
                for image in images:
                    # Getting label from label map and saving data path
                    self.img_paths.append(os.path.join(self.data_path, c, "images", image))
                    self.img_labels.append(self.label_map[c])
        else:
            # Randomly selecting a subset of the data for val/test dataset
            images = os.listdir(os.path.join(self.data_path, "images"))
            images = random.sample(images, int(len(images) * self.prop))

        # Saving data paths and labels for a validation dataset
        if self.data_type == "val":
            # Loading label file as a pandas dataframe
            columns = ["filename", "class", "x1", "y1", "x2", "y2"]
            label_file = os.path.join(self.data_path, "val_annotations.txt")
            label_df = pd.read_csv(label_file, sep = "\t", header = None, names = columns)
            for image in images:
                # Getting label from label df and then label map and saving data path
                self.img_paths.append(os.path.join(self.data_path, "images", image))
                label = label_df.loc[label_df["filename"] == image, "class"].values[0]
                self.img_labels.append(self.label_map[label])
        
        # Saving data paths for a test dataset
        if self.data_type == "test":
            # Saving image path
            for image in images:
                self.img_paths.append(os.path.join(self.data_path, "images", image))

        # Swapping the label map
        self.label_map = {v: k for k, v in self.label_map.items()}


class ImageNet10Dataset(ProjDataset):
    def __init__(self, data_path: str, transform: T.Compose):
        super().__init__(data_path, transform)
        classes = os.listdir(os.path.join("data", "ImageNet10", "train"))
        self.label_map = {label: i for i, label in enumerate(classes)}
        self.n_classes = len(self.label_map)

        # For training images add both the label and the image
        if self.data_type == "train" or self.data_type == "val":
            for label, dir_name in enumerate(os.listdir(self.data_path)):
                for file_name in os.listdir(os.path.join(self.data_path, dir_name)):
                    self.img_paths.append(os.path.join(self.data_path, 
                                                   dir_name, file_name))
                    self.img_labels.append(label)
        # For testing images add only the image
        else:
            files = os.listdir(self.data_path)
            for file in files:
                self.img_paths.append(os.path.join(self.data_path, file))

        # Swapping the label map
        self.label_map = {v: k for k, v in self.label_map.items()}