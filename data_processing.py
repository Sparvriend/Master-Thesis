from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import random
from tqdm import tqdm
import shutil
import zipfile

from synthetic_data import setup_data_generation 

# Forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = os.path.join("data", "NTZFilter", "train")
VAL_DESTINATION = os.path.join("data", "NTZFilter", "val")
TEST_DESTINATION = os.path.join("data", "NTZFilter", "test")


# Forming file paths for source of data, as well as the different classes
DATA_LOCATION_NTZ = os.path.join("raw_data", "NTZ_filter_label_data")
DATA_LOCATION_SYN = os.path.join("raw_data", "NTZ_filter_synthetic", "synthetic_data")
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed",
                "fail_label_not_fully_printed", "no_fail"]


# Forming train/validation splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


def split_and_move_NTZ():
    """Function that takes images from NTZ_filter_label_data and splits them
    into training, testing and validation data. In this function it is assumed
    that the directories to paste the images in already exist in data/*/*.
    """
    # 80/10/10 split for training/validation/testing data,
    # test split is 1 - train_split - val_split

    # Setting a maximum of samples for each class (70)
    max_samples = 70

    for data_class in DATA_CLASSES:
        # Combining path and listing all files in the path
        path = os.path.join(DATA_LOCATION_NTZ, data_class)
        files = [f for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))]
        # Removal of .ivp files and other things
        cleaned_files = []
        for img_file in files:
            if img_file.endswith("bmp"):
                cleaned_files.append(img_file)
        
        # Shuffling the images [WHEN REDOING THE MOVE, ENSURE THE MODEL IS
        # NEVER TRAINED ON VALIDATION/TESTING DATA FROM PREVIOUS SPLIT]
        random.shuffle(cleaned_files)

        # Selecting only max_samples per class
        if len(cleaned_files) > max_samples:
            cleaned_files = cleaned_files[:max_samples]

        # Splitting data into train, test and validation
        train_prop = int(len(cleaned_files) * TRAIN_SPLIT)
        val_prop = int(len(cleaned_files) * (TRAIN_SPLIT + VAL_SPLIT))
        train, val, test = np.split(cleaned_files, [train_prop, val_prop])

        # Removing all old files present in the folders
        for destination_type in [TRAIN_DESTINATION, TEST_DESTINATION,
                                 VAL_DESTINATION]:
            path = os.path.join(destination_type, data_class)
            files = [f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]
            for old_file in files:
                file_path = os.path.join(path, old_file)
                os.unlink(file_path)

        # Moving all files to appropriate directories
        for train_file in train:
            destination_path = os.path.join(TRAIN_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION_NTZ, data_class, train_file)
            shutil.copy(file_path, destination_path)
        
        for val_file in val:
            destination_path = os.path.join(VAL_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION_NTZ, data_class, val_file)
            shutil.copy(file_path, destination_path)

        for test_file in test:
            destination_path = os.path.join(TEST_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION_NTZ, data_class, test_file)
            shutil.copy(file_path, destination_path)
        

def split_and_move_CIFAR():
    """Function that moves images from the CIFAR dataset to a
    validation directory, from the training directory, based
    on a validation split percentage. The function assumes that
    a directory called CIFAR10 exists in the data directory,
    including a train directory with training images.

    The images can be downloaded from Kaggle:
    https://www.kaggle.com/competitions/cifar-10/data?select=train.7z
    """
    # Defining amount of data to use for validation
    val_split = 0.2

    # Extracting the data from the zip directories directly results in
    # the images existing in train/train, so this is checked and corrected
    if os.path.exists(os.path.join("data", "CIFAR10", "train", "train")):
        train_path = os.path.join("data", "CIFAR10", "train", "train")
        files = os.listdir(train_path)
        for file in files:
            file_path = os.path.join(train_path, file)
            destination_path = os.path.join("data", "CIFAR10", "train", file)
            shutil.move(file_path, destination_path)
        os.rmdir(train_path)

    # Selecting the amount of files to move from the train directory
    train_path = os.path.join("data", "CIFAR10", "train")
    files = os.listdir(train_path)
    move_amount = int(len(files)*val_split)
    files = files[:move_amount]

    # Creating validation directory   
    if not os.path.exists(os.path.join("data", "CIFAR10", "val")):
        os.mkdir(os.path.join("data", "CIFAR10", "val"))

    # Moving files to validation directory
    for file in files:
        file_path = os.path.join(train_path, file)
        destination_path = os.path.join("data", "CIFAR10", "val", file)
        shutil.move(file_path, destination_path)


def create_dirs():
    """Function that creates all directories required for running modules
    of the project. It is assumed that the folder NTZ_filter_label_data
    is already available and is ordered correctly. 
    """
    # Creating some directories for setting up the NTZFilter dataset.
    for destination_type in [TRAIN_DESTINATION, VAL_DESTINATION, 
                             TEST_DESTINATION]:
        for data_class in DATA_CLASSES:
            path = os.path.join(destination_type, data_class)
            if not os.path.exists(path):
                os.makedirs(path)
    
    # Creating all directories needed in Results/
    dirs = ["Experiment-Results", "Explainability-Results", "Test-Predictions"]
    for dir in dirs:
        path = os.path.join("Results", dir)
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == '__main__':
    # Make sure to run create_dirs() before running split_and_move()
    create_dirs()
    split_and_move_NTZ()
    split_and_move_CIFAR()