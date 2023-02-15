import shutil
import os
from os import listdir
import numpy as np
import random

# First forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = "data/train"; VAL_DESTINATION = "data/val"; TEST_DESTINATION = "data/test"; TEST_PREDICTIONS_DESTINATION = "data/test_predictions"

# Forming file paths for source of data, as well as the different classes
DATA_LOCATION = "NTZ_filter_label_data"
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed", "fail_label_not_fully_printed", "no_fail"]

# Function that takes images from NTZ_filter_label_data and splits them into training, testing and validation data
# In this function it is assumed that the directories to paste the images in already exist in data/*/*
def split_and_move():
    # 80/10/10 split for training/validation/testing data, test split is 1 - train_split - val_split
    train_split = 0.8; val_split = 0.1

    # Setting the data origin location
    data_location = "NTZ_filter_label_data"

    # Setting a maximum of samples for each class (60)
    max_samples = 70

    for data_class in DATA_CLASSES:
        # Combining path and listing all files in the path
        path = os.path.join(data_location, data_class)
        files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]
        # Removal of .ivp files and other things
        cleaned_files = []
        for img_file in files:
            if img_file.endswith('bmp'):
                cleaned_files.append(img_file)
        
        # Shuffling the images [WHEN REDOING THE MOVE, ENSURE THE MODEL IS NEVER TRAINED ON VALIDATION/TESTING DATA FROM PREVIOUS SPLIT]
        random.shuffle(cleaned_files)

        # Selecting only max_samples per class
        if len(cleaned_files) > max_samples:
            cleaned_files = cleaned_files[:max_samples]

        # Splitting data into train, test and validation
        train, val, test = np.split(cleaned_files, [int(len(cleaned_files)*train_split), int(len(cleaned_files)*(train_split+val_split))])

        # Removing all old files present in the folders
        for destination_type in [TRAIN_DESTINATION, TEST_DESTINATION, VAL_DESTINATION]:
            path = os.path.join(destination_type, data_class)
            files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]
            for old_file in files:
                file_path = os.path.join(path, old_file)
                os.unlink(file_path)

        # Moving all files to appropriate directories
        for train_file in train:
            destination_path = os.path.join(TRAIN_DESTINATION, data_class)
            file_path = os.path.join(data_location, data_class, train_file)
            shutil.copy(file_path, destination_path)
        
        for val_file in val:
            destination_path = os.path.join(VAL_DESTINATION, data_class)
            file_path = os.path.join(data_location, data_class, val_file)
            shutil.copy(file_path, destination_path)

        for test_file in test:
            destination_path = os.path.join(TEST_DESTINATION, data_class)
            file_path = os.path.join(data_location, data_class, test_file)
            shutil.copy(file_path, destination_path)

def create_dirs():
    # Creating directories for training, testing and validation data
    for destination_type in [TRAIN_DESTINATION, VAL_DESTINATION, TEST_DESTINATION, TEST_PREDICTIONS_DESTINATION]:
        for data_class in DATA_CLASSES:
            path = os.path.join(destination_type, data_class)
            if not os.path.exists(path):
                os.makedirs(path)
    # Creating the temproary augmentation directory, to check augmentation images
    if not os.path.exists("augmented_images"):
        os.makedirs("augmented_images")

if __name__ == '__main__':
    # Make sure to run create_dirs() before running split_and_move()
    create_dirs()
    split_and_move()