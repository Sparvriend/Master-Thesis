import shutil
import os
from os import listdir
import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance

# First forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = "data/train"; VAL_DESTINATION = "data/val"; TEST_DESTINATION = "data/test"; TEST_NO_LABEL_DESTINATION = "data/test_no_label"

# Forming file paths for source of data, as well as the different classes
DATA_LOCATION = "NTZ_filter_label_data"
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed", "fail_label_not_fully_printed", "no_fail"]

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
            shutil.copy(file_path, TEST_NO_LABEL_DESTINATION)

if __name__ == '__main__':
    # Make sure to run split and move before augment
    split_and_move()