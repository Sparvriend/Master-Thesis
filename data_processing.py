import shutil
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
from PIL import Image
from PIL import ImageOps
# First forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = "data/train"; TEST_DESTINATION = "data/test"; VAL_DESTINATION = "data/val"

# Forming file paths for source of data, as well as the different classes
DATA_LOCATION = "NTZ_filter_label_data"
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed", "fail_label_not_fully_printed", "no_fail"]

# TODO: 
# - Apply augmentations on augmentations in augment_data()
# - Finish augmentation techniques in augment_data()

def augment_data():
    # Set a maximum number of augmentations to perform after each other
    max_augmentations = 4
    augmentation_techniques = [
                                [Image.Image.rotate, 180, "_rotated"], [Image.Image.transpose, Image.FLIP_TOP_BOTTOM, "_flipped"], [Image.Image.convert, 'L', "_grayscaled"],
                                [ImageOps.autocontrast, None, "_autocontrasted"],  [ImageOps.invert, None, "_inverted"], [ImageOps.equalize, None, "_equalized"],
                                
                                
                                ]

    # Implemented: Rotate, HorizontalFlip, Grayscale, AutoContrast, Invert, Equalize
    # Not implemented: ShearX/Y, TranslateX/Y, , Solarize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout and Sample Pairing

    for data_class in DATA_CLASSES:
        path = TRAIN_DESTINATION + "/" + data_class
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            img_path = path + "/" + file
            img = Image.open(img_path)
            # Apply each augmentation one time first
            for item in augmentation_techniques:
                # Apply technique to image and save it
                augmented_img = item[0](img, item[1])
                augmented_img.save(img_path + item[2] + ".bmp")            


def split_and_move():
    # 80/10/10 split for training/validation/testing data
    train_split = 0.8; val_split = 0.1; test_split = 0.1

    # Setting the data origin location
    data_location = "NTZ_filter_label_data"

    # Setting a maximum of samples for each class (60)
    max_samples = 60

    for data_class in DATA_CLASSES:
        # Combining path and listing all files in the path
        path = data_location + "/" + data_class
        files = [f for f in listdir(path) if isfile(join(path, f))]
        # Removal of .ivp files and other things
        cleaned_files = []
        for file in files:
            if file.endswith('bmp'):
                cleaned_files.append(file)
        
        # Shuffling the images [WHEN REDOING THE MOVE, ENSURE THE MODEL IS NEVER TRAINED ON VALIDATION/TESTING DATA FROM PREVIOUS SPLIT]
        random.shuffle(cleaned_files)

        # Selecting only max_samples per class
        if len(cleaned_files) > max_samples:
            cleaned_files = cleaned_files[:max_samples]

        # Splitting data into train, test and validation
        train, val, test = np.split(cleaned_files, [int(len(cleaned_files)*train_split), int(len(cleaned_files)*(train_split+val_split))])

        # Removing all old files present in the folders
        for type in [TRAIN_DESTINATION, TEST_DESTINATION, VAL_DESTINATION]:
            path = type + "/" + data_class
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for file in files:
                file_path = path + "/" + file
                os.unlink(file_path)

        # Moving all files to appropriate directories
        for train_file in train:
            destination_path = TRAIN_DESTINATION + "/" + data_class
            file_path = data_location + "/" + data_class + "/" + train_file
            shutil.copy(file_path, destination_path)
        
        for val_file in val:
            destination_path = VAL_DESTINATION + "/" + data_class
            file_path = data_location + "/" + data_class + "/" + val_file
            shutil.copy(file_path, destination_path)

        for test_file in test:
            destination_path = TEST_DESTINATION + "/" + data_class
            file_path = data_location + "/" + data_class + "/" + test_file
            shutil.copy(file_path, destination_path)

if __name__ == '__main__':
    # Make sure to run split and move before augment
    # If not, augmentations will be applied to augmented images unregulated
    split_and_move()
    augment_data()