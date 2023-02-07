import shutil
import os
from os import listdir
import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance

# First forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = "data/train"; TEST_DESTINATION = "data/test"; VAL_DESTINATION = "data/val"

# Forming file paths for source of data, as well as the different classes
DATA_LOCATION = "NTZ_filter_label_data"
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed", "fail_label_not_fully_printed", "no_fail"]

# Augmentation of data
# Currently there are 15 different augmentations being applied to each image in parallel
# The rotated and flipped images are then also have all augmentations applied to them
# Resulting in 48 * 42 (16 original + 13 for rotated and + 13 for flipped) = 2016 images per class instead of the original 48

def augment_data():
    # Defining augmentation techniques: Rotate, HorizontalFlip, Grayscale, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness
    augmentation_techniques = [[Image.Image.rotate, 180, "_rotated"], [Image.Image.transpose, Image.FLIP_TOP_BOTTOM, "_flipped"], [Image.Image.convert, 'L', "_grayscaled"],
                                [ImageOps.autocontrast, None, "_autocontrasted"],  [ImageOps.invert, None, "_inverted"], [ImageOps.equalize, None, "_equalized"],
                                [ImageOps.solarize, 128, "_solarized128"], [ImageOps.posterize, 4, "_posterized4"], [ImageOps.posterize, 3, "_posterized3"],
                                [ImageEnhance.Color, 0.5, "_colored05"], [ImageEnhance.Contrast, 0.5, "_contrasted05"], [ImageEnhance.Brightness, 0.5, "_brightened05"],
                                [ImageEnhance.Brightness, 0.75, "_brightened075"], [ImageEnhance.Sharpness, 0, "_blurred"], [ImageEnhance.Sharpness, 2, "_sharpened"]]

    # Cutout is not generally applicable, since it might remove some vital information from the image
    # -> Similar for ShearX/Y and TranslateX/Y
    # Sample pairing also not applicable since the images can become unrecognizable

    # Applying augmentations only in parallel
    for data_class in DATA_CLASSES:
        # Making file selection
        path = os.path.join(TRAIN_DESTINATION, data_class)
        files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]

        apply_augmentation(files, path, augmentation_techniques)    

    # Applying augmentations in parallel and iteratively     
    for data_class in DATA_CLASSES:
        # First selecting all files like normal
        path = os.path.join(TRAIN_DESTINATION, data_class)
        files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Then making a selection based on the rotated or flipped version
        for name in ['_rotated.bmp', '_flipped.bmp']:
            cleaned_files = []
            for img_file in files:
                if img_file.endswith(name):
                    cleaned_files.append(img_file)

            # Applying augmentation with flip and rotation cut off, since that is already applied to the images.
            apply_augmentation(cleaned_files, path, augmentation_techniques[2:])
            
def apply_augmentation(files, path, augmentation_techniques):
    # Opening images
    for img_name in files:
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        img_path = img_path.replace(".bmp", "")

        # Apply each augmentation once
        for augmentation in augmentation_techniques:
            # Apply technique to image and save it
            # Enhance requires a different check and ImageOps sometimes do not have inputs
            if augmentation[1] == None:
                augmented_img = augmentation[0](img)
            elif augmentation[0].__name__ in ["Color", "Contrast", "Brightness", "Sharpness"]:
                enhancer = augmentation[0](img)
                augmented_img = enhancer.enhance(augmentation[1])
            else:
                augmented_img = augmentation[0](img, augmentation[1])
            augmented_img.save(img_path + augmentation[2] + ".bmp")

def split_and_move():
    # 80/10/10 split for training/validation/testing data
    train_split = 0.8; val_split = 0.1; test_split = 0.1

    # Setting the data origin location
    data_location = "NTZ_filter_label_data"

    # Setting a maximum of samples for each class (60)
    max_samples = 60

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

def set_grayscale_to_rgb():
    for data_class in DATA_CLASSES:
        # Making file selection
        path = os.path.join(TRAIN_DESTINATION, data_class)
        files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Making a selection based on grayscale images
        cleaned_files = []
        for img_file in files:
            if img_file.endswith('grayscaled.bmp'):
                cleaned_files.append(img_file)
        
        # Saving the grayscaled images as RGB
        for img_file in cleaned_files:
            img_path = os.path.join(path, img_file)
            img = Image.open(img_path)
            RGB_img = img.convert('RGB')
            RGB_img.save(img_path)

if __name__ == '__main__':
    # Make sure to run split and move before augment
    # If not, augmentations will be applied to augmented images unregulated
    split_and_move()
    augment_data()
    set_grayscale_to_rgb()