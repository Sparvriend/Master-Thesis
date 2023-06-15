import numpy as np
import os
import random
import shutil
import zipfile
from tqdm import tqdm

# Forming file paths for destinations of training, testing and validation data
TRAIN_DESTINATION = os.path.join("data", "NTZFilter", "train")
VAL_DESTINATION = os.path.join("data", "NTZFilter", "val")
TEST_DESTINATION = os.path.join("data", "NTZFilter", "test")

# Forming file paths for source of data, as well as the different classes
DATA_LOCATION = "NTZ_filter_label_data"
DATA_CLASSES = ["fail_label_crooked_print", "fail_label_half_printed",
                "fail_label_not_fully_printed", "no_fail"]


def split_and_move_NTZ():
    """Function that takes images from NTZ_filter_label_data and splits them
    into training, testing and validation data. In this function it is assumed
    that the directories to paste the images in already exist in data/*/*.
    """
    # 80/10/10 split for training/validation/testing data,
    # test split is 1 - train_split - val_split
    train_split = 0.8
    val_split = 0.1

    # Setting a maximum of samples for each class (70)
    max_samples = 70

    for data_class in DATA_CLASSES:
        # Combining path and listing all files in the path
        path = os.path.join(DATA_LOCATION, data_class)
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
        train_prop = int(len(cleaned_files) * train_split)
        val_prop = int(len(cleaned_files) * (train_split + val_split))
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
            file_path = os.path.join(DATA_LOCATION, data_class, train_file)
            shutil.copy(file_path, destination_path)
        
        for val_file in val:
            destination_path = os.path.join(VAL_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION, data_class, val_file)
            shutil.copy(file_path, destination_path)

        for test_file in test:
            destination_path = os.path.join(TEST_DESTINATION, data_class)
            file_path = os.path.join(DATA_LOCATION, data_class, test_file)
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


def check_correspondence(subset_class_imgs: list, class_imgs: list,
                         class_name: str):
    """Function that checks if all images in a subset are images
    in a class.
    
    Args:
        subset_class_imgs: List of images to check
        class_imgs: List of images in the class
        class_name: Name of the class
    """
    flag = True
    for img in subset_class_imgs:
        if img not in class_imgs:
            print("Resampling for class " + str(class_name))
            flag = False
            break
    return flag


def get_sample_data_ImageNet():
    """Function that takes the ImageNet dataset, extracts training images
    from it corresponding to training images that exist in tinyImageNet:
    https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
    From the training images, 10 are randomly selected and added to a special
    testing set. This testing's set only purpose is to show of explainability
    and uncertainty usability; since it concerns training data, it is not
    usable for assesing model performance. Testing/validation data extraction
    from ImageNet is difficult since the filenames in tinyImageNet do not
    correspond to those from ImageNet, so it is uncertain if the class
    from an image is one trained on in tinyImageNet.
    """
    # It is assumed that the ImageNet dataset is available in data/ImageNet
    # It is also assumed that the tinyImageNet dataset is available in
    # data/tinyImageNet200
    tinyImageNet_path = "data/TinyImageNet200/train"
    ImageNet_path = "data/ImageNet/ILSVRC/Data/CLS-LOC/train/"
    ImageNetSample_path = "data/ImageNetSamples"
    tinyImageNet_classes = os.listdir(tinyImageNet_path)
    os.mkdir(ImageNetSample_path)
    n = 10

    # For each class, randomly select n images
    # and copy them to the ImageNetSamples directory
    for class_name in tinyImageNet_classes:
        os.mkdir(os.path.join(ImageNetSample_path, class_name))
        class_imgs_tinyImageNet = os.listdir(os.path.join(tinyImageNet_path, class_name))
        class_imgs_ImageNet = os.listdir(os.path.join(ImageNet_path, class_name))
        subset_class_imgs = random.sample(class_imgs_tinyImageNet, n)

        # Forming a loop in which the subset is checked for correspondence
        # Resample until all images are tinyImageNet images
        flag = check_correspondence(subset_class_imgs, class_imgs_ImageNet, class_name)
        while flag == False:
            flag = check_correspondence(subset_class_imgs, class_imgs_ImageNet, class_name)

        # Then copying all
        for img in subset_class_imgs:
            shutil.copy(os.path.join(ImageNet_path, class_name, img), os.path.join(ImageNetSample_path, class_name, img))


if __name__ == '__main__':
    # Make sure to run create_dirs() before running split_and_move()
    #create_dirs()
    #split_and_move_NTZ()
    #split_and_move_CIFAR()
    get_sample_data_ImageNet()
