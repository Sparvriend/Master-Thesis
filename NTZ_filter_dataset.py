from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import copy
import os

# NTZFilterDataset class, to use for any dataset formed out of NTZ filter images
class NTZFilterDataset(Dataset):
    def __init__(self, data_path, transform):
        self.img_paths = []
        self.img_labels = []
        self.data_type = os.path.normpath(data_path).split(os.sep)[1]
        self.transform = transform  

        # If testing data, then only create the paths
        if data_path.endswith("test_no_label"):
            for file_name in os.listdir(data_path):
                # Omitting the .txt file with test predictions if it exists
                if file_name.endswith(".bmp"):
                    self.img_paths.append(os.path.join(data_path, file_name))

        # If training/validating data, also create the labels and augmentation for training
        else:
            # Listing class directories
            dir_names = []
            for dir_name in os.listdir(data_path):
                if os.path.isdir(os.path.join(data_path, dir_name)):
                    dir_names.append(dir_name)
            
            # Setting the paths for each image and a label for each image
            label = 0
            for dir_name in dir_names:
                for file_name in os.listdir(os.path.join(data_path, dir_name)):
                    self.img_paths.append(os.path.join(data_path, dir_name, file_name))
                    self.img_labels.append(label)
                label += 1

    # Function to return the length of the dataset
    def __len__(self):
        return len(self.img_paths)

    # Function to return attributes per item in the dataset
    # The sep_collate function in train.py ensures that for batches, only the label and images are returned.
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        raw_image = Image.open(path)

        # Augmenting the image if it is from the training dataset
        if self.data_type == "train":
            image = self.transform(raw_image)
            image_copy = copy.deepcopy(image)

            # Denormalizing the augmented image and saving it
            denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225])
            image_copy = denormalize(image_copy)
            to_PIL = transforms.ToPILImage()
            PIL_image = to_PIL(image_copy)
            img_name = os.path.normpath(path).split(os.sep)[-1]
            PIL_image.save(os.path.join("augmented_images", img_name))
        else:
            image = transforms.Compose(self.transform.transforms[1:])(raw_image)

        if self.img_labels == []:
            label = None
        else:
            label = self.img_labels[idx]

        return path, image, label