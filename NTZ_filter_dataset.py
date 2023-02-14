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

        # Setting the paths for each image and a label if it concerns training or validation data, labels are enumerated over
        for label, dir_name in enumerate(os.listdir(data_path)):
            for file_name in os.listdir(os.path.join(data_path, dir_name)):
                self.img_paths.append(os.path.join(data_path, dir_name, file_name))
                if self.data_type == "train" or self.data_type == "val":
                    self.img_labels.append(label)

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

            # Saving the label
            label = self.img_labels[idx]
        else:
            # Performing the normal transform without augmentation and saving the label if it exists
            image = transforms.Compose(self.transform.transforms[1:])(raw_image)
            if self.data_type == "val":
                label = self.img_labels[idx]
            else:
                label = None

        return path, image, label