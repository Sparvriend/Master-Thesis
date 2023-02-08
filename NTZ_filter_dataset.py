from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

# NTZFilterDataset class, to use for any dataset formed out of NTZ filter images
class NTZFilterDataset(Dataset):
    def __init__(self, data_path, transform):
        self.img_paths = []
        self.img_labels = []
        self.augmentations = []
        self.transform = transform

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
        
        # Function that reads out the augmentations for each image and saves them in an array
        if data_path.endswith("train"):
            for img_path in self.img_paths:
                augments = []
                stripped = img_path.split("_")
                # If the string contains rotated or flipped in the 1 before last position, there have been 2 augments
                if stripped[len(stripped)-2] == "rotated" or stripped[len(stripped)-2] == "flipped":
                    augments.append(stripped[len(stripped)-2])
                # This statement only adds the second (or first) augmentation if they are present in the name (e.g. it omits the image without augmentations)
                if stripped[len(stripped)-3].startswith("No") or stripped[len(stripped)-2].startswith("No") :
                    augments.append(stripped[len(stripped)-1].replace(".bmp", ""))
                self.augmentations.append(augments)
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        raw_image = Image.open(self.img_paths[idx])
        image = self.transform(raw_image)
        label = self.img_labels[idx]
        if self.augmentations == []:
            augmentations = None
        else:
            augmentations = self.augmentations[idx]
        return path, image, label, augmentations