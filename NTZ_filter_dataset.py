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
        print(dir_names)
        
        # Setting the paths for each image and a label for each image
        label = 0
        for dir_name in dir_names:
            for file_name in os.listdir(os.path.join(data_path, dir_name)):
                # Omitting the augmentation textfile
                if file_name.endswith(".bmp"):
                    self.img_paths.append(os.path.join(data_path, dir_name, file_name))
                    self.img_labels.append(label)
            label += 1
        
        # if data_path.endswith("train"):
        #     # Setting the augmentations for each image
        #     for dir_name in dir_names:
        #         # Opening augmentations file and saving the line by line info
        #         augmentation_file = open(os.path.join(data_path, dir_name, dir_name + "_augmentation_data.txt"), "r")
        #         line_by_line = augmentation_file.readlines()
        #         # Splitting the augmentations and saving them to a list, since there can be more than one
        #         for line in line_by_line:
        #             for augmentation in line.split()[1:]:
        #                 self.augmentations.append(augmentation)
        
        if data_path.endswith("train"):
            for img_path in self.img_paths:
                seperator = "_"
                stripped = img_path[::-1].split(seperator, 2)[0]
                stripped = stripped[::-1]
                print(stripped)
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        raw_image = Image.open(self.img_paths[idx])
        image = self.transform(raw_image)
        label = self.img_labels[idx]
        augmentations = self.augmentations[idx]
        return image, label, augmentations