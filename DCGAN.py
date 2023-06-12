import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import get_data_loaders, get_transforms, get_device, remove_predicts
from datasets import NTZFilterDataset


class DCGAN_generator(nn.Module):
    def __init__(self):
        super(DCGAN_generator, self).__init__()
        self.nz = 100
        self.ngf = 64
        self.nc = 3
        self.forward_call = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # Size = (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # Size = (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # Size = (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # Size = (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Size = (nc) x 64 x 64
        )
    

    def forward(self, input):
        return self.forward_call(input)
    
    
class DCGAN_discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_discriminator, self).__init__()
        self.ndf = 64
        self.nc = 3
        self.forward_call = nn.Sequential(
            # Input size is nc x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.forward_call(input)


def split_loader(train_loader, standard_transform):
    """This function takes the standard NTZfilterdataset loader
    and splits it up into 4 new loaders, one for each class.
    This is necessary to allow the DCGAN to generate images
    per class, not for the entire dataset. 

    Args:
        train_loader: The standard NTZfilterdataset loader.
        standard_transform: no_augment transform.
    
    Returns:
        List of 4 dataloaders.
    """
    complete_dataset = train_loader.dataset
    dataset_paths = complete_dataset.img_paths
    datasets = []
    data_loaders = []
    NTZ_path = os.path.join("data", "NTZFilter", "train")

    # Creating 4 datasets based on standard NTZFilter setup
    # But removing image paths and image labels
    for idx in range(len(complete_dataset.label_map)):
        dataset = NTZFilterDataset(NTZ_path, standard_transform)
        dataset.img_paths = []
        dataset.img_labels = []
        datasets.append(dataset)
    
    # Adding labels/img_paths to the 4 datasets
    for idx, img_path in enumerate(dataset_paths):
        label = complete_dataset.img_labels[idx]
        datasets[label].img_paths.append(img_path)
        datasets[label].img_labels.append(label)
        
    # Creating dataloaders for each class
    for dataset in datasets:
        data_loaders.append(DataLoader(dataset, batch_size = 
                                       train_loader.batch_size,
                                       collate_fn = train_loader.collate_fn,
                                       shuffle = True,
                                       num_workers = train_loader.num_workers))
    return data_loaders


def generate_images(gen_model: DCGAN_generator, n_imgs: int, nz: int,
                    device: torch.device, path: str):
    """This function takes a generative model, provides noise
    data for it, which it then uses to generate n_imgs amount
    of images that are similar to the dataset trained on.

    Args:
        gen_model: The generative model to use.
        n_imgs: The amount of images to generate.
        nz: The size of the latent vector.
        device: The device to use.
        path: The path to save the images to.
    """
    # Converting to evaluation, since no training should occur
    gen_model.eval()

    # Generate random noise
    noise = torch.randn(n_imgs, nz, 1, 1, device = device)

    # Generate fake images
    fake_images = gen_model(noise).cpu()

    # Check if the directory exits, remove old images if it does.
    remove_predicts(path)

    # Converting to PIL and saving image
    to_PIL = transforms.ToPILImage()
    for idx, img in enumerate(fake_images):
        pil_image = to_PIL(img)
        
        # Save PIL image to disk
        pil_image.save(os.path.join(path, ("gen_img_" + str(idx) + ".png")))


def weights_init(model: nn.Module):
    """Function to initialize the weights of the model.
    The weights are initalized in place, so there is no
    need to return the model. The mean is set to 0
    and the standard deviation is set to 0.02.
    
    Args:
        m: The model to initialize.
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def train_DCGAN(gen_model: DCGAN_generator, disc_model: DCGAN_discriminator,
                train_loader: DataLoader, criterion: nn.BCELoss,
                optimizer_gen: optim.Adam, optimizer_disc: optim.Adam,
                device: torch.device, nz: int):
    """This function takes a generator and a discriminator of a
    GAN setup and trains them both to produce generative data.

    Args:
        gen_model: The generator model.
        disc_model: The discriminator model.
        train_loader: The dataloader for the training data.
        criterion: The loss function to use.
        optimizer_gen: The optimizer for the generator.
        optimizer_disc: The optimizer for the discriminator.
        device: The device to use.
        nz: The size of the latent vector.
    """
    real_label = 1
    fake_label = 0
    epochs = 300

    # Setting models to training mode
    gen_model.train()
    disc_model.train()

    # Moving models to device
    gen_model.to(device)
    disc_model.to(device)

    # Main training loop
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        disc_loss = 0
        for input, _ in tqdm(train_loader):
            # First part consists of updating the discriminator
            # Removing previous gradients
            gen_model.zero_grad()

            # Training with real data batch
            batch_size = input.size(0)
            labels = torch.full((batch_size,), real_label, dtype = torch.float,
                                device = device)
            
            # Forwards pass, loss calculation and backwards pass
            # For discriminator on real batch
            output = disc_model(input.to(device)).view(-1)
            disc_loss_real = criterion(output, labels)
            disc_loss_real.backward()

            # Training with fake data batch
            noise = torch.randn(batch_size, nz, 1, 1, device = device)
            fake_data = gen_model(noise)
            labels.fill_(fake_label)
            
            # Forwards pass, loss calculation and backwards pass
            # For discriminator on fake batch
            output = disc_model(fake_data.detach()).view(-1)
            disc_loss_fake = criterion(output, labels)
            disc_loss_fake.backward()

            # Optimizer step after discriminator real and fake updates
            optimizer_disc.step()

            # Second part consists of updating the generator
            # Removing previous gradients
            gen_model.zero_grad()
            labels.fill_(real_label)
            # Get outputs and calculate loss
            output = disc_model(fake_data).view(-1)
            gen_loss = criterion(output, labels)
            # Backwards pass and optimizer step
            gen_loss.backward()
            optimizer_gen.step()
            
            disc_loss += disc_loss_real.item() + disc_loss_fake.item()
        # Printing average loss over epoch to console, should converge to 0.5
        print("Discriminator loss: " + str(round(disc_loss/len(train_loader), 4)))

    return gen_model


def setup_DCGAN(train_type: str, n_imgs: int, latent_vector_size: int):
    """Function that sets up a DCGAN model. The function prepares
    a training phase and afterwards takes n images generated by the
    model as synthetic data. The only data this is useful for is the 
    NTZ dataset. DCGAN definitions based on Pytorch tutorial:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    and DCGAN paper:
    https://arxiv.org/abs/1511.06434
    NOTE: If using DCGAN change T.CenterCrop(224) to T.CenterCrop(64),
    in utils.py, get_default_transforms function. Since DCGAN is not used
    further, this was not adapted.

    Args:
        train_type: Train GAN on class data combined or seperate.
        n_imgs: The amount of images to generate.
        latent_vector_size: The size of the latent vector, synonym to nz.
    """    
    # Setting device to use
    device = get_device()

    # Getting NTZ filter data 
    standard_transform = get_transforms(transform_type = "no_augment")
    complete_train_loader = get_data_loaders(transform = standard_transform)["train"]

    # Splitting up into 4 classes if desired training type
    if train_type == "seperate":
        train_loaders = split_loader(complete_train_loader, standard_transform)
        class_names = complete_train_loader.dataset.label_map
    else:
        train_loaders = [complete_train_loader]
        class_names = {0: "combined"}

    for idx, train_loader in enumerate(train_loaders):
        # Setup generator and discriminator
        gen_model = DCGAN_generator()
        disc_model = DCGAN_discriminator()

        # Setting their weights to mean 0 and std dev = 0.02
        gen_model.apply(weights_init)
        disc_model.apply(weights_init)

        # Setting criterion
        criterion = nn.BCELoss()

        # Setup Adam optimizers for both G and D
        optimizer_gen = optim.Adam(gen_model.parameters(), lr = 0.0002,
                                betas = (0.5, 0.999))
        optimizer_disc = optim.Adam(disc_model.parameters(), lr = 0.0002,
                                    betas = (0.5, 0.999))

        gen_model = train_DCGAN(gen_model, disc_model, train_loader, criterion, optimizer_gen,
                                optimizer_disc, device, latent_vector_size)
        
        class_name = class_names[idx]
        path = os.path.join("data", "NTZFilterGenerative", class_name)
        generate_images(gen_model, n_imgs, latent_vector_size, device, path)

if __name__ == '__main__':
    # Forming argparser with optional arguments for training type,
    # amount of images and latent vector size
    parser = argparse.ArgumentParser()
    parser.add_argument("train_type", type = str, default = "seperate",
                        choices = ["seperate", "combined"])
    parser.add_argument("--n_imgs", type = int, default = 50)
    parser.add_argument("--latent_vector_size", type = int, default = 100)
    args = parser.parse_args()
    setup_DCGAN(args.train_type, args.n_imgs, args.latent_vector_size)