import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from tqdm import tqdm

from datasets import NTZFilterDataset
from torch.utils.data import DataLoader
from utils import get_data_loaders, get_transforms, get_device, remove_predicts


class LSGAN_generator(nn.Module):
    """LSGAN definitions taken from 
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/lsgan/lsgan.py
    """
    def __init__(self, img_size, channels, latent_dim):
        super(LSGAN_generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim

        self.init_size = self.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class LSGAN_discriminator(nn.Module):
    """LSGAN definitions taken from 
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/lsgan/lsgan.py
    """
    def __init__(self, img_size, channels):
        super(LSGAN_discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def split_loader(train_loader: DataLoader, standard_transform: T.Compose,
                 dataset: str) -> list:
    """This function takes the standard loader and splits
    it up into n new loaders, one for each class in the dataset.
    This is necessary to allow the GAN to generate images
    per class, not for the entire dataset. 

    Args:
        train_loader: The standard loader.
        standard_transform: no_augment transform.
    
    Returns:
        List of dataloaders per class.
    """
    complete_dataset = train_loader.dataset
    dataset_paths = complete_dataset.img_paths
    datasets = []
    data_loaders = []
    path = os.path.join("data", dataset.removesuffix("Dataset"), "train")

    # Creating 4 datasets based on standard NTZFilter setup
    # But removing image paths and image labels
    for idx in range(len(complete_dataset.label_map)):
        sep_dataset = eval(dataset)(path, standard_transform)
        sep_dataset.img_paths = []
        sep_dataset.img_labels = []
        datasets.append(sep_dataset)
    
    # Adding labels/img_paths to the datasets
    for idx, img_path in enumerate(dataset_paths):
        label = complete_dataset.img_labels[idx]
        datasets[label].img_paths.append(img_path)
        datasets[label].img_labels.append(label)
        
    # Creating dataloaders for each class
    for sep_dataset in datasets:
        data_loaders.append(DataLoader(sep_dataset, batch_size = 
                                       train_loader.batch_size,
                                       collate_fn = train_loader.collate_fn,
                                       shuffle = True,
                                       num_workers = train_loader.num_workers))
    return data_loaders


def generate_images(gen_model: LSGAN_generator, n_imgs: int, nz: int,
                    device: torch.device, path: str, normalize: T.Normalize):
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
    noise = torch.randn(n_imgs, nz, device = device)

    # Generate fake images
    fake_images = gen_model(noise).cpu()

    # Check if the directory exits, remove old images if it does.
    remove_predicts(path)

    # Make an inverse normalization transform
    mean = normalize.mean
    std = normalize.std
    inverse_mean = [-m/s for m, s in zip(mean, std)]
    inverse_std = [1/s for s in std]
    inverse_normalize = T.Normalize(mean = inverse_mean, std = inverse_std)

    # Converting to PIL and saving image
    post_process = T.Compose([inverse_normalize, T.ToPILImage()])
    for idx, img in enumerate(fake_images):
        pil_image = post_process(img)
        
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


def train_GAN(gen_model: LSGAN_generator, disc_model: LSGAN_discriminator,
              train_loader: DataLoader, criterion: nn.BCELoss,
              optimizer_gen: optim.Adam, optimizer_disc: optim.Adam,
              device: torch.device, nz: int, epochs: int) -> LSGAN_generator:
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
        epochs: Amount of epochs to train for.
    Returns:
        The trained generator model.
    """
    real_label = 1
    fake_label = 0

    # Setting models to training mode
    gen_model.train()
    disc_model.train()

    # Moving models to device
    gen_model.to(device)
    disc_model.to(device)

    # Main training loop
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        disc_loss_epoch = 0
        gen_loss_epoch = 0
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
            noise = torch.randn(batch_size, nz, device = device)
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
            
            disc_loss_epoch += disc_loss_real.item() + disc_loss_fake.item()
            gen_loss_epoch += gen_loss.item()

        # Printing average discriminator and generator loss over epoch
        # Discriminator loss should converge to 0.5
        # Generator loss should reduce during training
        print("Discriminator loss: " + str(round(disc_loss_epoch/len(train_loader), 4))
              + "\tGenerator loss: " + str(round(gen_loss_epoch/len(train_loader), 4)))
    return gen_model


def setup_GAN(train_type: str, n_imgs: int, latent_vector_size: int, epochs: int):
    """Function that sets up a GAN model. The function prepares
    a training phase and afterwards takes n images generated by the
    model as synthetic data. The only data this is useful for is the 
    NTZ dataset.
    
    Args:
        train_type: Train GAN on class data combined or seperate.
        n_imgs: The amount of images to generate.
        latent_vector_size: The size of the latent vector, synonym to nz.
        dataset: The dataset to use.
        epochs: The amount of epochs to train for.
    """    
    # Setting device & dataset
    device = get_device()
    dataset = "NTZFilterDataset"

    # Getting data
    standard_transform = get_transforms(transform_type = "no_augment")
    complete_train_loader = get_data_loaders(transform = standard_transform,
                                             dataset = eval(dataset))["train"]
    normalize = standard_transform.transforms[-1]

    # Depending on the dataset, the amount of channels and image size should be set
    batch = next(iter(complete_train_loader))
    img_size = batch[0][0].size(1)
    channels = batch[0][0].size(0)
    latent_dim = 100
    dataset_name = dataset.removesuffix("Dataset") + "Generative"

    # Create directory if it does not exist
    if not os.path.exists(os.path.join("data", dataset_name)):
        os.mkdir(os.path.join("data", dataset_name))
        class_names = ["fail_label_not_fully_printed",
                       "fail_label_half_printed",
                       "fail_label_crooked_print",
                       "no_fail"]
        for class_name in class_names:
            os.mkdir(os.path.join("data", dataset_name, class_name))

    # Splitting up into classes if desired training type
    if train_type == "seperate":
        train_loaders = split_loader(complete_train_loader, standard_transform,
                                     dataset)
        class_names = complete_train_loader.dataset.label_map
    else:
        train_loaders = [complete_train_loader]
        class_names = {0: "combined"}

    for idx, train_loader in enumerate(train_loaders):
        # Setup generator and discriminator
        gen_model = LSGAN_generator(img_size, channels, latent_dim)
        disc_model = LSGAN_discriminator(img_size, channels)

        # Setting their weights to mean 0 and std dev = 0.02
        gen_model.apply(weights_init)
        disc_model.apply(weights_init)

        # Setting criterion
        criterion = nn.MSELoss()

        # Setup Adam optimizers for both G and D
        optimizer_gen = optim.Adam(gen_model.parameters(), lr = 0.0002,
                                betas = (0.5, 0.999))
        optimizer_disc = optim.Adam(disc_model.parameters(), lr = 0.0002,
                                    betas = (0.5, 0.999))

        gen_model = train_GAN(gen_model, disc_model, train_loader, criterion, optimizer_gen,
                                optimizer_disc, device, latent_vector_size, epochs)
        
        class_name = class_names[idx]
        path = os.path.join("data", dataset_name, class_name)
        generate_images(gen_model, n_imgs, latent_vector_size, device, path, normalize)


if __name__ == '__main__':
    # Forming argparser with optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_type", type = str,
                        choices = ["seperate", "combined"])
    parser.add_argument("--n_imgs", type = int, default = 50)
    parser.add_argument("--latent_vector_size", type = int, default = 100)
    parser.add_argument("--epochs", type = int, default = 100)
    args = parser.parse_args()
    setup_GAN(args.train_type, args.n_imgs, args.latent_vector_size, args.epochs)