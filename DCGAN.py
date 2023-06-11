import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from utils import get_data_loaders, get_transforms, get_device
import torchvision.transforms as transforms
import os

class DCGAN_generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(DCGAN_generator, self).__init__()
        self.forward_call = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Size = (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size = (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Size = (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size = (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Size = (nc) x 64 x 64
        )
    

    def forward(self, input):
        return self.forward_call(input)
    
    
class DCGAN_discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(DCGAN_discriminator, self).__init__()
        self.forward_call = nn.Sequential(
            # Input size is nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size = (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.forward_call(input)
    

def generate_images(gen_model, n_imgs, nz, device, path):
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

    # Converting to PIL and saving image
    to_PIL = transforms.ToPILImage
    for idx, img in enumerate(fake_images):
        pil_image = to_PIL(img)
        
        # Save PIL image to disk
        pil_image.save(os.path.join(path, ("gen_img_" + str(idx) + ".png")))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_DCGAN(gen_model, disc_model, train_loader, criterion, optimizer_gen,
                optimizer_disc, device, nz):
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
    epochs = 5

    # Setting models to training mode
    gen_model.train()
    disc_model.train()

    # Main training loop
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        for input, _ in tqdm(train_loader):
            # First part consists of updating the discriminator
            # Removing previous gradients
            gen_model.zero_grad()

            # Training with real data batch
            batch_size = input.size(0)
            labels = torch.full((batch_size,), real_label, dtype = torch.float,
                                device = device)
            real_labels = labels.fill_(real_label)
            
            # Forwards pass, loss calculation and backwards pass
            # For discriminator on real batch
            output = disc_model(input.to(device)).view(-1)
            disc_loss_real = criterion(output, real_labels)
            disc_loss_real.backward()

            # Training with fake data batch
            noise = torch.randn(batch_size, nz, 1, 1, device = device)
            fake_data = gen_model(noise)
            fake_labels = labels.fill_(fake_label)
            
            # Forwards pass, loss calculation and backwards pass
            # For discriminator on fake batch
            output = disc_model(fake_data.detach()).view(-1)
            disc_loss_fake = criterion(output, fake_labels)
            disc_loss_fake.backward()

            # Optimizer step after discriminator real and fake updates
            optimizer_disc.step()

            # TODO: Does using real_labels here causes issues?
            # e.g. is it overwritten by the fake labels?

            # Second part consists of updating the generator
            # Removing previous gradients
            gen_model.zero_grad()
            # Get outputs and calculate loss
            output = disc_model(fake_data).view(-1)
            gen_loss = criterion(output, real_labels)
            # Backwards pass and optimizer step
            gen_loss.backward()
            optimizer_gen.step()

    return gen_model


def setup_DCGAN():
    """Function that sets up a DCGAN model. The function prepares
    a training phase and afterwards takes n images generated by the
    model as synthetic data. The only data this is useful for is the 
    NTZ dataset. DCGAN definitions based on Pytorch tutorial:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    and DCGAN paper:
    https://arxiv.org/abs/1511.06434
    """    
    # Setting device to use
    device = get_device()

    # Amount of images to generate with the trained generator
    n_imgs = 10

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of feature maps in discriminator
    ndf = 64

    # Getting NTZ filter data
    train_loader = get_data_loaders(transform = get_transforms(transform_type = "no_augment"))["train"]

    # TODO: Adapt the train loader, such that it seperates into 4, one loader for each class
    # these loaders should then be used to generate from each image class seperately

    # Setup generator and discriminator
    gen_model = DCGAN_generator(nz, ngf, nc)
    disc_model = DCGAN_discriminator(nc, ndf)

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
                            optimizer_disc, device, nz)
    
    path = os.path.join("data", "NTZFilterGenerative")
    generate_images(gen_model, n_imgs, nz, device, path)

if __name__ == '__main__':
    setup_DCGAN()
