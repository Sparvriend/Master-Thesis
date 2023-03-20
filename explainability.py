import captum
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torchvision.models
import torchvision.transforms as T


def visualize_explainability(img_data: torch.Tensor, img_paths: list):
    """Function that visualizes an explainability result.
    It combines the original image with image data that explains
    the decision making of the model.

    Args:
        img_data: torch tensor with image data.
        img_paths: list of paths to the original images.
    """
    # Custom transform for only resizing and then cropping to center
    transform = T.Compose([T.Resize(256, max_size = 320), T.CenterCrop(224)])

    # Creating custom colormap
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')],
                                                     N = 256)
    
    # Iterating over all image data and saving 
    # them in combination with the original image
    for i, img in enumerate(img_data):
        norm_img = Image.open(img_paths[i])
        norm_img = transform(norm_img)
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                   np.asarray(norm_img),
                                                   methods = ["original_image", "heat_map"],
                                                   signs = ["all", "positive"],
                                                   cmap = default_cmap,
                                                   show_colorbar = True)
        img_name = os.path.normpath(img_paths[i]).split(os.sep)[-1]
        fig.savefig(os.path.join("Explainability-Results", img_name.replace(".bmp", ".png")))
        plt.close()


def integrated_gradients(model: torchvision.models, predicted_labels: list,
                         img_paths: list, input_concat: torch.Tensor, 
                         device: torch.device, noise_tunnel: bool = False):
    """Function that runs integrated gradients on the model.

    Args:
        model: model to run integrated gradients on.
        predicted_labels: list of predicted labels.
        img_paths: list of paths to the original images.
        input_concat: concatenated input tensor.
                      The input is shuffled, hence this being necessary.
        device: device to run the model on.
        noise_tunnel: whether to use noise tunnel or not.
    """
    print("Running integrated gradients for model explanation")
    
    # Creating integrated gradients object via Captum
    data_len = len(predicted_labels)
    int_grad = captum.attr.IntegratedGradients(model)
    gradient_attr = int_grad.attribute(input_concat.to(device),
                                         target = predicted_labels,
                                         internal_batch_size = data_len,
                                         n_steps = 200)
    if noise_tunnel:
        print("Refining integrated gradients with noise tunnel")
        # Noise tunnel causes CUDA out of memory errors
        # When running it with more than a few inputs
        # Hence, running it in pairs of 2
        gradient_attr = []
        noise_tunnel = captum.attr.NoiseTunnel(int_grad)
        i = 2
        while data_len > i:
            ig_nt_element = noise_tunnel.attribute(input_concat[i-2:i].to(device),
                                                   target = predicted_labels[i-2:i],
                                                   nt_samples_batch_size = 1,
                                                   nt_samples = 10, 
                                                   nt_type='smoothgrad_sq')
            gradient_attr.extend(ig_nt_element)
            i += 2
        if data_len % 2 != 0:
            ig_nt_element = noise_tunnel.attribute(input_concat[data_len-2:].to(device),
                                                   target = predicted_labels[data_len-2:],
                                                   nt_samples_batch_size = 1,
                                                   nt_samples = 10, 
                                                   nt_type='smoothgrad_sq')
            gradient_attr.append(ig_nt_element[1])
    visualize_explainability(gradient_attr, img_paths)


def deeplift():
    # Use Captum
    print("Not yet implemented")


def guided_backpropagation():
    # Use Captum's GuidedBackprop
    print("Not yet implemented")


def saliency_map():
    # Use Captum's Saliency
    # Also possible to implement by hand
    print("Not yet implemented")


def deep_uncertainty_quantification():
    # Average centroids need to be calculated during training
    # The function here can then use these centroids to calculate
    # Uncertainty by distance.
    print("Not yet implemented")


def deep_ensemble_uncertainty():
   # Hardest to implement? -> No clear library available
   # During training time/testing time.
    print("Not yet implemented")