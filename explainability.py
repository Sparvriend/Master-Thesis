import captum
from captum.attr import visualization as viz
import torch
import torchvision.models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap

from train_utils import get_default_transform

def integrated_gradients(model: torchvision.models, data_loader: DataLoader,
                         predicted_labels: list, img_paths:list):
    # SHAP/Captum
    # SHAP: GradientExplainer/DeepExplainer
    # CAPTUM: IntegratedGradients/DeepLift
    # ONLY FOR INFERENCE

    # Getting default transform and cutting off normalization
    transform = T.Compose(get_default_transform().transforms[:-1])

    input_concat = []
    for inputs, _ in data_loader:
        input_concat.append(inputs)
    
    input_concat_tensor = input_concat[0]
    for item in input_concat[1:]:
        input_concat_tensor = torch.cat((input_concat_tensor, item))
    
    # Creating integrated gradients object via Captum
    int_grad = captum.attr.IntegratedGradients(model)
    attributions_ig = int_grad.attribute(input_concat_tensor.cuda(), target = predicted_labels, internal_batch_size = 16, n_steps = 200)
    # Noise tunnel causes CUDA out of memory errors
    # noise_tunnel = captum.attr.NoiseTunnel(int_grad)
    # attributions_ig_nt = noise_tunnel.attribute(inputs, nt_samples=10, nt_samples_batch_size = 16, nt_type='smoothgrad_sq', target = predicted_labels)

    default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'),
                                                     (0.25, '#000000'), (1, '#000000')],
                                                     N=256)

    for i, img in enumerate(attributions_ig):
        norm_img = Image.open(img_paths[i])
        norm_img = transform(norm_img)
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(np.asarray(norm_img), (1,2,0)),
                             methods=["original_image", "heat_map"],
                             signs=["all", "positive"],
                             cmap=default_cmap,
                             show_colorbar=True)
        fig.savefig(os.path.join("temp", str(i) + ".png"))
        plt.close()


def guided_backpropagation():
    # SHAP/Captum
    # SHAP: KernelExplainer
    # Captum: GuidedBackprop
    # ONLY FOR INFERENCE
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