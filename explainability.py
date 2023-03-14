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
from tqdm import tqdm
from train_utils import convert_to_list, flatten_list

from matplotlib.colors import LinearSegmentedColormap
from NTZ_filter_dataset import NTZFilterDataset

from test import sep_test_collate

def integrated_gradients(model: torchvision.models):
    # SHAP/Captum
    # SHAP: GradientExplainer/DeepExplainer
    # CAPTUM: IntegratedGradients/DeepLift
    # ONLY FOR INFERENCE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])
    denormalize = T.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225])

    transform_without_norm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    test_path = os.path.join("data", "test")
    batch_size = 16
    # Creating the dataset and transferring to a DataLoader
    test_data = NTZFilterDataset(test_path, transform)
    test_loader = DataLoader(test_data, batch_size = batch_size,
                             collate_fn = sep_test_collate, shuffle = True, num_workers = 4)

    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

    img_paths = []
    input_concat = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, paths in tqdm(test_loader):
            inputs = inputs.to(device)

            # Getting model output and adding labels/paths to lists
            model_output = model(inputs)
            predicted_labels.append(model_output.argmax(dim=1))
            img_paths.append(paths)

            input_concat.append(inputs)
    
    prediction_list = convert_to_list(predicted_labels)
    img_paths = flatten_list(img_paths)
    input_concat_tensor = input_concat[0]
    for item in input_concat[1:]:
        input_concat_tensor = torch.cat((input_concat_tensor, item))
    
    # Creating integrated gradients object via Captum
    int_grad = captum.attr.IntegratedGradients(model)
    attributions_ig = int_grad.attribute(input_concat_tensor, target = prediction_list, internal_batch_size = 16, n_steps = 200)
    # Noise tunnel causes CUDA out of memory errors
    # noise_tunnel = captum.attr.NoiseTunnel(int_grad)
    # attributions_ig_nt = noise_tunnel.attribute(inputs, nt_samples=10, nt_samples_batch_size = 16, nt_type='smoothgrad_sq', target = predicted_labels)

    for i, img in enumerate(attributions_ig):
        norm_img = Image.open(img_paths[i])
        norm_img = transform_without_norm(norm_img)
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


if __name__ == '__main__':
    experiment = "Experiment3-RandAugment06-03-2023-16-04"
    path = os.path.join("Master-Thesis-Experiments", experiment, "model.pth")
    model = torch.load(path)
    integrated_gradients(model)
    #guided_backpropagation()
    #deep_uncertainty_quantification()
    #deep_ensemble_uncertainty()