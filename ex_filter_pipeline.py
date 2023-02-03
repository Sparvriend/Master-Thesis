from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import copy

DARKNESS_PARAM = 0.4

def create_example_img(name, save_loc):
    # Loading the image
    original_filter_img = Image.open(name + ".jpg")

    # Setup image enhancer with PIL
    enhancer = ImageEnhance.Brightness(original_filter_img)

    # Darken the image with a factor of 0.22
    original_filter_img = enhancer.enhance(DARKNESS_PARAM)
    #original_filter_img.save(str(name) + "darkened.jpg")

    # Save the image widht and height
    width = original_filter_img.size[0]
    length = original_filter_img.size[1]

    # Get a pixel accessing object
    px_access = original_filter_img.load()

    # Set each pixel to a dead pixel based on a 5% chance
    for pixel_w in range(width):
        for pixel_l in range(length):
            if np.random.random(1)[0] > 1:
                px_access[pixel_w, pixel_l] = (0, 0, 0)

    # Saving the darkened image with noise
    original_filter_img.save(str(save_loc) + "_darkened_noise.jpg")

def edge_detection(name, save_loc):
    # Loading image and converting to grayscale for the edge detection
    original_filter_img = Image.open(name + ".jpg")
    grayscale_original = original_filter_img.convert("L")
 
    # Defining the Sobel & Laplacian kernels
    sobel_kernel =     (1, 0, -1,
                        2, 0, -2,
                        1, 0, -1)
    laplacian_kernel = (1, 1, 1,
                        1, -8, 1,
                        1, 1, 1)

    # Calculating edges using a filter
    edges_original = grayscale_original.filter(ImageFilter.Kernel((3, 3), laplacian_kernel,  1, 0))
 
    # Saving the edges image
    edges_original.save(str(save_loc) + "_edges.jpg")

def img_segmentation(name, save_loc):
    # Loading the image and normalizing all values
    original_filter_img = Image.open(name + ".jpg")
    original_np = np.array(original_filter_img)

    clusters = 20
    # Segmenting the image using Simple Linear Iterative Clustering (SLIC)
    original_segmented = slic(original_np, n_segments=clusters, compactness=10)

    # Saving the image with all clusters overlayed
    sav = label2rgb(original_segmented, original_np, kind = 'overlay')
    im = Image.fromarray(np.uint8(sav*255))
    im.save(str(save_loc) + "_segmented.jpg")

    # Creating images for each cluster segment, with all other pixels black
    original_np_copy = copy.deepcopy(original_np)

    for cluster in range(1, clusters):
        original_np = copy.deepcopy(original_np_copy)
        for i, value in enumerate(original_segmented):
            for j, val in enumerate(value):
                if val != cluster:
                    original_np[i][j] *= 0

        im = Image.fromarray(np.uint8(original_np))
        im.save(str(save_loc) + "_segmented" + str(cluster) + ".jpg")

def histogram_equalization(name, save_loc):
    # Loading the image
    darkened_img = Image.open(save_loc + "_darkened_noise.jpg")
    darkened_img_grayscale = darkened_img.convert("L")
    
    # Applying equalize method 
    equalized_img = ImageOps.equalize(darkened_img, mask = None)
    equalized_img_grayscale = ImageOps.equalize(darkened_img_grayscale, mask = None)

    # Saving equalized image
    equalized_img.save(str(save_loc) + "_equalized.jpg")
    equalized_img_grayscale.save(str(save_loc) + "_equalized_grayscale.jpg")

def remove_noise(name, save_loc):
    # Loading the image
    img = Image.open(save_loc + "_equalized.jpg")

    # Denoising the image
    denoised_img = img.filter(ImageFilter.GaussianBlur)

    # Saving denoised image
    denoised_img.save(str(save_loc) + "_equalized_denoised.jpg")

def augment_image(name, save_loc):
    # Loading the image
    img = Image.open(save_loc + "_equalized.jpg")

    # Applying basic augmentation techniques
    for technique, aug_name in [[ImageOps.invert, "_inverted"], [ImageOps.flip, "_flipped"], [ImageOps.mirror, "_mirrored"]]:
        augmented = technique(img)
        augmented.save(str(save_loc) + aug_name + "equalized_denoised.jpg")

if __name__ == '__main__':
    name = "preliminary_pipeline/industrial_label_example"
    save_loc = "example_pipeline_imgs/industrial_label_example"
    create_example_img(name, save_loc)
    # edge_detection(name, save_loc)
    # img_segmentation(name, save_loc)
    histogram_equalization(name, save_loc)
    remove_noise(name, save_loc)
    augment_image(name, save_loc)
    
    

