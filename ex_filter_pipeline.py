from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import copy

def create_example_img(name):
    # Loading the image
    original_filter_img = Image.open(name + ".jpg")

    # Setup image enhancer with PIL
    enhancer = ImageEnhance.Brightness(original_filter_img)

    # Darken the image with a factor of 0.22
    original_filter_img = enhancer.enhance(0.2)
    #original_filter_img.save(str(name) + "darkened.jpg")

    # Save the image widht and height
    width = original_filter_img.size[0]
    length = original_filter_img.size[1]

    # Get a pixel accessing object
    px_access = original_filter_img.load()

    # Set each pixel to a dead pixel based on a 15% chance
    for pixel_w in range(width):
        for pixel_l in range(length):
            if np.random.random(1)[0] > 0.85:
                px_access[pixel_w, pixel_l] = (0, 0, 0)

    # Saving the darkened image with noise
    original_filter_img.save(str(name) + "_darkened_noise.jpg")

def edge_detection(name):
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
    edges_original.save(str(name) + "_edges.jpg")

def img_segmentation(name):
    # Loading the image and normalizing all values
    original_filter_img = Image.open(name + ".jpg")
    original_np = np.array(original_filter_img)

    clusters = 20
    # Segmenting the image using Simple Linear Iterative Clustering (SLIC)
    original_segmented = slic(original_np, n_segments=clusters, compactness=10)

    # Saving the image with all clusters overlayed
    sav = label2rgb(original_segmented, original_np, kind = 'overlay')
    im = Image.fromarray(np.uint8(sav*255))
    im.save(str(name) + "_segmented.jpg")

    # Creating images for each cluster segment, with all other pixels black
    original_np_copy = copy.deepcopy(original_np)

    for cluster in range(1, clusters):
        original_np = copy.deepcopy(original_np_copy)
        for i, value in enumerate(original_segmented):
            for j, val in enumerate(value):
                if val != cluster:
                    original_np[i][j] *= 0

        im = Image.fromarray(np.uint8(original_np))
        im.save(str(name) + "_segmented" + str(cluster) + ".jpg")

if __name__ == '__main__':
    name = "example_pipeline_imgs/industrial_label_example" 
    create_example_img(name)
    edge_detection(name)
    img_segmentation(name)

