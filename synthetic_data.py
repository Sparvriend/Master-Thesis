from PIL import Image, ImageFilter
import os
from skimage.segmentation import slic
from skimage.color import label2rgb
import numpy as np
import cv2

def get_edges():
    # Retrieving example image:
    im = get_example_img()

    # Converting to grayscale
    im = im.convert("L")

    # Defining the Sobel & Laplacian kernels
    sobel_kernel =     (1, 0, -1,
                        2, 0, -2,
                        1, 0, -1)
    laplacian_kernel = (1, 1, 1,
                        1, -8, 1,
                        1, 1, 1)
    
    # Calculating edges using a filter
    im_sobel = im.filter(ImageFilter.Kernel((3, 3), sobel_kernel,  1, 0))
    im_laplacian = im.filter(ImageFilter.Kernel((3, 3), laplacian_kernel,  1, 0))
 
    # Saving the edges image
    im_sobel.save("example_img_sobel.jpg")
    im_laplacian.save("example_img_laplacian.jpg")


def segment_slic():
    im = get_example_img()
    im = np.array(im)
    im_segments = slic(im, n_segments = 6, compactness = 10)
    
    # Saving the image with all clusters overlayed
    save = label2rgb(im_segments, im, kind = "overlay")
    im = Image.fromarray(np.uint8(save*255))
    im.save("example_slic_segmented.jpg")

def get_template_match(image, template):
    # TODO: Fix angled rectangle mapping
    # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    # Try different angles and pick best one based on some criteria

    # Convert both to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Compute the mean and standard deviation of the template
    mean, std_dev = cv2.meanStdDev(gray_template)

    # Compute the size of the template
    h, w = gray_template.shape[:2]

    # Compute the normalized cross-correlation
    result = np.zeros((image.shape[0] - h + 1, image.shape[1] - w + 1),
                       dtype = np.float32)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            patch = gray_image[y:y + h, x:x + w]
            patch_mean, patch_std_dev = cv2.meanStdDev(patch)
            numerator = np.sum((patch - patch_mean) * (gray_template - mean))
            denominator = patch_std_dev * std_dev * h * w
            result[y, x] = numerator / denominator

    # Find the location with the highest cross-correlation
    y, x = np.unravel_index(np.argmax(result), result.shape)
    top_left = (x, y)
    bottom_right = (x + w, y + h)

    return top_left, bottom_right

def template_matching():
    class_names = ["fail_label_not_fully_printed", "fail_label_half_printed",
                   "fail_label_crooked_print", "no_fail"]
    data_types = ["train", "val", "test"]
    template = cv2.imread("template.png")    

    for data_type in data_types:
        for class_name in class_names:
            path = os.path.join("data", "NTZFilterDataset", data_type, class_name)
            files = os.listdir(path)
            for file in files:
                img_path = os.path.join(path, file)
                im = cv2.imread(img_path)
                top_left, bottom_right = get_template_match(im, template)
                # Draw a rectangle around the matched area
                # and saving the image
                cv2.rectangle(im, top_left, bottom_right, (0, 0, 255), 2)
                img_destination = os.path.join("data", "NTZFilterSynthetic",
                                                data_type, class_name, file)
                cv2.imwrite(img_destination, im)


def create_synthetic_data_dirs():
    # Creating the directories for the synthetic data
    class_names = ["fail_label_not_fully_printed", "fail_label_half_printed",
                   "fail_label_crooked_print", "no_fail"]
    data_types = ["train", "val", "test"]
    for data_type in data_types:
        for class_name in class_names:
            path = os.path.join("data", "NTZFilterSynthetic", data_type, class_name)
            os.makedirs(path)


def get_example_img():
    # For synthetic data generation practice with a single example image
    path = os.path.join("data", "NTZFilterDataset", "train", 
                        "fail_label_not_fully_printed",
                        "20230130_131235(025)_No.93.bmp")
    return Image.open(path)


def active_contours():
    # Uses active contours models
    # https://scikit-image.org/docs/stable/auto_examples/edges/plot_active_contours.html
    print("Not implemented yet.")


def create_synthetic_data():
    create_synthetic_data_dirs()
    #get_edges(im)
    #segment_slic(im)
    template_matching()
    

if __name__ == '__main__':
    create_synthetic_data()