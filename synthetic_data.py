from PIL import Image, ImageFilter
import os
from skimage.segmentation import slic
from skimage.color import label2rgb
import numpy as np
import cv2

def get_edges(im):
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
    im_sobel = im.filter(ImageFilter.Kernel((3, 3), laplacian_kernel,  1, 0))
    im_laplacian = im.filter(ImageFilter.Kernel((3, 3), laplacian_kernel,  1, 0))
 
    # Saving the edges image
    im_sobel.save("example_img_sobel.jpg")
    im_laplacian.save("example_img_laplacian.jpg")


def segment_slic(im):
    im = np.array(im)
    im_segments = slic(im, n_segments = 7, compactness = 10)
    
    # Saving the image with all clusters overlayed
    save = label2rgb(im_segments, im, kind = 'overlay')
    im = Image.fromarray(np.uint8(save*255))
    im.save("example_segmented.jpg")


def template_matching(path):
    im = cv2.imread(os.path.join(path, "20230130_131235(002)_No.70.bmp"))
    template = cv2.imread("template.png")
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(im, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(im, top_left, bottom_right, 255, 2)
    cv2.imwrite("template_matched.jpg", im)


def create_synthetic_data():
    # First practicing with a single image
    path = os.path.join("data", "train", "fail_label_not_fully_printed")
    im = Image.open(os.path.join(path, "20230130_131235(002)_No.70.bmp"))
    
    #get_edges(im)
    #segment_slic(im)
    template_matching(path)


if __name__ == '__main__':
    create_synthetic_data()