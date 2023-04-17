from PIL import Image, ImageStat
import os
import numpy as np
import cv2


def create_synthetic_data_dirs(class_names: list):
    """Function that creates all directories required for synthetic
    data generation and saving progress of intermediate steps.
    
    Args:
        class_names: List of class names
    """
    # Creating top level directory
    path = os.path.join("data", "NTZFilterSynthetic")
    if not os.path.exists(path):
        os.mkdir(path)

        # Creating directories for saving intermediate phases
        # Only the class_labels directory has class directories
        os.mkdir(os.path.join(path, "class_labels"))
        os.mkdir(os.path.join(path, "filter_selected"))
        os.mkdir(os.path.join(path, "label_removed"))

        # Lisiting and creating class directories for class_labels
        path = os.path.join(path, "class_labels")
        
        for class_name in class_names:
            dir_path = os.path.join(path, class_name)
            os.mkdir(dir_path)
    else:
        pass
            
def get_removed_label(class_names: list):
    # 1. Detect the class label in the image, save it to appropriate directory in class_labels
    # 2. Remove the class label from the image (save all images to label_removed, no class directories)
    # 3. Remove (manually?) the images that have wrong label removal
    #     -> Perhaps this can be done procedurally?

    # Listing paths
    data_types = ["train", "val", "test"]
    template_path = os.path.join("Synthetic-Samples", "templates")
    label_box_path = os.path.join("data", "NTZFilterSynthetic", "class_labels")

    # Iterating over all images usable for data generation
    for data_type in data_types:
        for class_name in class_names:
            class_path = os.path.join("data", "NTZFilter", data_type, class_name)
            files = os.listdir(class_path)
            template = Image.open(os.path.join(template_path, class_name + "_template.png"))
            for file in files:
                img = Image.open(os.path.join(class_path, file))

                # Matching each file to the class template
                top_left, bottom_right = get_template_match(img, template)
                # Cropping and saving the resulting bounding box with the label
                label_box = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
                label_box.save(os.path.join(label_box_path, class_name, file))

                # Removing the label from the image and filling up with background
                img = remove_and_fill(img, top_left, bottom_right)

                # Saving the resulting image
                img_destination = os.path.join("data", "NTZFilterSynthetic",
                                               "label_removed", file)
                img.save(img_destination)


def get_template_match(img, template):
    """This is a function that takes in an image and a template.
    It looks for the closest match of the template in the image.
    The image is normalized such that the template does not have to
    exactly match the image.

    Args:
        img: PIL image
        template: PIL image of template available in img

    Returns:
        Two tuples of (x, y) coordinates of the top left and bottom right
        bounding box coordinates.
    """
    # Converting image and template to greyscale since its easier to work with
    grey_img = np.array(img.convert("L"))
    grey_template = np.array(template.convert("L"))

    # Computing the mean and standard deviation of the template
    # and normalizing template
    mean = np.mean(grey_template)
    std_dev = np.std(grey_template)
    normalized_template = (grey_template - mean) / std_dev

    # Getting grey_template shape
    w, h = grey_template.shape
    
    # Computing the size of the resulting match
    # and initializing match array
    match_shape = (grey_img.shape[0] - w + 1,
                   grey_img.shape[1] - h + 1)
    match = np.zeros(match_shape)

    # Computing the cross-correlation between the template and the image
    for y in range(match_shape[0]):
        for x in range(match_shape[1]):
            # Getting patch from image
            patch = grey_img[y:y + w, x:x + h]

            # Computing mean and standard deviation of patch
            patch_mean = np.mean(patch)
            patch_std_dev = np.std(patch)

            # Computing normalized patch and cross-correlation
            normalized_patch = (patch - patch_mean) / patch_std_dev
            match[y, x] = np.sum(normalized_patch * normalized_template)

    # Find the location with the highest cross-correlation
    x, y = np.unravel_index(np.argmax(match), match_shape)
    top_left = (y, x)
    bottom_right = (y + h, x + w)

    return top_left, bottom_right


def remove_and_fill(img, top_left, bottom_right):
    """Function that takes a bounding box in an image
    and based on pixel intensity criteria, removes pixels
    thought to be label pixels.

    Args:
        img: PIL image
        top_left: Tuple of (x, y) coordinates of top left corner of bounding box.
        bottom_right: Tuple of (x, y) coordinates of bottom right corner of bounding box.

    Returns:
        An image with label pixels removed in the bounding box.
    """
    # Converting to greyscale and CV2 image, since its required for inpainting
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # x and y range of the bounding box
    x_range = list(range(bottom_right[0] - top_left[0]))
    y_range = list(range(bottom_right[1] - top_left[1]))
    
    # Creating initially empty mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)

    # Iterating over all pixels in the bounding box
    # Any pixel that is not very white is likely to be a label print
    for x in x_range:
        for y in y_range:
            point_x = top_left[0] + x
            point_y = top_left[1] + y
            if grey_image[point_y, point_x] < 250:
                # The pixel is likely a label print pixel
                mask[point_y, point_x] = 1

    # Using inpainting to remove label print
    # From OpenCV library, since it does not exist in PIL
    img_label_removed = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)
    # And converting back to PIL
    img_label_removed = Image.fromarray(cv2.cvtColor(img_label_removed,
                                                     cv2.COLOR_BGR2RGB))
    return img_label_removed


def active_contours():
    # Uses active contours models
    # https://scikit-image.org/docs/stable/auto_examples/edges/plot_active_contours.html
    print("Not implemented yet.")


def setup_data_generation():
    # Creating synthetic data pipeline:
    # Phase 1 - Getting all the data together
    # 1. Detect the class label in the image, save it to appropriate directory in class_labels
    # 2. Remove the class label from the image (save all images to label_removed, no class directories)
    # 3. Remove (manually?) the images that have wrong label removal
    #     -> Perhaps this can be done procedurally?
    # 4. Select the filter in the image by using the filter template, save those to filter_selected
    # 5. Remove (manually?) the images that have wrong background removal
    #     -> Again, perhaps this can be done procedurally
    # Phase 2 - Generating synthetic data
    # 6. Loop through all filters available in filter_selected, paste a class label from class_labels
    #    on it, and paste that onto the background. Voila, synthetic data.

    # OPTIONAL: Perform a dataset comparison study, is the model capable of recognizing if
    # an image is synthetic or not?
    # TODO: Fix angled rectangle mapping in template_matching()
    # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
    # Try different angles and pick best one based on some criteria
    # TODO: Fix half-print label removal (none of the labels are correctly extracted)

    class_names = ["fail_label_not_fully_printed", "fail_label_half_printed",
                   "fail_label_crooked_print", "no_fail"]

    create_synthetic_data_dirs(class_names)
    get_removed_label(class_names) # Steps 1-3
    #get_selected_filter() # Steps 4-5
    #generate_synthetic_data() # Step 6


if __name__ == '__main__':
    setup_data_generation()