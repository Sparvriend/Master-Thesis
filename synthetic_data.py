import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import random
import time
import torchvision.transforms as T


# Path definitions:
TEMPLATE_PATH = os.path.join("Synthetic-Samples", "templates")
BACKGROUNDS_PATH = os.path.join("Synthetic-Samples", "background")
CLASS_LABELS_PATH = os.path.join("data", "NTZFilterSynthetic", "class_labels")
FILTER_SELECTED_PATH = os.path.join("data", "NTZFilterSynthetic", "filter_selected")
FILTER_REMOVED_PATH = os.path.join("data", "NTZFilterSynthetic", "label_removed")
SYNTHETIC_EX_PATH = os.path.join("data", "NTZFilterSynthetic", "synthetic_data_ex")


def create_synthetic_data_dirs(class_names: list):
    """Function that creates all directories required for synthetic
    data generation and saving progress of intermediate steps.

    Args:
        class_names: List of class names.
    """
    # Creating top level directory
    path = os.path.join("data", "NTZFilterSynthetic")
    if not os.path.exists(path):
        os.mkdir(path)

        # Creating directories for saving intermediate phases
        # class_labels and synthetic_data_ex have class subdirectories
        dirlist = [CLASS_LABELS_PATH, FILTER_SELECTED_PATH, 
                   FILTER_REMOVED_PATH, SYNTHETIC_EX_PATH]
        for dir in dirlist:
            os.mkdir(dir)

        # Lisiting and creating class directories for class_labels
        class_labels_path = os.path.join(path, "class_labels")
        synthetic_data_ex_path = os.path.join(path, "synthetic_data_ex")
        for class_name in class_names:
            dir_path = os.path.join(class_labels_path, class_name)
            os.mkdir(dir_path)
            dir_path = os.path.join(synthetic_data_ex_path, class_name)
            os.mkdir(dir_path)

            
def get_removed_label(class_names: list, data_types: list):
    """Function that does two primary things: 1. Matching the class template
    to the class label in the image (get_template_match), saving that
    template to the class_labels directory. 2. Removing the class label
    from the image, by filling in the pixels in the bounding box that fall
    over a certain insensity threshold (remove_and_fill).

    Args:
        class_names: List of class names
        data_types: List of data types (train, val, test)
    """
    # Iterating over all images usable for data generation
    for data_type in data_types:
        for class_name in class_names:
            class_path = os.path.join("data", "NTZFilter", data_type, class_name)
            files = os.listdir(class_path)
            template = Image.open(os.path.join(TEMPLATE_PATH, class_name + "_template.png"))
            for file in files:
                img = Image.open(os.path.join(class_path, file))

                # Matching each file to the class template
                top_left, bottom_right = get_template_match(img, template)
                # Cropping and saving the resulting bounding box with the label
                label_box = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

                # Counting the amount of label pixels
                label_pixels = np.array(label_box.convert("L"))
                label_count = 0
                for y in range(label_pixels.shape[0]):
                    for x in range(label_pixels.shape[1]):
                        if label_pixels[y, x] < 220:
                            label_count += 1
                # If not within this range, template matching failed
                if label_count < 3000 and label_count > 1000:
                    label_box.save(os.path.join(CLASS_LABELS_PATH, class_name, file))

                    # Only taking filters from fail_label_not_fully_printed and no_fail
                    # Since the other classes have unstable inpainting
                    if class_name == "fail_label_not_fully_printed" or class_name == "no_fail":
                        # Removing the label from the image and filling up with background
                        img = remove_and_fill(img, top_left, bottom_right)

                        # Saving the resulting image
                        img_destination = os.path.join(FILTER_REMOVED_PATH, file)
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


def get_selected_filter():
    """This function takes all filters with their labels removed from them
    applies normalized pattern matching to each filter to see where in the image
    the filter is located. A subset of those filters is then saved as a filter image
    in the filters_selected directory, removing the filters that had their label
    removed in an incorrect manner.
    """
    # Paths used in this function
    filter_template_path = os.path.join(TEMPLATE_PATH, "filter_template.png")

    # Get filter template and list of all filters with labels removed
    # Get a list of all filters with labels removed
    filter_template = Image.open(filter_template_path)
    files = os.listdir(FILTER_REMOVED_PATH)
    filters_selected = []

    # Setting length and width thresholds for bounding box
    # The thresholds are there such that only pixels in the middle are
    # considered for label pixel counting
    length_threshold = 12
    width_threshold = 12

    # Select the filter in the image
    for file in files:
        img = Image.open(os.path.join(FILTER_REMOVED_PATH, file))

        # Matching each file to filter template
        top_left, bottom_right = get_template_match(img, filter_template)

        # Working with a grey image to check the intensity
        grey_image = img.convert("L")

        # x and y range of the bounding box
        x_range = list(range(bottom_right[0] - top_left[0] - 2 * width_threshold))
        y_range = list(range(bottom_right[1] - top_left[1] - 2 * length_threshold))
        label_pixels = 0

        # Looping over all pixels, the intensity range of 120-150 was picked
        # by trial and error. The problem is that pixels on the edges of the filter
        # have similar intensities to label pixels.
        for x in x_range:
            for y in y_range:
                point_x = top_left[0] + x + width_threshold
                point_y = top_left[1] + y + length_threshold
                if grey_image.getpixel((point_x, point_y)) < 150 and \
                   grey_image.getpixel((point_x, point_y)) > 120:
                    # The pixel is likely a label print pixel
                    label_pixels += 1
        
        # If there are too many label pixels, the label was probably not removed correctly
        if label_pixels < 100:
            # Cropping and saving the resulting bounding box with the label
            filter_box = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
            filter_box.save(os.path.join(FILTER_SELECTED_PATH, file))
            filters_selected.append(file)
        
    print("Amount of filters selected out of total: " + 
          str(len(filters_selected)) + "/" + str(len(files)))


def paint_rect_edges(img, top_left, bottom_right, edge_width, radius):
    """This function takes a PIL image, converts to cv2, creates a mask,
    fills the mask with a rectangle based on top_left and bottom_right
    based on an edge_width. CV2's inpainting function is then used to
    normalize these mask pixels by comparing to nearest pixels within radius.

    Args:
        img: PIL image
        top_left: Tuple of (x, y) coordinates of top left corner of bounding box.
        bottom_right: Tuple of (x, y) coordinates of bottom right corner of bounding box.
        edge_width: Width of the edges of the rectangle to be filled.
        radius: Radius of the inpainting function.
    """
    # Converting to CV2, create mask and do inpainting
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Checking if the mask goes out of bounds with added edge width
    if x1 - edge_width < 0:
        x1 = 0
    if y1 - edge_width < 0:
        y1 = 0
    if x2 + edge_width > img.shape[1]:
        x2 = img.shape[1] - 1
    if y2 + edge_width > img.shape[0]:
        y2 = img.shape[0] - 1

    # Creating mask
    mask[y1 - edge_width:y1, x1 - edge_width:x2 + edge_width+1] = 1
    mask[y2 + 1:y2 + 1 + edge_width, x1 - edge_width:x2 + edge_width + 1] = 1
    mask[y1:y2 + 1, x1 - edge_width:x1] = 1
    mask[y1:y2 + 1, x2 + 1:x2 + 1 + edge_width] = 1
    img = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)

    # And converting back to PIL
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def generate_synthetic_data(class_names: list, n_data: int):
    # Getting a list of all filter images and subsetting
    all_filters = os.listdir(FILTER_SELECTED_PATH)
    subset_filters = np.random.choice(all_filters, size = n_data * len(class_names))

    # Backgrounds
    backgrounds = os.listdir(BACKGROUNDS_PATH)
    
    for i, class_name in enumerate(class_names):
        class_labels = os.listdir(os.path.join(CLASS_LABELS_PATH, class_name))
        subset_class_labels = np.random.choice(class_labels, size = n_data)
        for j, class_label in enumerate(subset_class_labels):
            # Paste the class label randomly somewhere on the filter
            # But restricted, such that it does fall in the right area
            filter_img = Image.open(os.path.join(FILTER_SELECTED_PATH,
                                                 subset_filters[i*len(class_names)+j]))
            class_label_img = Image.open(os.path.join(CLASS_LABELS_PATH,
                                                      class_name, class_label))
            
            # Converting the class_label_img to a version that only has pixels for the label
            # All other pixels should be transparent/e.g. taken from the filter iself
            class_label_img = class_label_img.convert("RGBA")
            class_label_img_grey = class_label_img.convert("L")
            mask = np.zeros((class_label_img.size[1], class_label_img.size[0]), dtype = np.uint8)
            for y in range(class_label_img.size[1]):
                for x in range(class_label_img.size[0]):
                    if class_label_img_grey.getpixel((x, y)) < 220 and \
                        class_label_img_grey.getpixel((x, y)) > 120:
                        mask[y, x] = 255
            class_label_img.putalpha(Image.fromarray(mask))
            
            # Randomly selecting label on filter location
            w_label, h_label = class_label_img.size
            w_filter, h_filter = filter_img.size
            pix_to_edge = 15
            x = random.randint(pix_to_edge, w_filter - w_label - pix_to_edge)
            y = random.randint(pix_to_edge, h_filter - h_label - pix_to_edge)

            # Applying random rotation and random horizontal flip transformation
            # Then pasting on the filter
            transform = T.Compose([T.RandomChoice([T.RandomRotation(degrees = (0, 10)),
                                                   T.Lambda(lambda x: x)]),
                                   T.RandomHorizontalFlip(p = 0.25)])
            class_label_img = transform(class_label_img)
            filter_img.paste(class_label_img, (x, y), mask = class_label_img.split()[-1])

            # Post processing step of inpainting the edges of the filter
            filter_img = paint_rect_edges(filter_img, (x, y),
                             (x + class_label_img.size[0], y + class_label_img.size[1]),
                             edge_width = 3, radius = 4)

            # Then paste the filter img randomly somewhere on the background
            # Ensuring that the top 60 pixels and bottom 30 pixels are not
            # possible, since the edge of the filter system is there
            background_img = Image.open(os.path.join(BACKGROUNDS_PATH,
                                                     random.choice(backgrounds)))
            w_background, h_background = background_img.size
            x = random.randint(0, w_background - w_filter)
            y = random.randint(60, h_background - h_filter - 30)

            # Pasting filter with label, applying post-processing and saving
            background_img.paste(filter_img, (x, y))
            background_img = paint_rect_edges(background_img, (x, y),
                             (x + filter_img.size[0], y + filter_img.size[1]),
                             edge_width = 8, radius = 20)
            background_img.save(os.path.join(SYNTHETIC_EX_PATH, class_name,
                                             "example_" + str(i*len(class_names) + j) + ".png"))


def check_incorrect_match(x_range, y_range, image, top_left, threshold):
    # Working with a grey image to check the intensity
    grey_image = image.convert("L")
    pixels = 0
    # Looping over pixels
    for x in x_range:
        for y in y_range:
            point_x = top_left[0] + x
            point_y = top_left[1] + y 
            if grey_image.getpixel((point_x, point_y)) < threshold:
                pixels += 1
    return pixels


def test():
    # THIS FUNCTION REPLACES get_selected_filter()
    # TODO: In generate_synthetic_data, only print the label in the confines of the coordinates given in
    # the filter_coordinates dictionary for the file. Also, only print any pixels that are not completely
    # black (0, 0, 0) - see putalpha in generate_synthetic_data how to do that with opacity.
    # TODO: Replace incorrect match checks with the check_incorrect_match() function
    # TODO: After extraction of the image, do a personal check of the extracted images
    # to see which one were done correctly and which ones were not.

    files = os.listdir(FILTER_REMOVED_PATH)
    cutoff_template = Image.open(os.path.join(TEMPLATE_PATH, "filter_cutoff_template.png"))
    succes_count = 0
    filter_coordinates = {}

    # Create cylinder template list
    cylinder_templates = ["filter_cylinder_template_r.png", "filter_cylinder_template_l1.png",
                          "filter_cylinder_template_l2.png"]

    for file in files:
        img = Image.open(os.path.join(FILTER_REMOVED_PATH, file))

        # The cutoff template is almost always (at least partially) correct
        top_left_f, bottom_right_f = get_template_match(img, cutoff_template)
        filter_coordinates[file] = (top_left_f, bottom_right_f)

        # It can sometimes happen that the label on the filter is removed
        # incorrectly in get_removed_label, but those are often not selected
        # anyway in this procedure, if in the future that changes, see old commits
        # for a method on how to solve that.

        for cylinder_template_name in cylinder_templates:
            cylinder_template = Image.open(os.path.join(TEMPLATE_PATH, cylinder_template_name))
            top_left_c, bottom_right_c = get_template_match(img, cylinder_template)

            x_range = list(range(bottom_right_c[0] - top_left_c[0]))
            y_range = list(range(bottom_right_c[1] - top_left_c[1]))

            # Checking for an incorrect match
            cylinder_pixels = check_incorrect_match(x_range, y_range, img, top_left_c, threshold = 60)

            if cylinder_pixels > 200:
                # If the match is accurate, combine the two rectangles
                top_left_cc = (min(top_left_f[0], top_left_c[0]), min(top_left_f[1], top_left_c[1]))
                bottom_right_cc = (max(bottom_right_f[0], bottom_right_c[0]), max(bottom_right_f[1], bottom_right_c[1]))

                # Creating a new image to draw the two rectangles on
                width = bottom_right_cc[0] - top_left_cc[0]
                height = bottom_right_cc[1] - top_left_cc[1]
                combined = Image.new('RGB', (width, height), (0, 0, 0))

                # Filter image crop and paste
                area_f = img.crop((top_left_f[0], top_left_f[1], bottom_right_f[0], bottom_right_f[1]))
                combined.paste(area_f, (top_left_f[0] - top_left_cc[0], top_left_f[1] - top_left_cc[1]))

                # Black cylinder image crop and paste
                area_c = img.crop((top_left_c[0], top_left_c[1], bottom_right_c[0], bottom_right_c[1]))
                combined.paste(area_c, (top_left_c[0] - top_left_cc[0], top_left_c[1] - top_left_cc[1]))

                # Add in the space between the two template matches, if it is not connected
                # This is the cylinder template on the right side case
                if top_left_f[0] != 0 and bottom_right_f[0] != top_left_c[0]:
                    top_left_i = (bottom_right_f[0], top_left_c[1])
                    bottom_right_i = bottom_right_c
                    area_i = img.crop((top_left_i[0], top_left_i[1], bottom_right_i[0], bottom_right_i[1]))
                    combined.paste(area_i, (top_left_i[0] - top_left_cc[0], top_left_i[1] - top_left_cc[1]))

                # Cylinder template on the left side case
                if bottom_right_f[0] != 0 and top_left_f[0] != bottom_right_c[0]:
                    top_left_i = (bottom_right_c[0], top_left_c[1])
                    bottom_right_i = (top_left_f[0], bottom_right_c[1])
                    area_i = img.crop((top_left_i[0], top_left_i[1], bottom_right_i[0], bottom_right_i[1]))
                    combined.paste(area_i, (top_left_i[0] - top_left_cc[0], top_left_i[1] - top_left_cc[1]))

                # Perform a final check to remove incorrect template matches
                x_range = list(range(width))
                y_range = list(range(height))
                black_pixels = check_incorrect_match(x_range, y_range, combined, [0, 0], threshold = 1)
                if black_pixels < 1500:
                    succes_count += 1
                    combined.save(os.path.join("data", "NTZFilterSynthetic", "test", file + "_test" + ".bmp"))
                    break

    print("Succesful template matches = " + str(succes_count) + "/" + str(len(files)))
    return filter_coordinates
        

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
    #    on it, and paste that onto the background.
    # 7. Post-processing: Cleaning the images pasted on top of each other
    # Applying inpainting around the rectangle that is pasted on the image
    # (Applied after each pasting step)

    # Listing class names and data types
    class_names = ["fail_label_not_fully_printed", "fail_label_half_printed",
                   "fail_label_crooked_print", "no_fail"]
    data_types = ["train", "val", "test"]
    start_time = time.time()

    # print("Creating Synthetic data directories")
    # create_synthetic_data_dirs(class_names)
    # print("Getting class labels and saving filters without labels")
    # get_removed_label(class_names, data_types) # Steps 1-3
    # print("Matching filters")
    # get_selected_filter() # Steps 4-5
    # print("Generating Synthetic data")
    # generate_synthetic_data(class_names, 25) # Step 7

    filter_coordinates = test()

    # Recording total time passed
    # Steps 1-3 take about 9 minutes, the other steps take much less time.
    elapsed_time = time.time() - start_time
    print("Total data generation time (H/M/S) = ", 
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


if __name__ == '__main__':
    setup_data_generation()