import os
import fnmatch

import cv2

def get_image_files(image_path):
    """Return a list of all image files from given image path.
    """
    image_files = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(image_path)
            for f in fnmatch.filter(files, '*.png')]

    return image_files


def get_images(image_files):
    images = []
    for f in image_files:
        image = cv2.imread(f)

        images.append(image)

    return images


def get_images_from_path(image_path):
    image_files = get_image_files(image_path)
    images = get_images(image_files)

    return images
