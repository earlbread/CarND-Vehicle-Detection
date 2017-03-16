import os
import fnmatch

def get_image_files(image_path):
    """Return a list of all image files from given image path.
    """
    image_files = [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(image_path)
            for f in fnmatch.filter(files, '*.png')]

    return image_files
