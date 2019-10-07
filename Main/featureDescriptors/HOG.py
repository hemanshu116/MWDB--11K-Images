import os

from skimage import io
from skimage.feature import hog
from skimage.transform import rescale

import Main.config as config


class HOG:
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    bins = 9  # number of orientation bins

    @classmethod
    def HOGForSingleImage(self, filename):
        image = io.imread(filename)
        image = rescale(image, 1.0 / 10, anti_aliasing=True)

        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True, transform_sqrt=True)
        return fd

    @classmethod
    def HOGFeatureDescriptor(self):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images
        storeHogFD = []
        hog = HOG();
        for file in os.listdir(str(config.IMAGE_FOLDER)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                hognp = hog.HOGForSingleImage(str(config.IMAGE_FOLDER) + "\\" + filename)
                storeHogFD.append(hognp.tolist())
        return storeHogFD
