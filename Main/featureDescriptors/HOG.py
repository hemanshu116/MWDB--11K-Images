from skimage import io
from skimage.feature import hog
from skimage.transform import rescale


class HOG:
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    bins = 9  # number of orientation bins

    def __init__(self, filename):
        self.filename = filename

    def calculateFeatureDiscriptor(self):
        image = io.imread(self.filename)
        image = rescale(image, 1.0 / 10, anti_aliasing=True)

        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True, transform_sqrt=True)
        return fd
