import numpy as np
import sys
import skimage
from skimage import io
from skimage.exposure import histogram
from skimage.feature import local_binary_pattern

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class LBP:
    def __init__(self, fileName):
        self.fileName = fileName

    def calculateFeatureDiscriptor(self):
        block_r = 100
        block_c = 100
        imageInput = io.imread(self.fileName)
        imageInput = skimage.color.rgb2gray(imageInput)
        n_points = 8
        radius = 1
        METHOD = "uniform"
        lbphist = []
        for r in range(0, imageInput.shape[0], block_r):
            for c in range(0, imageInput.shape[1], block_c):
                window = imageInput[r:r + block_r, c:c + block_c]
                hist = local_binary_pattern(window, n_points, radius, METHOD)
                output = histogram(hist,nbins=9)
                lbphist.append(output[0])
        lbphist = np.array(lbphist)
        lbphist = lbphist.flatten()
        print(lbphist)
        return lbphist
