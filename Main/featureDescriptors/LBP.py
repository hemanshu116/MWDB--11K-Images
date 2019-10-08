import os

import numpy as np
import sys
import skimage
from skimage import io
from skimage.exposure import histogram
from skimage.feature import local_binary_pattern

from Main import config
from Main.helper import progress

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class LBP:

    def LBPForSingleImage(self,fileName):
        block_r = 100
        block_c = 100
        imageInput = io.imread(fileName)
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
        return lbphist

    @classmethod
    def LBPFeatureDescriptor(self):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images
        storeLbpFD = []
        lbp = LBP();
        files = os.listdir(str(config.IMAGE_FOLDER))  # dir is your directory path
        number_files = len(files)
        i = 0;
        for file in os.listdir(str(config.IMAGE_FOLDER)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                hognp = lbp.LBPForSingleImage(str(config.IMAGE_FOLDER) + "\\" + filename)
                storeLbpFD.append(hognp.tolist())
                i = i + 1
                progress(i, number_files)
        print()
        return storeLbpFD
