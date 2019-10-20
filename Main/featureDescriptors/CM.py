import os

import cv2
from scipy.stats.mstats import skew
import numpy as np

# a method that partitions a input image into 100 * 100 windows
from Main import config
from Main.helper import progress


def get_100_by_100_windows(input_image):
    vertical_partitions = input_image.shape[1] / 100
    horizontal_partitions = input_image.shape[0] / 100
    windows_set = []
    windows_set_1 = np.vsplit(input_image, horizontal_partitions)
    for np_array in windows_set_1:
        windows_set_2 = np.hsplit(np_array, vertical_partitions)
        for i in windows_set_2:
            windows_set.append(i)
    return windows_set


class CM:

    @classmethod
    def CMForSingleImage(self, filename):
        # Computing feature descriptors for color moments for task 1.
        input_image = cv2.imread(filename)
        # converting the input image to yuv before computing image color moments
        yuv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
        windows_set = get_100_by_100_windows(yuv_image)

        y_channel_descriptor = []
        u_channel_descriptor = []
        v_channel_descriptor = []
        for i in windows_set:
            y_channel = i[:, :, 0]
            u_channel = i[:, :, 1]
            v_channel = i[:, :, 2]
            # computing the mean(first moment) for each channel
            first_moment_y = np.mean(y_channel)
            first_moment_u = np.mean(u_channel)
            first_moment_v = np.mean(v_channel)
            # computing the standard deviation(second moment) for each channel
            second_moment_y = np.std(y_channel)
            second_moment_u = np.std(u_channel)
            second_moment_v = np.std(v_channel)
            # computing the skewness(third moment) for each channel
            third_moment_y = skew(y_channel, axis=None)
            third_moment_u = skew(u_channel, axis=None)
            third_moment_v = skew(v_channel, axis=None)
            # each of the moment value is rounded to three decimals. Easy to read
            np.around([0.37, 1.64], decimals=1)
            y_channel_descriptor.extend(
                [np.around(first_moment_y, 3), np.around(second_moment_y, 3), np.around(third_moment_y, 3)])
            u_channel_descriptor.extend(
                [np.around(first_moment_u, 3), np.around(second_moment_u, 3), np.around(third_moment_v, 3)])
            v_channel_descriptor.extend(
                [np.around(first_moment_v, 3), np.around(second_moment_v, 3), np.around(third_moment_v, 3)])
        return np.asarray(y_channel_descriptor + u_channel_descriptor + v_channel_descriptor)

    @classmethod
    def CMFeatureDescriptor(self):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images
        storeCmFD = []
        cm = CM()
        files = os.listdir(str(config.IMAGE_FOLDER))  # dir is your directory path
        number_files = len(files)
        i = 0
        for file in os.listdir(str(config.IMAGE_FOLDER)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                hognp = cm.CMForSingleImage(str(config.IMAGE_FOLDER) + "\\" + filename)
                storeCmFD.append(hognp.tolist())
                i = i + 1
                progress(i, number_files)
        # print()
        return storeCmFD

    @classmethod
    def CMFeatureDescriptorForImageSubset(self, imageSet):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images in the subset
        storeCmFD = []
        cm = CM()
        files = os.listdir(str(config.IMAGE_FOLDER))  # dir is your directory path
        number_files = len(imageSet)
        i = 0
        for filename in imageSet:
            hognp = cm.CMForSingleImage(str(config.IMAGE_FOLDER) + "\\" + filename)
            storeCmFD.append(hognp.tolist())
            i = i + 1
            progress(i, number_files)
        print(len(storeCmFD))
        return storeCmFD