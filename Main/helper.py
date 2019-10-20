import numpy as np
import pickle
import os


def progress(count, total):
    print('\r', str((count / total) * 100) + "% completed", end=' ')


def findDistance(selectedImage, latentFeatureDict):
    output = {}
    for key, value in latentFeatureDict.items():
        imageFromDatabase = np.asarray(latentFeatureDict[key])
        output[key] = np.linalg.norm(selectedImage - imageFromDatabase)
    return output


def find_distance_2_vectors(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def save_pickle(pickle_file_path, obj):
    file_path = os.path.join(pickle_file_path)
    filehandler = open(file_path, 'wb')
    pickle.dump(obj, filehandler)


def load_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        return pickle.load(f)

