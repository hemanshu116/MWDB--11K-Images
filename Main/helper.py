import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')


def findDistance(selectedImage, latentFeatureDict):
    output = {}
    for key, value in latentFeatureDict.items():
        imageFromDatabase = np.asarray(latentFeatureDict[key])
        output[key] = np.linalg.norm(selectedImage - imageFromDatabase)
    return output


def normalize_score(data):
    scaler = MinMaxScaler()
    data_array = np.asarray(data).reshape((len(data), 1))
    scaled_values = scaler.fit_transform(data_array)
    return pd.Series(scaled_values.reshape(len(scaled_values)))

def find_distance_2_vectors(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def save_pickle(pickle_file_path, obj):
    file_path = os.path.join(pickle_file_path)
    filehandler = open(file_path, 'wb')
    pickle.dump(obj, filehandler)


def load_pickle(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        return pickle.load(f)

