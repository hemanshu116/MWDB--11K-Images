import cv2
import numpy as np
import pandas as pd
from easygui import *
from scipy.stats.mstats import skew
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn import preprocessing

np.set_printoptions(precision=1)
np.set_printoptions(threshold=np.inf)


# Function to load images
# Supports color image and Y channel

def load_img_by_id(imgPath, return_gray=True):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if return_gray:
        return img[:, :, 0]
    return img


# Function to compute color moments
def compute_color_moments(img, x_window_size=100, y_window_size=100):
    x_num_windows = img.shape[0] / x_window_size
    y_num_windows = img.shape[1] / y_window_size

    x_splitted = np.split(img, x_num_windows)
    y_splitted = map(lambda x: np.asarray(np.split(x, y_num_windows, axis=1)), x_splitted)
    y_splitted_xsize_ysize = np.asarray(y_splitted)

    mean_Y = np.mean(y_splitted_xsize_ysize[:, :, :, :, 0], (2, 3), dtype=np.float32)
    mean_U = np.mean(y_splitted_xsize_ysize[:, :, :, :, 2], (2, 3), dtype=np.float32)
    mean_V = np.mean(y_splitted_xsize_ysize[:, :, :, :, 1], (2, 3), dtype=np.float32)

    std_Y = np.std(y_splitted_xsize_ysize[:, :, :, :, 0], (2, 3), dtype=np.float32)
    std_U = np.std(y_splitted_xsize_ysize[:, :, :, :, 1], (2, 3), dtype=np.float32)
    std_V = np.std(y_splitted_xsize_ysize[:, :, :, :, 2], (2, 3), dtype=np.float32)

    skew_Y = skew(y_splitted_xsize_ysize[:, :, :, :, 0].reshape(x_num_windows, y_num_windows,
                                                                x_window_size * y_window_size), axis=2)
    skew_U = skew(y_splitted_xsize_ysize[:, :, :, :, 1].reshape(x_num_windows, y_num_windows,
                                                                x_window_size * y_window_size), axis=2)
    skew_V = skew(y_splitted_xsize_ysize[:, :, :, :, 2].reshape(x_num_windows, y_num_windows,
                                                                x_window_size * y_window_size), axis=2)

    all_features = np.dstack((mean_Y, std_Y, skew_Y, mean_U, std_U, skew_U, mean_V, std_V, skew_V))

    return all_features.reshape(x_num_windows * y_num_windows * all_features.shape[2])


# Function to compute sift features
def compute_sift_features(img_gray):
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, desc = sift.detectAndCompute(img_gray, None)
    return np.asarray(desc)


# Function to execute a given feature extraction technique per folder
def compute_features_by_folder(folder_name, ftr_comp_func, return_gray=True,
                               folder_base_path='/Users/vedavyas/Desktop/CSE515/dataset/'):
    folder_path = folder_base_path + folder_name + '/'
    fileNames = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and not (f.startswith('.')))]
    feature_list = []
    for fileName in fileNames:
        print(fileName)
        # print folder_path
        img = load_img_by_id(fileName, return_gray)
        features = ftr_comp_func(img)
        feature_list.append(features)
    return pd.DataFrame({'FileName': fileNames, 'Features': feature_list})


# Function to execute a given feature extraction technique per single file
def compute_features_by_file(file_name, ftr_comp_func, return_gray=True,
                             file_base_path='/Users/vedavyas/Desktop/CSE515/dataset/Hands_Test/'):
    feature_list = []
    fileNames = [file_name]
    print(file_name)
    img = load_img_by_id(file_name, return_gray)
    features = ftr_comp_func(img)
    feature_list.append(features)
    return pd.DataFrame({'FileName': fileNames, 'Features': feature_list})


# Function to Bag of Visual Words
def compute_BOVW(feature_descriptors, n_clusters=100):
    print("Bag of visual words with clusters: ", n_clusters)
    # print feature_descriptors.shape

    combined_features = np.vstack(np.array(feature_descriptors))
    print("Size of stacked features: ", combined_features.shape)

    std_scaler = StandardScaler()
    combined_features = std_scaler.fit_transform(combined_features)

    print("Starting K-means training")
    kmeans = KMeans(n_clusters=n_clusters, random_state=777).fit(combined_features)

    print("Finished K-means training, moving on to prediction")
    bovw_vector = np.zeros([len(feature_descriptors), n_clusters])

    for index, features in enumerate(feature_descriptors):
        features_scaled = std_scaler.transform(features)
        for i in kmeans.predict(features_scaled):
            bovw_vector[index, i] += 1

    bovw_vector_normalized = preprocessing.normalize(bovw_vector, norm='l2')

    print("Finished K-means")
    return list(bovw_vector_normalized)


# Function to save features for the purpose of viewing in a human readable format
def save_features_for_view(filename, feature_df_input, save_folder='FileFeatures',
                           save_base_path='/Users/vedavyas/Desktop/CSE515/dataset/'):
    feature_df = feature_df_input.copy()
    pd.set_option('display.max_colwidth', -1)

    def str_formatter(string):
        return np.array2string(string, formatter={'float_kind': lambda x: "%.3f" % x}, separator=', ')

    feature_df['Features'] = feature_df['Features'].apply(str_formatter)

    path = save_base_path + '/' + save_folder + '/' + filename
    feature_df.to_html(path, index=False, formatters={'Features': lambda x: x.replace("\\n", "<br>")})


# Function to save features in a pickle format convenient for retrieval
def save_features_pickle(filename, feature_df, save_folder='FileFeatures',
                         save_base_path='/Users/vedavyas/Desktop/CSE515/dataset/'):
    path = save_base_path + '/' + save_folder + '/' + filename
    feature_df.to_pickle(path)


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(np.subtract(vec1, vec2))))


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    return dot_product / (norm_1 * norm_2)


def compute_distance(vec_list, curr_vec, distance_metric_func):
    return vec_list.apply(distance_metric_func, args=(curr_vec,))


# Function to perfrom min max normalization
def normalize_score(data):
    scaler = MinMaxScaler()
    data_array = np.asarray(data).reshape((len(data), 1))
    scaled_values = scaler.fit_transform(data_array)
    return pd.Series(scaled_values.reshape(len(scaled_values)))


# main function to launch the visual prompt
if __name__ == "__main__":

    msg = "Choose a Task"
    title = "Task selection"
    choices = ["Single Image (Task 1)", "Folder (Task 2)", "Search (Task 3)"]
    model_run_mode = choicebox(msg, title, choices)
    print("Input Entered:", model_run_mode)

    msg = "Choose a model"
    title = "Model selection"
    choices = ["Color Moments", "SIFT"]
    input_model = choicebox(msg, title, choices)
    print("Input Entered:", input_model)

    title = "Name"
    if (model_run_mode == "Single Image (Task 1)"):
        msg = "Enter a file name"
        input_file = enterbox(msg, title, )
    elif (model_run_mode == "Folder (Task 2)"):
        msg = "Enter a folder name"
        input_file = enterbox(msg, title, )
    else:
        msg = "Enter a image name to search"
        multi_input = multenterbox(msg, title, ['Image Name', 'Number of images'])
        input_file = multi_input[0]
        num_images = int(multi_input[1])

    print("Input Entered:", input_file)

    if (input_model == "Color Moments"):
        if (model_run_mode == "Single Image (Task 1)"):
            feature_df = compute_features_by_file(input_file, compute_color_moments, return_gray=False)
            out_feature_filename = input_file + "_color_moments_features.html"
            save_features_for_view(out_feature_filename, feature_df)
        elif model_run_mode == "Folder (Task 2)":
            feature_df = compute_features_by_folder(input_file, compute_color_moments, return_gray=False)
            save_features_for_view("color_moments_features.html", feature_df, 'FolderFeatures')
            save_features_pickle("color_moments_features.pkl", feature_df, 'FolderFeatures')
        else:
            loaded_features_df = pd.read_pickle(
                '/Users/vedavyas/Desktop/CSE515/dataset/FolderFeatures/color_moments_features.pkl')
            print("Size of features loaded: ", loaded_features_df.shape)
            curr = loaded_features_df[loaded_features_df['FileName'] == input_file]['Features']
            dists = compute_distance(loaded_features_df['Features'], list(curr)[0], euclidean_distance)
            loaded_features_df['Distance'] = dists
            loaded_features_df['Normalized Distance'] = normalize_score(dists)
            loaded_features_df = loaded_features_df.sort_values(by='Distance')
            distances = loaded_features_df[['FileName', 'Distance', 'Normalized Distance']]
            distances.to_csv('Result_Color_' + input_file + '.csv', index=False, float_format='%.2f')

            fig = plt.figure(figsize=(20, 20))
            columns = 6
            rows = (num_images / 6) + 1
            for i in range(1, columns * rows + 1):
                if i > num_images:
                    break
                file_name = list(distances['FileName'])[i - 1]
                distance = list(distances['Normalized Distance'])[i - 1]
                img = load_img_by_id(file_name, return_gray=False)
                ax = fig.add_subplot(rows, columns, i)
                ax.set_title(file_name + "_" + str("{0:.2f}".format(distance)))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_YUV2RGB))
            plt.show()

    else:
        if (model_run_mode == "Single Image (Task 1)"):
            feature_df = compute_features_by_file(input_file, compute_sift_features)
            out_feature_filename = input_file + "_sift_features.html"
            save_features_for_view(out_feature_filename, feature_df)
        elif (model_run_mode == "Folder (Task 2)"):
            feature_df = compute_features_by_folder(input_file, compute_sift_features)
            bow_list = compute_BOVW(feature_df['Features'], 300)
            feature_df['BOVW_features'] = pd.Series(bow_list)
            save_features_for_view("sift_features.html", feature_df, 'FolderFeatures')
            save_features_pickle("sift_features.pkl", feature_df, 'FolderFeatures')
        else:
            loaded_features_df = pd.read_pickle(
                '/Users/vedavyas/Desktop/CSE515/dataset/FolderFeatures/sift_features.pkl')
            print("Size of features loaded: ", loaded_features_df.shape)
            curr = loaded_features_df[loaded_features_df['FileName'] == input_file]['BOVW_features']
            dists = compute_distance(loaded_features_df['BOVW_features'], list(curr)[0], euclidean_distance)
            loaded_features_df['Distance'] = dists
            loaded_features_df['Normalized Distance'] = normalize_score(dists)
            loaded_features_df = loaded_features_df.sort_values(by='Distance', ascending=True)

            distances = loaded_features_df[['FileName', 'Distance', 'Normalized Distance']]

            distances.to_csv('Result_SIFT_' + input_file + '.csv', index=False, float_format='%.2f')

            # loop over the results
            fig = plt.figure(figsize=(20, 20))
            columns = 6
            rows = (num_images / 6) + 1
            for i in range(1, columns * rows + 1):
                if i > num_images:
                    break
                file_name = list(distances['FileName'])[i - 1]
                distance = list(distances['Normalized Distance'])[i - 1]
                img = load_img_by_id(file_name, return_gray=False)
                ax = fig.add_subplot(rows, columns, i)
                ax.set_title(file_name + "_" + str("{0:.2f}".format(distance)))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_YUV2RGB))
            plt.show()
