import cv2
import pandas as pd
import numpy as np
import Main.config as config
from Main.featureDescriptors.HOG import HOG
from Main.featureDescriptors.CM import CM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


def startTask6():
    print("Please enter the subject id: ")
    subject_id = input()

    meta_df = pd.read_csv(config.METADATA_FOLDER)

    try:
        sim_df = pd.read_pickle(os.path.join(config.DATABASE_FOLDER, 'Similarity_Matrix.pkl'))
    except:
        final_feature_df = create_feature_df()
        sim_df = generate_similarity(final_feature_df)
    else:
        print("Similarity matrix already exists. Do you want to recompute y/n ?")
        str_input = input()
        if str_input == 'y':
            final_feature_df = create_feature_df()
            sim_df = generate_similarity(final_feature_df)

    sim_df.to_pickle(os.path.join(config.DATABASE_FOLDER, 'Similarity_Matrix.pkl'))

    print(get_similar_subjects(sim_df, int(subject_id), 5, meta_df).to_string())

def create_feature_df():
    image_names_list = [file for file in os.listdir(str(config.IMAGE_FOLDER))]

    meta_df = pd.read_csv(config.METADATA_FOLDER)
    imageSet = meta_df.loc[(meta_df['imageName'].isin(image_names_list))]

    hog = HOG()
    featureVector_hog = hog.HOGFeatureDescriptorForImageSubset(imageSet['imageName'])

    cm = CM()
    featureVector_cm = cm.CMFeatureDescriptorForImageSubset(imageSet['imageName'])

    featureVector = np.concatenate([featureVector_hog, featureVector_cm], axis=1)
    final_feature_df = pd.concat([pd.DataFrame(imageSet).reset_index(drop=True), pd.DataFrame(featureVector)], axis=1)
    return final_feature_df


def generate_similarity(all_featureVector_df):
    reduced_features_df = apply_pca(all_featureVector_df[all_featureVector_df.columns[9:]], 90)
    final_feature_df = pd.concat([all_featureVector_df[all_featureVector_df.columns[0:9]].reset_index(drop=True)
                                     , pd.DataFrame(reduced_features_df)], axis=1)
    sim_df_cos = compute_subject_subject_similarity(final_feature_df, cosine_similarity)
    return sim_df_cos


def apply_pca(featureVector_df, k):
    scaler = StandardScaler()
    scaled_feature_df = scaler.fit_transform(featureVector_df)

    pca = PCA()
    pca.fit(scaled_feature_df)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    pca = PCA(k)
    reduced_features_df = pca.fit_transform(scaled_feature_df)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    print('Retained Variance: ', np.cumsum(pca.explained_variance_ratio_).max())
    return reduced_features_df


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(np.subtract(vec1, vec2))))


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    return (1 + (dot_product / (norm_1 * norm_2))) / 2


def compute_subject_subject_similarity(final_feature_df, metric):
    groups_df = final_feature_df.groupby(['id'])
    subject_id_list = []
    subject_feature_list = []
    for group in groups_df:
        subject_id = group[0]
        subject_df = group[1]
        subject_features_df = subject_df[subject_df.columns[9:]]
        subject_features = subject_features_df.mean(axis=0)
        subject_id_list.append(subject_id)
        subject_feature_list.append(subject_features)

    distance_list = []
    for i_vec in subject_feature_list:
        distance_list_temp = []
        for j_vec in subject_feature_list:
            dist = metric(i_vec, j_vec)
            distance_list_temp.append(dist)
        distance_list.append(distance_list_temp)

    sim_df = pd.DataFrame(distance_list)
    sim_df.columns = subject_id_list
    sim_df.index = subject_id_list
    return sim_df


def get_similar_subjects(sim_df, subject_id, num_similar, meta_df,ascending=False):
    sim_values = sim_df[subject_id].sort_values(ascending=ascending).head(num_similar)
    subject_meta = meta_df.loc[(meta_df['id'].isin(sim_values.index))]

    temp_df = subject_meta[
        ['id', 'age', 'gender', 'skinColor', 'accessories', 'nailPolish', 'irregularities']].drop_duplicates()
    subject_info_df = pd.DataFrame(
        temp_df.groupby(['id', 'age', 'gender', 'skinColor'], sort=False).max()).reset_index()
    subject_info_df.index = subject_info_df['id']
    subject_info_df['similarity'] = sim_values
    return subject_info_df.sort_values('similarity', ascending=ascending)

if __name__ == "__main__":
    startTask6()