from os.path import join

import pandas as pd
import os
import numpy as np
from sklearn.decomposition import NMF
import json

from Main import config


def startTask8():
    image_names_list = [file for file in os.listdir(str(config.IMAGE_FOLDER))]

    meta_df = pd.read_csv(config.METADATA_FOLDER)
    handinfo = meta_df.loc[(meta_df['imageName'].isin(image_names_list))]

    x = list(handinfo.iloc[:, [1, 2, 3, 4, 5, 6, 8]].values)  # column for the things
    z = list(handinfo.iloc[:, 7])

    y = np.zeros((np.size(handinfo, 0), 8))
    images_list = []
    final = np.array(y)
    for i in range(0, len(x)):
        images_list.append(z[i])  # storing the imageId's in this list

        if x[i][1] == "male":  # 0- male
            final[i][0] = 1
            final[i][1] = 0
        elif x[i][1] == "female":  # 1- female
            final[i][0] = 0
            final[i][1] = 1

        if x[i][5] == "dorsal left":
            final[i][2] = 1  # 2- dorsal
            final[i][3] = 0  # 3- palmer
            final[i][4] = 1  # 4- left
            final[i][5] = 0  # 5- right

        elif x[i][5] == "dorsal right":
            final[i][2] = 1  # 2- dorsal
            final[i][3] = 0  # 3- palmer
            final[i][4] = 0  # 4- left
            final[i][5] = 1  # 5- right

        elif x[i][5] == "palmar left":
            final[i][2] = 0  # 2- dorsal
            final[i][3] = 1  # 3- palmer
            final[i][4] = 1  # 4- left
            final[i][5] = 0  # 5- right

        elif x[i][5] == "palmar right":
            final[i][2] = 0  # 2- dorsal
            final[i][3] = 1  # 3- palmer
            final[i][4] = 0  # 4- left
            final[i][5] = 1  # 5- right

        if x[i][3] == 1:
            final[i][6] = 1  # 6- Accesories
            final[i][7] = 0  # 7- Not Accesories
        else:
            final[i][6] = 0  # 6- Accesories
            final[i][7] = 1  # 7- Not Accesories

    Final = pd.DataFrame(final)
    Final.columns = ["Male", "Female", "Dorsal", "Palmar", "Left", "Right", "Accessories", "Not Accesories"]
    Final.head(30)

    image_name_df = pd.concat([pd.Series(images_list), Final], axis=1)
    image_name_df.head()

    k = int(input("Enter the k for Latent Semantics: "))

    model = NMF(n_components=k, init='random', random_state=0)
    W = model.fit_transform(Final)  # Excluding the first column because it is the image Id
    H = model.components_

    # handinfo is a pandas.core.series.Series, thats why we need to change it to data frame

    def display_term_weight_pairs(components, col_index, ch):
        components_df = pd.DataFrame(components)
        components_df.columns = col_index
        components_df = components_df
        output = {}
        num_components = len(components_df)
        for i in range(1, num_components + 1):
            sorted_vals = components_df.iloc[i - 1, :].sort_values(ascending=False)
            output[i] = (list(zip(sorted_vals, sorted_vals.index)))
            if ch == 'W':
                fp = open(join(config.DATABASE_FOLDER, 'task8_image_' + str(k) + '.json'), 'w+')
            elif ch == 'H':
                fp = open(join(config.DATABASE_FOLDER, 'task8_metadata_' + str(k) + '.json'), 'w+')
            json.dump(output, fp)

    display_term_weight_pairs(H, image_name_df.columns[1:], 'H')
    display_term_weight_pairs(W.T, image_name_df[0], 'W')
    print("Top k latent semantics for image and metadata space are stored successfully in the database folder")
    print("Path for metadata:",join(config.DATABASE_FOLDER, 'task8_metadata_' + str(k) + '.json'))
    print("Path for image:", join(config.DATABASE_FOLDER, 'task8_image_' + str(k) + '.json'))


if __name__ == "__main__":
    startTask8()
