from Main import config
from Main.tasks.Task6 import create_feature_df, generate_similarity
import pandas as pd
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import json
import os


def startTask7():
    print("Please enter the k: ")
    k = input()

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

    nmf = NMF(n_components=int(k), init='random', random_state=0)
    nmf.fit(sim_df)
    components = nmf.components_
    plot_term_weight_pairs(components, list(map(str, sim_df.index)))


def plot_term_weight_pairs(components, col_index):
    components_df = pd.DataFrame(components)
    components_df.columns = col_index

    num_components = len(components_df)
    columns = 1
    rows = (num_components / columns) + 1

    output = {}

    fig = plt.figure(figsize=(20, 40))
    for i in range(1, num_components + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title("Topic_" + str(i))
        sorted_vals = components_df.iloc[i - 1, :].sort_values(ascending=False)
        plt.bar(sorted_vals.index, sorted_vals)

        output[i] = (list(zip(sorted_vals, sorted_vals.index)))

        ax.tick_params(direction='inout')
        ax.set_xticklabels(sorted_vals.index, rotation=45, ha='right')
    plt.show()

    fp = open(os.path.join(config.DATABASE_FOLDER , 'Task7.json'), 'w+')
    json.dump(output, fp)


if __name__ == "__main__":
    startTask7()
