import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Main.config import VISUALIZATION_FOLDER


def visualizeObjectLatentSemantic(reducer):
    with open(VISUALIZATION_FOLDER + reducer + '.json') as f:
        data = json.load(f)
    objectLatent = pd.DataFrame.from_dict(data, orient='index')
    print(objectLatent.sort_values(by=['1'], ascending=False))


visualizeObjectLatentSemantic('svd')