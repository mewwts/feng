import pandas as pd
import numpy as np

def information_gain(x, y):
    "Computes the information gain of x on y"

    def entropy(x):
        p_x = x.value_counts(normalize=True)
        return -p_x.map(lambda x: x * np.log2(x)).sum()

    prior_entropy = entropy(y)
    df = pd.concat([x, y], axis=1)
    df.columns = ["x","y"]

    p_x = x.value_counts(normalize=True)
    cond_entropy = pd.Series(dict([(x_i, entropy(df[df["x"] == x_i]["y"])) for x_i in p_x.index]))
    post_entropy = p_x.dot(cond_entropy)
    information_gain = prior_entropy - post_entropy
    return information_gain

def rank_information_gain(X, y):
    information_gains = pd.Series(dict([(x, information_gain(x,y)) for x in X]))
    information_gains.column = ["feature", "information_gain"]
    return information_gains