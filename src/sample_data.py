# %% Imports
import pandas as pd
import numpy as np

# %% DATA LOADING
if __name__ == '__main__':

    df = pd.read_csv("./../data/archive/final_train.csv")

    # %% SAMPLING
    df = df.sample(n=10000)
    df["words"] = df["text"].apply(lambda x: list(filter(lambda x: len(x)>0, x.split(" "))))
    df["word_count"] = df["words"].apply((lambda x: len(x)))
    df["avg_word_len"] = df["words"].apply((lambda x: np.mean([len(word) for word in x])))
    df.to_csv("./../data/data_sample.csv")


# %%
