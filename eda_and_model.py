# %% Imports
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib as plt


# %% DATA LOADING
df = pd.read_csv("data/train_essays.csv")
sns.countplot(data=df, x="generated")
# TODO: Check if we have to balance dataset for EDA
print(f"Humam written essays: {len(df[df['generated'] == 0])}")
print(f"AI generated essays: {len(df[df['generated'] == 1])}")

# %% This is another cell

