# %% Imports
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# %% DATA LOADING
df = pd.read_csv("data/train_essays.csv")
sns.countplot(data=df, x="generated")
# TODO: Check if we have to balance dataset for EDA
print(f"Humam written essays: {len(df[df['generated'] == 0])}")
print(f"AI generated essays: {len(df[df['generated'] == 1])}")

# %% COMPUTE CALCULATED FIELDS
df["tokens"] = df["text"].apply(lambda x: list(filter(lambda x: len(x)>0, x.split(" "))))
df["word_count"] = df["tokens"].apply((lambda x: len(x)))
df["avg_word_len"] = df["tokens"].apply((lambda x: np.mean([len(word) for word in x])))
df.head(5)

# %% INITIAL PLOTS OF THE COMPUTED VALUES
fig = plt.figure()
sns.boxplot(data=df, x="word_count", hue="generated")

fig = plt.figure()
sns.histplot(data=df, x="word_count", kde=True, hue="generated")

fig = plt.figure()
sns.histplot(data=df, x="avg_word_len", kde=True, hue="generated")

fig = plt.figure()
sns.scatterplot(data=df, x="word_count", y="avg_word_len", hue="generated")

fig = plt.figure()
text = ""
for idx, row in df.iterrows():
    text += row['text']

wordcloud = WordCloud().generate(text)

# %%
