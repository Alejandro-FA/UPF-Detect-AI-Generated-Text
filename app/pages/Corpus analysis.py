import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
from wordcloud import WordCloud
import spacy
from collections import Counter

@st.cache_data
def load_data():
    df = pd.read_csv("data/data_sample.csv")
    cols = ["word_count", "avg_word_len"]
    z = np.abs(stats.zscore(df[cols]))
    df_without_outliers = df[(z<3).all(axis=1)]

    return df, df_without_outliers



if __name__ == '__main__':

    st.header("Corpus Analysis")

    df_outliers, df_without_outliers = load_data()
   
    is_toggled = st.toggle("Remove outliers", value=True)

    if not is_toggled:
        df_plot = df_outliers
    else:
        df_plot = df_without_outliers

    fig = plt.figure()
    grouped = df_plot.groupby("label").count().values
    data = [grouped[0,0], grouped[1,0]]
    colors = sns.color_palette('pastel')[0:2]
    plt.pie(data, labels = ["Human", "AI"], colors = colors, autopct='%.0f%%')
    st.pyplot(fig)

    st.subheader("Numbers of words per essay")
    fig = plt.figure()
    sns.histplot(data=df_plot, x="word_count", kde=True, hue="label")
    st.pyplot(fig)

    fig = plt.figure()
    sns.boxplot(data=df_plot, x="word_count", hue="label")
    st.pyplot(fig)

    st.divider()
    st.subheader("Numbers of characters per word")

    fig = plt.figure()
    if not is_toggled:
        bins = 100
    else:
        bins = "auto"
    sns.histplot(data=df_plot, x="avg_word_len", kde=True, hue="label", bins=bins)
    st.pyplot(fig)

    fig = plt.figure()
    sns.boxplot(data=df_plot, x="avg_word_len", hue="label")
    st.pyplot(fig)

    st.divider()
    st.subheader("Word length vs word count")

    fig = plt.figure()
    sns.scatterplot(data=df_plot, x="word_count", y="avg_word_len", hue="label")
    st.pyplot(fig)

    st.divider()
    st.subheader("Most frequent words")
    print("Loading text")
    text = ""
    for idx, row in df_plot.iterrows():
        text += f" {row['text']}"
    print("text loaded")
    fig = plt.figure()
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

    nlp = spacy.load("en_core_web_sm")
    results = nlp(text[0:1000000])
    labels = [x.label_ for x in results.ents]
    counter = Counter(labels)
    count=counter.most_common()
    x,y=map(list,zip(*count))
    fig = plt.figure()
    sns.barplot(x=y,y=x, hue=x)
    st.pyplot(fig)


    
    