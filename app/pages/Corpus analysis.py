import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/archive/final_train.csv")


if __name__ == '__main__':

    st.header("Corpus Analysis")

    #df = load_data()
    df["words"] = df["text"].apply(lambda x: list(filter(lambda x: len(x)>0, x.split(" "))))
    df["word_count"] = df["words"].apply((lambda x: len(x)))
    df["avg_word_len"] = df["words"].apply((lambda x: np.mean([len(word) for word in x])))

    # %% INITIAL PLOTS OF THE COMPUTED VALUES
    fig = plt.figure()
    sns.boxplot(data=df, x="word_count", hue="generated")
    st.pyplot(fig)

    fig = plt.figure()
    sns.histplot(data=df, x="word_count", kde=True, hue="generated")
    st.pyplot(fig)

    fig = plt.figure()
    sns.histplot(data=df, x="avg_word_len", kde=True, hue="generated")
    st.pyplot(fig)


    fig = plt.figure()
    sns.scatterplot(data=df, x="word_count", y="avg_word_len", hue="generated")
    st.pyplot(fig)

    
    