import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
from wordcloud import WordCloud
import spacy
from collections import Counter
import datasets

@st.cache_data
def load_dataset():
    return datasets.load_dataset('Alejandro-FA/ma_ai_text_data', name='eda_embedding_sample', split='train')

@st.cache_data
def load_data():
    df = pd.read_csv("data/data_sample.csv")
    cols = ["word_count", "avg_word_len"]
    z = np.abs(stats.zscore(df[cols]))
    df_without_outliers = df[(z<3).all(axis=1)]

    return df, df_without_outliers

@st.cache_data
def plot_wordcloud():
    text = ""
    for idx, row in df_plot.iterrows():
        text += f" {row['text']}"
    fig = plt.figure()
    wordcloud = WordCloud(background_color="white").generate(text[0:1000000])
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')

    st.pyplot(fig)

@st.cache_data
def plot_ner():
    text = ""
    for idx, row in df_plot.iterrows():
        text += f" {row['text']}"

    nlp = spacy.load("en_core_web_sm")
    results = nlp(text[0:1000000])
    labels = [x.label_ for x in results.ents]
    counter = Counter(labels)
    count=counter.most_common()
    x,y=map(list,zip(*count))
    fig = plt.figure()
    sns.barplot(x=y,y=x, hue=x)
    st.pyplot(fig)

def print_description(toggled_description, nt_description, is_toggled):
    if is_toggled:
        st.write(toggled_description)
    else:
        st.write(nt_description)

if __name__ == '__main__':
    st.set_page_config(
        page_title="Corpus Analysis",
        page_icon="🔎",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.header("Corpus Analysis 🔎")
    df_outliers, df_without_outliers = load_data()
   
    st.warning(
        """
        ⚠️ By toggling **Remove outliers** some of the plots and their corresponding analysis will change.\n\n
        Have fun :) !
        """
    )

    is_toggled = st.sidebar.toggle("Remove outliers", value=True)

    if not is_toggled:
        df_plot = df_outliers
    else:
        df_plot = df_without_outliers

    st.divider()
    st.subheader("Distribution of our two classes")
    fig = plt.figure()
    grouped = df_plot.groupby("label").count().values
    data = [grouped[0,0], grouped[1,0]]
    colors = sns.color_palette('pastel')[0:2]
    plt.pie(data, labels = ["Human ≡ 0", "AI ≡ 1"], colors = colors, autopct='%.0f%%')
    st.pyplot(fig)
    print_description(
        "As it can be observed, the proportion of human written essays in our dataset is slightly higher than the AI written ones.", 
        "As it can be observed, the proportion of human written essays in our dataset is slightly higher than the AI written ones.", 
        is_toggled
    )
    st.divider()

    st.subheader("Numbers of words per essay")
    tab1, tab2 = st.tabs(["Distribution", "Boxplots"])
    fig = plt.figure()
    sns.histplot(data=df_plot, x="word_count", kde=True, hue="label")
    with tab1: 
        st.pyplot(fig)

    fig = plt.figure()
    sns.boxplot(data=df_plot, y="word_count", hue="label")
    with tab2:
        st.pyplot(fig)

    st.write("""
        The number of words per essay seems to follow in both cases a Poission distribution. There is not a significative difference between both distributions apart from the skeweness. The human written essays tend to be larger than the AI generated ones.\n\n
        This does not give us any special insight since the length of the AI essay could be larger, there is not a limit on the maximum length that an AI essay can generate. Therefore, we can affirm that this effect is due to the nature of this specific dataset.
        """
    )
    st.divider()

    st.subheader("Numbers of characters per word")
    tab1, tab2 = st.tabs(["Distribution", "Boxplots"])
    fig = plt.figure()
    if not is_toggled:
        bins = 100
    else:
        bins = "auto"
    sns.histplot(data=df_plot, x="avg_word_len", kde=True, hue="label", bins=bins)
    with tab1:
        st.pyplot(fig)

    fig = plt.figure()
    sns.boxplot(data=df_plot, y="avg_word_len", hue="label")
    with tab2:
        st.pyplot(fig)
    print_description(
        """
        As we can observe there is a substantial difference in the average word length between the two classes. Both distributions are Normal. Despite there is a clear overlap between both of them, AI generated essays clearly have a higher word length than human ones.\n\n
        This can be clearly seen in the boxplot, where the overlap between the 25% and 75% quantiles is almost negligible. If we were to develop a model to classify them based on *"computed metrics"*, the average word length would be a determining variable.  
        """,
        """
        The presence of outliers does not let us analyze the distribution of the average word length. We can see that there is some non-overlapping zone, but we have to remove the outliers to do a better analysis.
        """ 
        , 
        is_toggled
    )

    st.divider()
    st.subheader("Word length vs word count")

    fig = plt.figure()
    sns.scatterplot(data=df_plot, x="word_count", y="avg_word_len", hue="label")
    st.pyplot(fig)
    print_description(
        """
        When we compare the two previously seen features, we can observe some pattern that we could already expect from the individual analysis. AI generated essays tend to have a higher average word length for both long and short essays. On the other hand, human written essays have a lower average word length for all text lenghts.
        \n\n
        It can also be observed that while AI generated essays have a higher variance in terms of average word length, human ones in general do not reach very high values.
        \n\n
        As a general conclusion, we can observe that a predictive model based on these two features would not be very precise since there is only a feature that differentiates the two classes, and not very clearly.
        """,
        """
        At first sight we can observe that there is some kind of split between AI and human essays. Human ones are longer and with a lower average word length. On the other hand, AI ones are shorter and with longer words. 
        \n\n
        By removing the outliers we can take a closer look to the data to check more in detail whether this behaviour also happens at a lower scale.
        """,
        is_toggled
    )

    st.divider()
    st.subheader("Most frequent words")
    st.warning(
        """
        ⚠️ The following plots have been performed from a **sample of our training data of 10.000 essays**. The high computation requirements along with the length of the texts results into a very time consuming process to do the plots. This is why a sampled subset of our data has been taken.
        """
    )
    plot_wordcloud()
    st.write(
        """
        From the wordcloud we can infer which is the main topic of the training essays. The most repeated terms are *student, school, people, help*..., which gives us a hint that most of the essays are related whith academic topics.
        """     
    )

    st.divider()
    st.subheader("Most popular entity types")
    plot_ner()
    st.write(
        """
        As it can be observed, the most frequent type of terms are organizations, then person names and then cardinal numbers. This does not imply that they must appear in the wordcloud, since there can be many of these words but with a low frequency.
        """     
    )   

    st.write("Type the entity above to see a more detailed description")
    entity_type = st.text_input("Entity type")
    st.write(f"Description: *{spacy.explain(entity_type)}*")

    nlp = spacy.load("en_core_web_sm")
    st.write("Type a word to check its type")
    word = st.text_input("Word")
    processed_word = ""
    if len(word)>0:
        processed_word = nlp(word.strip())[0].tag_
    st.write(f"Description: *{processed_word}*")

    st.divider()
    st.subheader("TSNE Visualization")

    tsne_datasets = load_dataset()
    embeddings_tsne_0 = tsne_datasets['embedding_tsne']
    embeddings_tsne_1 = tsne_datasets['embedding_tsne_1']
    fig = plt.figure()
    sns.scatterplot(x=embeddings_tsne_0, y=embeddings_tsne_1, hue=tsne_datasets['label'])
    st.pyplot(fig)

    st.write(
        """
        We can observe that we can split easily the human and AI generated essays after performing TSNE on their embeddings. This leads us to the conclusion that our data is quite 'artificial'.\n\n
        In the real world this difference is not that clear. So, this training data we have will in fact help the model to differentiate between the two classes in purpose, but not by nature. 
        """
    )


    
    