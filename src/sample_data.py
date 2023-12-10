# %% Imports
import pandas as pd
import numpy as np
from transformers import pipeline
import os
import datasets
from utils import get_torch_device
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def embed(data):
  e = embedder(data['text'])
  n = np.array(e)
  mean_array = np.squeeze(np.mean(n, axis=1))
    
  return {"embedding": mean_array}


# %% DATA LOADING
if __name__ == '__main__':

    df = pd.read_csv("./../data/archive/final_train.csv")

    # %% SAMPLING
    df = df.sample(n=10000)
    df["words"] = df["text"].apply(lambda x: list(filter(lambda x: len(x)>0, x.split(" "))))
    df["word_count"] = df["words"].apply((lambda x: len(x)))
    df["avg_word_len"] = df["words"].apply((lambda x: np.mean([len(word) for word in x])))
    df.to_csv("./../data/data_sample.csv")


    # %% EMBEDDING MODEL LOADING
    DATA_FOLDER = '../data'

    embedder = pipeline(
        'feature-extraction',
        model='Alejandro-FA/ma_ai_text',
        device=get_torch_device(debug=True),
        truncation=True,
        padding=True,
        top_k=1,
    )
    # %% EMBEDDING SAMPLE TEXTS : This is computationally expensive so we recommend
    #    to run it on a GPU
    dataset = datasets.Dataset.from_pandas(df)
    print("Embedding sample dataset...")
    dataset = dataset.map(embed)
    print("Embedding ended")
        
    # %% TSNE
    text_embeddings = np.array(dataset['embedding'])
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(text_embeddings)

    # We persist the vectors after the TSNE to improve the streamlit app performance.
    dataset = dataset.add_column("embedding_tsne", embeddings_tsne[:, 0])
    dataset = dataset.add_column("embedding_tsne_1", embeddings_tsne[:, 1])
    dataset.save_to_disk(f"{DATA_FOLDER}/tokenized_datasets/sampled")

    # %% VISUALIZATION
    fig, ax = plt.subplots()
    scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c = dataset['label'])
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    plt.title('t-SNE Visualization of Text Embeddings')
    ax.add_artist(legend1)
    plt.show()

