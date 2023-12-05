# https://huggingface.co/docs/transformers/tasks/sequence_classification

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import datasets


def preprocess_function(data):
    return tokenizer(data['text'], truncation=True)

def to_dataset(texts, labels):
    assert(len(texts) == len(labels))
    output = {
        "text": list(map(lambda x: str(x).strip(), texts)),
        "label": labels,
    }
    return Dataset.from_dict(output)

def print_stats(name, dataset ):
    count_ai = len([i for i in dataset["label"] if i == 1])
    print(f"## Dataset: {name} ##")
    print(f" - AI generated: {count_ai}")
    print(f" - Human written: {len(dataset['label'])-count_ai}")

def balance(df):
    ai = df[df['generated']==1]
    human = df[df['generated']==0]
    if len(ai) < len(human):
        human = human.sample(n=len(ai))
    else:
        ai = ai.sample(n=len(human))
    return pd.concat([ai, human]).sample(frac=1)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

df = balance(pd.read_csv("data/train_essays.csv"))

assert(df[df["generated"]==0].count().generated == df[df["generated"]==1].count().generated)

X = df['text'].values
y = df['generated'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

datasets = datasets.DatasetDict({
    "train":to_dataset(X_train, y_train),
    "test":to_dataset(X_test, y_test)
})

print("Tokenizing...")
tokenized_datasets = datasets.map(preprocess_function, batched=True)
print("Tokenization ended")

for dataset in tokenized_datasets:
    print_stats(dataset, tokenized_datasets[dataset])

tokenized_datasets.save_to_disk("data/tokenized_datasets/balanced")