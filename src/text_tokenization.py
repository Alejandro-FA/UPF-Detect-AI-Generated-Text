# https://huggingface.co/docs/transformers/tasks/sequence_classification

# %% Imports and auxiliary functions
import pandas as pd
import datasets

def preprocess_function(data):
    return tokenizer(data['text'], truncation=True)

def to_dataset(texts, labels):
    assert(len(texts) == len(labels))
    output = {
        "text": list(map(lambda x: str(x).strip(), texts)),
        "label": labels,
    }
    return datasets.Dataset.from_dict(output)

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



if __name__ == '__main__':
    from transformers import AutoTokenizer
    from sklearn.model_selection import train_test_split

    # %% Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # %% Create pandas dataframe from CSV file
    # df = balance(pd.read_csv("data/train_essays.csv"))
    # assert(df[df["generated"]==0].count().generated == df[df["generated"]==1].count().generated)

    # %% Create dataset from dataframe and separate into train-test split
    # X = df['text'].values
    # y = df['generated'].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # my_datasets = datasets.DatasetDict({
    #     "train": to_dataset(X_train, y_train),
    #     "test": to_dataset(X_test, y_test)
    # })
    df_train = pd.read_csv('data/archive/final_train.csv')
    df_train.info()

    df_test = pd.read_csv('data/archive/final_test.csv')
    df_test.info()

    my_datasets = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(df_train),
        "test": datasets.Dataset.from_pandas(df_test)
    })

    # %% Tokenize dataset
    print("Tokenizing...")
    tokenized_datasets = my_datasets.map(preprocess_function, batched=True)
    print("Tokenization ended")

    for dataset in tokenized_datasets:
        print_stats(dataset, tokenized_datasets[dataset])

    tokenized_datasets.save_to_disk("data/tokenized_datasets/new")