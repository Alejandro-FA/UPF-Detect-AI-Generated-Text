# https://huggingface.co/docs/transformers/tasks/sequence_classification

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "human", 1: "ai"}
label2id = {"ai": 0, "human": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id
)


df = pd.read_csv("data/train_essays.csv").head(10)

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

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Training...")
trainer.train()