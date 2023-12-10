# %% Imports and auxiliary functions

import evaluate
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    from transformers import (
        AutoModelForSequenceClassification, 
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        AutoTokenizer
    )
    import datasets
    import torch

    # %% Training
    print("CUDA available:", torch.cuda.is_available())

    tokenized_datasets = datasets.load_from_disk("data/tokenized_datasets/new")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "human", 1: "ai"}
    label2id = {"human": 0, "ai": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="model/training_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        use_cpu=False, # Uses CUDA or mps if available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_dir='model')
    # trainer.push_to_hub()
