# %% Imports and helper functions
from utils import get_torch_device, plot_confusion_matrix

if __name__ == '__main__':
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    import datasets
    import os
    from tqdm.auto import tqdm
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, 
        confusion_matrix,
        classification_report,
        f1_score,
    )

    DATA_FOLDER = 'data'
    MODEL_FOLDER = 'model'

    # %% Load AI-generatet text classification model
    model_path = os.path.abspath(MODEL_FOLDER)
    classifier = pipeline(
        'text-classification',
        model=model_path,
        device=get_torch_device(debug=True),
        truncation=True,
        padding=True,
        top_k=1,
    )

    # %% Load test dataset
    tokenized_datasets = datasets.load_from_disk(f'{DATA_FOLDER}/tokenized_datasets/new')
    data = KeyDataset(tokenized_datasets['test'], 'text')
    id2label = {0: "human", 1: "ai"}
    label2id = {"human": 0, "ai": 1}

    # %% Load or create predictions
    try:
        y_true = np.load(f'{DATA_FOLDER}/y_true.npy')
        y_pred = np.load(f'{DATA_FOLDER}/y_pred.npy')
    except:
        outputs = classifier(data, batch_size=32)
        y_true = np.array( [x['label'] for x in tokenized_datasets['test']], dtype=np.bool_ )
        y_pred = np.zeros(len(data), dtype=np.bool_)

        for i, out in tqdm(enumerate(outputs), total=len(data)):
            y_pred[i] = label2id[out[0]['label']]
            i += 1

        np.save(f'{DATA_FOLDER}/y_true.npy', y_true)
        np.save(f'{DATA_FOLDER}/y_pred.npy', y_pred)


    # %% Evaluate performance
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    print(f'\nAccuracy: {accuracy:.4f}')
    print(f'\nf1 score: {f1:.4f}')
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=label2id.keys(), digits=4))

    plot_confusion_matrix(cm, classes=label2id.keys(), figsize=(8, 6), is_norm=True)
