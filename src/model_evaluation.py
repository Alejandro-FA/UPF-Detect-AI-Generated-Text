# %% Imports and helper functions
from utils import get_torch_device, plot_confusion_matrix


if __name__ == '__main__':
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    import datasets
    import os
    from tqdm.auto import tqdm
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, 
        confusion_matrix,
        classification_report,
        f1_score,
    )

    DATA_FOLDER = '../data'
    MODEL_FOLDER = '../model'

    # %% Load AI-generatet text classification models
    model_path = os.path.abspath(MODEL_FOLDER)
    classifier_ours = pipeline(
        task='text-classification',
        model=model_path,
        device=get_torch_device(debug=True),
        truncation=True,
        padding=True,
        top_k=1,
    )

    classifier_roberta = pipeline(
        task="text-classification",
        model="Hello-SimpleAI/chatgpt-detector-roberta",
        device=get_torch_device(debug=True),
        truncation=True,
        padding=True,
        top_k=1,
    )

    classifiers = [
        {
            'classifier': classifier_ours,
            'label2id': {"human": 0, "ai": 1},
            'name': 'ours',
        },
        {
            'classifier': classifier_roberta,
            'label2id': {"Human": 0, "ChatGPT": 1},
            'name': 'roberta',
        },
    ]


    # %% Create validation dataset
    human = (
        pd.read_json(f'{DATA_FOLDER}/CHEAT/ieee-init.jsonl', lines=True)
        .drop(columns=['id', 'title', 'keyword'])
        .rename(columns={'abstract': 'text'})
        .sample(n=750, random_state=42)
    )
    human['label'] = 0
    human['source'] = 'CHEAT'

    ai = (
        pd.read_json(f'{DATA_FOLDER}/CHEAT/ieee-chatgpt-generation.jsonl', lines=True)
        .drop(columns=['id', 'title', 'keyword'])
        .rename(columns={'abstract': 'text'})
        .sample(n=750, random_state=42)
    )
    ai['label'] = 1
    ai['source'] = 'CHEAT'

    mgt = pd.read_csv(f'{DATA_FOLDER}/MGTBench/TruthfulQA_LLMS.csv').dropna()
    mgt['source'] = 'MGTBench'
    human2 = pd.DataFrame({'text': mgt['Best Answer'], 'label': 0, 'source': mgt['source']})
    ai2 = pd.DataFrame({'text': mgt['ChatGPT-turbo_answer'], 'label': 1, 'source': mgt['source']})

    df = pd.concat([human, ai, human2, ai2]).reset_index(drop=True)
    df['source'] = df['source'].astype('category')
    df.to_csv(f'{DATA_FOLDER}/test_dataset.csv', index=False)
    df.info()
    df.head()


    # %% Create Huggingface dataset
    dataset = datasets.Dataset.from_pandas(df)
    key_dataset = KeyDataset(dataset, 'text')


    # %% Load or create predictions
    all_y_true = [None] * len(classifiers)
    all_y_pred = [None] * len(classifiers)

    for i, pipe in enumerate(classifiers):
        classifier = pipe['classifier']
        name = pipe['name']
        label2id = pipe['label2id']

        try:
            y_true = np.load(f'{DATA_FOLDER}/y_true_{name}.npy')
            y_pred = np.load(f'{DATA_FOLDER}/y_pred_{name}.npy')
        except:
            outputs = classifier(key_dataset, batch_size=32)
            y_true = np.array( [x['label'] for x in dataset], dtype=np.bool_ )
            y_pred = np.zeros(len(dataset), dtype=np.bool_)

            for j, out in tqdm(enumerate(outputs), total=len(dataset)):
                y_pred[j] = label2id[out[0]['label']]
                j += 1

            np.save(f'{DATA_FOLDER}/y_true_{name}.npy', y_true)
            np.save(f'{DATA_FOLDER}/y_pred_{name}.npy', y_pred)
        finally:
            all_y_true[i] = y_true
            all_y_pred[i] = y_pred


    # %% Evaluate performance
    for i, pipe in enumerate(classifiers):
        print(f'Performance of model: {pipe["name"]}')

        accuracy = accuracy_score(all_y_true[i], all_y_pred[i])
        f1 = f1_score(all_y_true[i], all_y_pred[i], average='macro')
        cm = confusion_matrix(all_y_true[i], all_y_pred[i], normalize='true')

        print(f'\nAccuracy: {accuracy:.4f}')
        print(f'\nf1 score: {f1:.4f}')
        print('\nClassification report:')
        print(classification_report(all_y_true[i], all_y_pred[i], target_names=label2id.keys(), digits=4))

        plot_confusion_matrix(cm, classes=label2id.keys(), figsize=(8, 6), is_norm=True)
    
