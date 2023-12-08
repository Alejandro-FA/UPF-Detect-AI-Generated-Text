# %% Imports and helper functions
import torch
import matplotlib.pyplot as plt
import itertools

def get_torch_device(use_gpu: bool = True, debug: bool = False) -> torch.device:
    """Obtains a torch device in which to perform computations

    Args:
        use_gpu (bool, optional): Use GPU if available. Defaults to True.
        debug (bool, optional): Whether to print debug information or not. Defaults to False.

    Returns:
        torch.device: Device in which to perform computations
    """    
    device = torch.device(
        'cuda:0' if use_gpu and torch.cuda.is_available() else
        'mps' if use_gpu and torch.backends.mps.is_available() else
        'cpu'
    )
    if debug: print("Device selected:", device)
    return device

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8), is_norm=True):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        title (str): Title for the plot.
        cmap (matplotlib colormap): Colormap for the plot.
    """
    # Create a figure with a specified size
    plt.figure(figsize=figsize)
    
    # Display the confusion matrix as an image with a colormap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Define tick marks and labels for the classes on the axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    if is_norm:
        fmt = '.3f'
    else:
        fmt = '.0f'
    # Add text annotations to the plot indicating the values in the cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # Label the axes
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Ensure the plot layout is tight
    plt.tight_layout()
    # Display the plot
    plt.show()


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

    DATA_FOLDER = '../data'
    MODEL_FOLDER = '../model'

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

    print(f'Accuracy: {accuracy:.4f}')
    print(f'f1 score: {f1:.4f}')
    plot_confusion_matrix(cm, classes=label2id.keys(), figsize=(8, 6), is_norm=True)
    print('Classification report:')
    print()
    print(classification_report(y_true, y_pred, target_names=label2id.keys(), digits=4))
    
