import torch
import matplotlib.pyplot as plt
import itertools
import numpy as np


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
    fig = plt.figure(figsize=figsize)
    
    # Display the confusion matrix as an image with a colormap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
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

    return fig