# %% Auxiliary methods and imports
import torch

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

if __name__ == '__main__':
    from transformers import pipeline
    import os
    import shap
    import pandas as pd
    import datasets

    # %% Load Huggingface model and data
    model_path = os.path.abspath('model_pretrained')
    classifier = pipeline(
        'text-classification',
        model=model_path,
        device=get_torch_device(debug=True),
        top_k=None,
        batch_size=16,
    )

    data = datasets.load_from_disk("data/tokenized_datasets/balanced")['test']
    short_data = [v[:500] for v in data["text"][:20]]

    # %% Create a SHAP explainer and explor shap values
    explainer = shap.Explainer(classifier)
    shap_values = explainer(short_data)
    shap.plots.text(shap_values)


    # %% Make a prediction

    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

    print(classifier(text))