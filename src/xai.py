# %% Auxiliary methods and imports
from utils import get_torch_device


if __name__ == '__main__':
    from transformers import pipeline
    import os
    import shap
    import datasets

    # %% Load Huggingface model and data
    model_path = os.path.abspath('../model')
    classifier = pipeline(
        'text-classification',
        model=model_path,
        device=get_torch_device(debug=True),
        top_k=None,
        batch_size=16,
        truncation=True,
        padding=True,
    )
    data = datasets.load_from_disk("../data/tokenized_datasets/new")['test']

    # %% Create a SHAP explainer and explor shap values
    explainer = shap.Explainer(classifier)
    short_data = [v[:500] for v in data["text"][:20]]
    shap_values = explainer(short_data)
    shap.plots.text(shap_values)

    # %% Make a prediction
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

    text2 = "I liked the book a lot, it captures your attention from the very first moment and you can't stop reading page after page. The story is profound and beautiful, so I am sure that it won't leave any reader unmoved."

    print(classifier(text2))

    # %%
    print(data[0]['text'])


# %%
