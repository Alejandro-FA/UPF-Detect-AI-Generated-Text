import streamlit as st
from utils import plot_confusion_matrix

if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, 
        confusion_matrix,
        classification_report,
        f1_score,
    )

    DATA_FOLDER = 'data'
    MODEL_FOLDER = 'model'


    # %% Load predictions
    y_true = np.load(f'{DATA_FOLDER}/y_true.npy')
    y_pred = np.load(f'{DATA_FOLDER}/y_pred.npy')
    label2id = {"human": 0, "ai": 1}

    # %% Evaluate performance
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    st.write(f'Accuracy: {accuracy:.4f}')
    st.write(f'\nf1 score: {f1:.4f}')
    st.write('\nClassification report:')
    st.write(classification_report(y_true, y_pred, target_names=label2id.keys(), digits=4))
    st.pyplot( plot_confusion_matrix(cm, classes=label2id.keys(), figsize=(8, 6), is_norm=True) )