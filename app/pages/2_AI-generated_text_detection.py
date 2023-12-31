import shap
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline, TextClassificationPipeline
from utils import get_torch_device
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Any


primary_color = 'red'
secondary_color = 'blue'


@dataclass(repr=False, eq=False, frozen=True)
class Model:
    name: str
    pipeline: TextClassificationPipeline
    explainer: shap.Explainer
    labels: dict[int, str]


@st.cache_resource
def load_models() -> dict[str, Model]:
    device = get_torch_device(debug=True)
    our_model = pipeline(
        task='text-classification',
        model='Alejandro-FA/ma_ai_text',
        device=device,
        top_k=1,
        truncation=True,
        padding=True,
    )
    simpleai_model = pipeline(
        task="text-classification",
        model="Hello-SimpleAI/chatgpt-detector-roberta",
        device=device,
        top_k=1,
        truncation=True,
        padding=True,
    )
    return {
        'Our model': Model(
            'Our model',
            our_model,
            shap.Explainer(our_model),
            {0:'human', 1:'ai'}
        ),
        'SimpleAI ChatGPT Detector': Model(
            'SimpleAI ChatGPT Detector',
            simpleai_model,
            shap.Explainer(simpleai_model),
            {0:'Human', 1:'ChatGPT'}
        ),
    }


@st.cache_data(
    show_spinner="Running model to check if this text has been generated by an AI...",
    hash_funcs={Model: lambda x: hash(x.name)}
)
def _predict_aux(text: str, model: Model) -> tuple[Any, Any]:
    return model.pipeline(text), model.explainer([text])


def compute_prediction(text: str, model: Model) -> None:
    st.session_state['text_input'] = text
    st.session_state['model_selection'] = model.name
    pred, shap_values = _predict_aux(text, model)
    st.session_state['prediction'] = pred
    st.session_state['shap_values'] = shap_values
    st.toast('Prediction completed!', icon="✅")


def init_state() -> None:
    # Initialize session state
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'shap_values' not in st.session_state:
        st.session_state['shap_values'] = None
    if 'text_input' not in st.session_state:
        st.session_state['text_input'] = None
    if 'model_selected' not in st.session_state:
        st.session_state['model_selected'] = None


def reset_state(model_selected: int) -> None:
    st.session_state['model_selected'] = model_selected
    st.session_state['prediction'] = None
    st.session_state['shap_values'] = None


def input_text_widget(model: Model) -> str:
    return st.text_area(
        label="Enter text to check",
        value=st.session_state['text_input'],
        max_chars=2500,
        height=300,
        placeholder="Copy the text that you want to check here.",
        label_visibility="collapsed",
        on_change=lambda: compute_prediction(
            st.session_state['text_area'],
            model
        ),
        key='text_area',
    )


@st.cache_data()
def get_shap_plots(prediction, _shap_values: shap._explanation.Explanation) -> tuple[str, Figure]:
    html = shap.plots.text(_shap_values[:, :, prediction[0][0]['label']], display=False)
    shap.plots.bar(_shap_values[0, :, prediction[0][0]['label']], show=False)
    return html, plt.gcf()



if __name__ == '__main__':
    st.set_page_config(
        page_title="AI-generated text detection",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )

    ############################ SECTION 00 ###################################
    st.title("Check whether a text has been generated by an AI or not 🤖")
    st.write('\n')
    st.markdown(
        """
        You can make a prediction with either our own model or with the model
        developed by [SimpleAI](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)¹. If you want more information about our model, we recommend to check
        the [training dataset analysis](/Corpus_analysis) page
        and the [model performance](/Model_evaluation) page.
        """
    )
    st.caption(
        """
        ¹Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., ... & Wu,
        Y. (2023). How close is chatgpt to human experts? comparison corpus,
        evaluation, and detection. arXiv preprint arXiv:2301.07597
        """
    )
    
    ############################ SECTION 01 ###################################
    # Load AI-generatet text classification model and SHAP explainers
    init_state()
    models = load_models()
    options = ('Our model', 'SimpleAI ChatGPT Detector')

    # Select which model to use
    model_name = st.selectbox(
        label=f':{primary_color}[Which model would you like to use?]',
        placeholder='Select an AI-generated text detection model',
        options=options,
        index=st.session_state['model_selected'],
        on_change=lambda: reset_state(options.index(st.session_state['selectbox'])),
        label_visibility='visible',
        key='selectbox',
    )

    ############################ SECTION 02 ###################################
    # Text area to input text to classify
    if model_name is not None:
        txt = input_text_widget(models[model_name])

        st.write('\n')
        col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
        if col2.button('Check text', type='primary', use_container_width=True):
            compute_prediction(txt, models[model_name])

    ############################ SECTION 03 ###################################
    prediction = st.session_state['prediction']
    shap_values = st.session_state['shap_values']

    if prediction is not None and shap_values is not None:
        st.divider()

        labels = models[model_name].labels
        prob = prediction[0][0]['score']
        if prediction[0][0]['label'] == labels[0]: prob = 1 - prob

        color = 'red' if prob > 0.5 else 'blue'
        prob_str = str(round(prob * 100, 1)) + '%'

        st.info(
            f"""
            ##### There is a :{color}[{prob_str}] chance that this text has been generated by an AI model
            """,
            icon='🤖'
        )
        st.write('\n')

        # SHAP plots
        html, fig = get_shap_plots(prediction, shap_values)
        st.subheader('Important elements of the text')
        st.markdown(
            """
            In this section you can get some **insights** about which words and
            sentences have contributed the most for the model to make this
            prediction.
            Red elements are perceived by the model as "probably AI-generated",
            while blue elements are perceived as "probably human-generated".
            """
        )
        height = 160 + 20*len(st.session_state['text_input']) / 100
        components.html(html, scrolling=True, height=int(height))

        st.write('\n')

        st.pyplot(fig)
        plt.cla()
        plt.close(fig)

        
