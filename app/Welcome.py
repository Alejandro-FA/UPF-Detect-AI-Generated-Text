
if __name__ == '__main__':
    import streamlit as st

    st.set_page_config(
        page_title="Welcome",
        page_icon="👋",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': 'https://github.com/Alejandro-FA/UPF-Detect-AI-Generated-Text',
            'Report a bug': 'https://github.com/Alejandro-FA/UPF-Detect-AI-Generated-Text/issues',
            'About': None,
        }
    )

    st.sidebar.header("Navigate through the different tabs to learn about all the features of this app")
    st.sidebar.write(" 📢 In the welcome tab you will find a brief introduction to the project: Detection of AI-generated texts.")

    # The title
    st.title("Detection of AI-generated texts 🕵️")

    # The subheader
    st.subheader("Introduction to the use case")

    # The text
    st.markdown(
        """
        The goal of the project is to build a classification model that is
        capable of accurately detecting whether a text has been generated
        by a LLM or by a student. The purpose of the model is to improve
        plagiarism detection tools in this new learning context defined by
        AI. Since we don't expect to get great accuracy results (especially
        in such short time), we will put a lot of emphasis in explainable
        AI. The idea is to build a tool capable of detecting potential
        plagiarism candidates, and then leave the final call to a human
        (normally a professor). To help this decision, information about
        why the model has predicted plagiarism (e.g. using SHAP) will be
        given to the professor.
        """
    )