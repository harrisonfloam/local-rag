import streamlit as st

from ui.components.chat import render_chat, render_chat_metrics
from ui.components.sidebar import fetch_models, render_sidebar
from ui.utils.session_state import init_session_state


def main():
    st.set_page_config(page_title="local-rag", layout="wide")
    st.title("local-rag")

    init_session_state()
    fetch_models()

    render_sidebar()

    render_chat()
    render_chat_metrics()


if __name__ == "__main__":
    main()
