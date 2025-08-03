import streamlit as st

# Import functions for the chat page
from ui.components.chat import render_chat, render_chat_metrics
from ui.components.sidebar import fetch_models, render_sidebar
from ui.utils.session_state import init_session_state
from ui.utils.title import render_dynamic_title

st.set_page_config(
    page_title="local-rag",
    page_icon="💬",
    layout="wide",
)

# Dynamic title with themed colors
render_dynamic_title("chat")

init_session_state()

# Initialize models in sidebar
fetch_models()

# Main layout with sidebar
render_sidebar()

# Main content
render_chat()
render_chat_metrics()
