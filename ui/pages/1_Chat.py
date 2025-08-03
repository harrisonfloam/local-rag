import streamlit as st

# Import functions for the chat page
from ui.components.chat import render_chat, render_chat_metrics
from ui.components.sidebar import render_sidebar
from ui.components.startup import run_startup_actions
from ui.utils.session_state import init_session_state
from ui.utils.title import render_dynamic_title

st.set_page_config(
    page_title="local-rag/chat",
    page_icon="💬",
    layout="wide",
)

render_dynamic_title("chat")

init_session_state()
run_startup_actions()

# Main layout with sidebar
render_sidebar()

# Main content
render_chat()
render_chat_metrics()
