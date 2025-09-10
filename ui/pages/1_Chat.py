import streamlit as st

# Import functions for the chat page
from ui.components.chat import render_chat, render_chat_metrics
from ui.components.sidebar import render_sidebar
from ui.components.startup import run_startup_actions
from ui.utils.title import render_dynamic_title
from ui.utils.exceptions import handle_streamlit_exceptions

st.set_page_config(
    page_title="local-rag/chat",
    page_icon="💬",
    layout="wide",
)

@handle_streamlit_exceptions
def main():
    """Main function for the Chat page."""
    render_dynamic_title("chat")
    run_startup_actions()
    # Main layout with sidebar
    render_sidebar()
    # Main content
    render_chat()
    render_chat_metrics()

if __name__ == "__main__":
    main()
else:
    # When called from streamlit navigation, run directly
    main()
