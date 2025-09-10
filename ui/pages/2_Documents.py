import streamlit as st

from ui.components.document_portal import render_document_portal
from ui.components.startup import run_startup_actions
from ui.utils.exceptions import handle_streamlit_exceptions
from ui.utils.title import render_dynamic_title

st.set_page_config(page_title="local-rag/documents", page_icon="📁", layout="wide")


@handle_streamlit_exceptions
def main():
    """Main function for the Documents page."""
    render_dynamic_title("documents")
    run_startup_actions()
    render_document_portal()


if __name__ == "__main__":
    main()
