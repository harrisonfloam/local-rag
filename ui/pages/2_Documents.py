import streamlit as st

from ui.components.document_portal import (
    render_document_browser,
    render_upload_interface,
)
from ui.utils.title import render_dynamic_title

st.set_page_config(page_title="📁 Documents", page_icon="📁", layout="wide")

# Dynamic title with themed colors
render_dynamic_title("documents")

tab1, tab2 = st.tabs(["🔍 Browse & Manage", "📤 Upload"])

with tab1:
    render_document_browser()

with tab2:
    render_upload_interface()
