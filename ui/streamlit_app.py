import streamlit as st

from app.utils.utils import init_logging

init_logging()

chat = st.Page("pages/1_chat.py", default=True)
documents = st.Page("pages/2_documents.py")

pg = st.navigation([chat, documents])
pg.run()
