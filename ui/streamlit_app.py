import streamlit as st

chat = st.Page("pages/1_chat.py", default=True)
documents = st.Page("pages/2_documents.py")

pg = st.navigation([chat, documents])
pg.run()
