import streamlit as st

chat = st.Page("pages/1_Chat.py", title="Chat", icon="💬", default=True)
documents = st.Page("pages/2_Documents.py", title="Documents", icon="📁")

pg = st.navigation([chat, documents])
pg.run()
