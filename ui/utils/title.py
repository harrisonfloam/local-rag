import streamlit as st

from app.settings import settings


def render_dynamic_title(page_name: str):
    """Render title with dynamic primary color from theme"""
    primary_color = st.get_option("theme.primaryColor") or "gold"

    st.markdown(
        f"""
        <h1>
            <span style="color: {primary_color};">{settings.app_title}</span><span style="color: {primary_color}; opacity: 0.6;">/</span><span>{page_name}</span>
        </h1>
        """,
        unsafe_allow_html=True,
    )
