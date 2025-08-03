import streamlit as st

from app.settings import settings


def render_dynamic_title(page_name: str):
    """Render title with dynamic primary color from theme"""
    # Get primary color from Streamlit theme
    try:
        primary_color = st.get_option("theme.primaryColor") or "gold"
    except Exception:
        primary_color = "gold"  # fallback

    st.markdown(
        f"""
        <h1>
            <span style="color: {primary_color};">{settings.app_title}</span><span>/{page_name}</span>
        </h1>
        """,
        unsafe_allow_html=True,
    )
