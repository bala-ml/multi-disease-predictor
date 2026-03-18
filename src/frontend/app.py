import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="Multi-Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
* {
    font-family: 'Poppins', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠Multi-Disease Predictor")

st.write(
    """
Use the left sidebar to navigate:
- 🩺 Diabetes Risk Predictor
- ❤️ Cardio Risk Predictor
"""
)

# st.info("Make sure the FastAPI backend is running before using predictions.")