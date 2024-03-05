import streamlit as st

# import utils and uiutils to re-use code.
from uiutils import build_page_structure, initialize
from utils import load_data_from_pdf

initialize()
build_page_structure("Upload your PDF", "📁")

st.subheader("Upload PDF")
pdf = st.file_uploader(
    label='Local PDF file', 
    type=['pdf'],
    accept_multiple_files=False,
    help="Local PDF file that you want to chat with.")

if st.button(label="Load Data"):
    with st.spinner("Loading data.."):
        result = load_data_from_pdf(
            pdf=pdf,
            bedrock_obj=st.session_state.bedrock_obj
            
        )
        if result==True:
            st.success(f"Loaded PDF successfully.")
        else:
            st.error("Unable to load the data.")