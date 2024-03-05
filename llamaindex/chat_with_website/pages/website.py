import streamlit as st
from uiutils import build_page_structure, initialize
from utils import load_data_from_url 

initialize()
build_page_structure("Add your WebPage", "🌐")

st.subheader("Load URL")
url = st.text_input(label="URL")
if st.button(label="Load Data"):
    with st.spinner("Loading data.."):
        result = load_data_from_url(
            url=url,
            bedrock_obj=st.session_state.bedrock_obj        
        )
        if result == True:
            st.success(f"Loaded {url} successfully.")
        else:
            st.error("Unable to load the data.")