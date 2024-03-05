import streamlit as st
from uiutils import build_page_structure

build_page_structure("RAG Demos", "🏠")

st.caption('Chat with your websites and documents using Amazon Bedrock.')

st.subheader('Objective')
st.write('The purpose of these webpages is to demonstrate the concept of retrieval augmented generation (RAG). This website supports scraping a webpage from a URL or reading a PDF document. Users can then chat about this website or PDF doc.')
st.write("""Currently, there are 2 options available:
1. Chatting with a website by providing a URL.
2. Chatting with a PDF by uploading the document.

Choose the option that works best for you and let me know your feedback!         
""")

st.warning('This code requires you to login to your Amazon Solutions account. Check and edit the file `.env` before running the application. ')