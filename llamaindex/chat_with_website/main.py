import streamlit as st

st.set_page_config(
    page_title="Chat with your WebPage",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

with st.sidebar:
    st.page_link("main.py", label="Home", icon="🏠")
    st.page_link("pages/pdf.py", label="PDF", icon="📁")
    st.page_link("pages/website.py", label="Website", icon="🌐")
    st.page_link("pages/chat.py", label="Chat", icon="💬")

st.title('🔎 RAG Demos 🔎')
st.caption('Chat with your websites and documents using Amazon Bedrock.')

st.subheader('Objective')
st.write('The purpose of these webpages is to demonstrate the concept of retrieval augmented generation (RAG). This website supports scraping a webpage from a URL or reading a PDF document. Users can then chat about this website or PDF doc.')
st.write("""Currently, there are 2 options available:
1. Chatting with a website by providing a URL.
2. Chatting with a PDF by uploading the document.

Choose the option that works best for you and let me know your feedback!         
""")

st.warning('This code requires you to login to your Amazon Solutions account. Check and edit the file `.env` before running the application. ')