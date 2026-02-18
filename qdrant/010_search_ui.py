import streamlit as st
import requests

st.title("📰 Search Blogs")

st.caption("Search through the blogs using the title, description or tags. The search is powered by Qdrant and Sentence Transformers.")
search_text = st.text_input(label="Search Text")
if st.button(label="Search") and search_text:
    with st.spinner(text="Searching.."):
        resp = requests.get(f"http://0.0.0.0:8000/search?q={search_text}")
        
        for res in resp.json()['result']:
            
            st.subheader(res['title'])
            #st.write(res['description'])
            st.write(res['text'])
            st.write(f"Tags: {res['tags']}")
            if res['image']:
                st.image(res['image'], width=400)
            print(res)