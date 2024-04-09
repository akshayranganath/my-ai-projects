import streamlit as st
import sys

sys.path.insert(1, "./lib/")
from chroma_utils import DBObject
from image_utils import download_and_save_image, delete_image

st.title("Train: Add Images to Index")

url = st.text_input(label="Cloudinary Image URL", help="Cloudinary Image URL")

if st.button("Add Image"):
    with st.spinner("Downloading image.."):
        img_name = download_and_save_image(url)
        st.image(img_name, width=250, caption="Uploaded Image")
    with st.spinner("Adding image to index.."):
        if "db" not in st.session_state:
            st.session_state.db = DBObject()
        if st.session_state.db.add_object_to_collection(file_name=img_name, url=url):
            st.success("Image added to index")
        else:
            st.error("Error adding image to index")
    # delete image
    delete_image(img_name)
