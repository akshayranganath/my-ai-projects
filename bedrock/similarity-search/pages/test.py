import streamlit as st
import sys

sys.path.insert(1, "./lib/")
from chroma_utils import DBObject
from image_utils import download_and_save_image, delete_image

st.title("Test: Search Image in Index")

url = st.text_input(label="Cloudinary Image URL", help="Cloudinary Image URL")

st.header("Search Result")
if st.button("Search"):
    with st.spinner("Downloading image.."):
        img_name = download_and_save_image(url)
        st.image(img_name, width=250, caption="Uploaded Image")
    st.divider()
    with st.spinner("Search image in index.."):
        if "db" not in st.session_state:
            st.session_state.db = DBObject()
        results = st.session_state.db.query_object_in_collection(img_name)
        # print(results)
        ids, distances, metadatas = (
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
        results = 0
        for i in range(len(ids)):
            if distances[i] > 0.5:
                break
            st.image(
                metadatas[i]["url"], width=250, caption=f"Distance: {distances[i]}"
            )
            results += 1
            # st.write(f"ID: {ids[i]}, Distance: {distances[i]}, Metadata: {metadatas[i]}")
        if results == 0:
            st.success("No results found")
        else:
            st.success(f"{results} results found")
    
    # clean up image
    delete_image(img_name)

