import streamlit as st

st.title("Visually Similar Search")
st.caption("Demo for using Multimodal Embedding with ChromaDB")

st.write(
    """Welcome to this demo for Multimodal Embedding. The purpose of this project is to demonstrate the use of MultiModal embedding for searching similar images in a data set. For this demo, we'll be using `ChromaDB` as the database. ChromaDB has an implementation of the `clip` embedding as `OpenCLIPEmbeddingFunction`. We'll be using this function to embed the images.

Search will use `l2`, the default algorithm of ChromaDB. This is basically the Eucleadean disatance. Each search will look for 3 nearest neighbors.
"""
)
