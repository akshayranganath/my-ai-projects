import streamlit as st
# import from utils
from utils.bedrock import get_text_embedding, create_session
from utils.opesearch import create_opensearch_client, vector_search

st.title("Re-Imagined Image Search")
st.write("This is a demo of an enhanced visual search application. To get started, enter a few paragraphps about the article you are planning to create.")

article_text = st.text_area("Article Text")

if st.button("Submit"):
    if article_text:
        with st.spinner('Finding a match for your article...'):                        
            # create a bedrock session, if it does not exist in the session state
            if 'bedrock' not in st.session_state:
                session, runtime = create_session()
                st.session_state.session = session
                st.session_state.bedrock = runtime
            
            # get the text embedding
            text_embedding = get_text_embedding(st.session_state.bedrock, article_text)
            
            
            # create an opensearch client, if it does not exist in the session state
            if 'opensearch' not in st.session_state:
                st.session_state.opensearch = create_opensearch_client(st.session_state.session)
            
            # perform the vector search
            search_results = vector_search(st.session_state.opensearch, text_embedding)

            # display the images from the search results
            for result in search_results:
                img_url = result['_source'].get('url', '')
                st.image(img_url)   
    else:
        st.write("Please enter the article text.")
