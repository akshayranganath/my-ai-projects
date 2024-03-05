import streamlit as st
from utils import get_vector_store_index_based_chat_engine, get_service_context_and_llm, get_vector_store
from botocore.exceptions import ClientError

def initialize():
    if "vector_store" not in st.session_state.keys():
        # setup LLM and service context for Bedrock
        st.session_state.service_context,st.session_state.llm = get_service_context_and_llm()
        # get the vector store
        st.session_state.vector_store = get_vector_store()


def build_page_structure(title:str, icon:str)->None:
    st.set_page_config(
    page_title=title,
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
    )
    
    st.title(f"{icon} {title} {icon}")

    # add a sidebar for chatting with PDF
    with st.sidebar:
        st.page_link("main.py", label="Home", icon="🏠")
        st.page_link("pages/pdf.py", label="PDF", icon="📁")
        st.page_link("pages/website.py", label="Website", icon="🌐")
        st.page_link("pages/chat.py", label="Chat", icon="💬")

def handle_chat(clear_session:bool=True)->None:
    st.subheader("Chat")
    if "messages" not in st.session_state.keys() or clear_session==True:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about the website that was just loaded..",
            }
        ]
    
    # check and initialize index
    if "chat_engine" not in st.session_state.keys():        
        st.session_state.chat_engine = get_vector_store_index_based_chat_engine(
            vector_store=st.session_state.vector_store, 
            service_context=st.session_state.service_context
        )

    if prompt := st.chat_input("Your question: "):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])            

    # if the last message is from a user, now, run the llm
    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner('Waiting for response from GPT...'):
            try:
                response = st.session_state.chat_engine.chat(message=prompt)
                st.write(response.response)

                # add the message into the message history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.response}
                )
            except ClientError as e:
                st.error(f"ClientError: {e}")