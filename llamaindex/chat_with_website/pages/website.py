import streamlit as st
from dotenv import load_dotenv
import os
import uuid

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.bedrock import Bedrock
from llama_index.core import ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings.bedrock import BedrockEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# enabling debug
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

# exceptions
from botocore.exceptions import ClientError

load_dotenv()

# initialize the service and storage context
debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[debug])
llm = Bedrock(
    model=os.environ["BEDROCK_MODEL"],
    temperature=os.environ["CHAT_TEMPERATURE"],
    profile_name=os.environ["AWS_PROFILE_NAME"],
)
embedding = BedrockEmbedding.from_credentials(
    aws_profile=os.environ["AWS_PROFILE_NAME"]
)
# create a service context using the Bedrock embeddings
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embedding  # , callback_manager=callback_manager
)

dbconn = chromadb.Client(
    chromadb.config.Settings(
        is_persistent=True, persist_directory=os.environ["PERSIST_DIR"]
    )
)
chroma_collection = dbconn.get_or_create_collection(os.environ["COLLECTION_NAME"])
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


def load_data(url: str) -> bool:
    # now create the uuid
    id = uuid.uuid4()
    # fetch the URL
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,        
        service_context=service_context,
        storage_context=storage_context        
    )

    # index.storage_context.persist(persist_dir=)
    return True


st.set_page_config(
    page_title="Chat with your WebPage",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# add a sidebar for chatting with PDF
with st.sidebar:
    st.page_link("main.py", label="Home", icon="🏠")
    st.page_link("pages/pdf.py", label="PDF", icon="📁")
    st.page_link("pages/website.py", label="Website", icon="🌐")
    st.page_link("pages/chat.py", label="Chat", icon="💬")
    

st.title("🦙 Chat with WebPage 🦙")

tab_load, tab_chat = st.tabs(["Load URL", "Chat"])


st.subheader("Load URL")
url = st.text_input(label="URL")
if st.button(label="Load Data"):
    with st.spinner("Loading data.."):
        if load_data(url=url) == True:
            st.write(f"Loaded {url} successfully.")
        else:
            st.error("Unable to load the data.")


st.subheader("Chat")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the website that was just loaded..",
        }
    ]

# check and initialize index
if "chat_engine" not in st.session_state.keys():
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True
    )

if prompt := st.chat_input("Your question: "):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # call the chat engine with user prompt.. variable prompt is defined earlier


# if the last message is from a user, now, run the llm
if st.session_state.messages[-1]["role"] != "assistant":
    try:
        response = st.session_state.chat_engine.chat(message=prompt)
        st.write(response.response)

        # add the message into the message history
        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )
    except ClientError as e:
        st.error(f"ClientError: {e}")
