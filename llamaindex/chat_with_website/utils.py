from dotenv import load_dotenv
import os
import uuid
import boto3

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.bedrock import Bedrock
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex, Document
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
# enabling debug
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
# libraries for website reader
from llama_index.readers.web import SimpleWebPageReader
# library for database
import chromadb
# libraries for PDF reader
from PyPDF2 import PdfReader



# exceptions
from botocore.exceptions import ClientError

load_dotenv()
## helper function to convert messages to chat messages
def convert_messages(messages):
    converted_messages = []
    for message in messages:
        role = message['role']
        content = message['content']
        if role == "assisstant":
            role = MessageRole.ASSISTANT
        elif role=="system":
            role = MessageRole.SYSTEM
        else:
            role = MessageRole.USER
        converted_messages.append(ChatMessage(role = role, content = content))
    return converted_messages

def get_service_context_and_llm():
    # initialize the service and storage context
    debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[debug])
    # Setup bedrock
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    llm = Bedrock(
        model=os.environ["BEDROCK_MODEL"],
        temperature=os.environ["CHAT_TEMPERATURE"],
        client=bedrock_runtime,
    )
    embedding = BedrockEmbedding(
        client=bedrock_runtime,
        model=os.environ['BEDROCK_EMBED_MODEL']
    )

    # to read on EC2/AWS instances, use these methods instead
    # https://medium.com/@dminhk/llamaindex-q-a-over-your-data-using-amazon-bedrock-and-streamlit-1e52a096ec7c
    # create a service context using the Bedrock embeddings
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embedding  # , callback_manager=callback_manager
    )
    return (service_context,llm)

def get_vector_store():
    dbconn = chromadb.Client(
        chromadb.config.Settings(
        is_persistent=True, persist_directory=os.environ["PERSIST_DIR"]
        )
    )
    chroma_collection = dbconn.get_or_create_collection(os.environ["COLLECTION_NAME"])
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def load_data_from_url(url: str, vector_store: VectorStoreIndex, service_context:ServiceContext) -> bool:
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

def load_data_from_pdf(pdf, vector_store: VectorStoreIndex, service_context:ServiceContext) -> bool:
    # now create the uuid
    id = uuid.uuid4()

    # first write the file to disk
    bytes_data = pdf.getvalue()
    #with open('./temp.pdf','wb') as f:
    #    f.write(bytes_data)
    reader = PdfReader(pdf)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    document = Document(text=text)
    # fetch the URL
    #loader = PyMuPDFReader()
    documents = [document]
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,        
        service_context=service_context,
        storage_context=storage_context,
        show_progress=True        
    )

    # index.storage_context.persist(persist_dir=)
    return True

def get_vector_store_index_based_chat_engine(vector_store: VectorStoreIndex, service_context:ServiceContext) -> bool:
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )
    chat_engine = index.as_chat_engine(
            chat_mode="context", verbose=True
    )
    return chat_engine