from dotenv import load_dotenv
import os

# import vector store
from llama_index.vector_stores import PineconeVectorStore
from pinecone import Pinecone
# load the PineCone vector store
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore

# enabling debug
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index import ServiceContext

load_dotenv()

# initialize pinecone
pinecone = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)


# pinecode
index_name = 'llamaindex-helper'
pinecone_index = pinecone.Index(index_name)


# since the environment variale is openai key, it will use openapi
debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)


query = 'What are the different pieces of advise given about owning and buying real-estate?'
query_engine = index.as_query_engine()
response = query_engine.query(query)
print(response)

