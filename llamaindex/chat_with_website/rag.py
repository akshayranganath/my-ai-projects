from llama_index import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms import Bedrock
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.embeddings import BedrockEmbedding
import os

def load_data_from_website(url:str)->bool:
    return True

