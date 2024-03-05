import uuid

from mybedrockobject import MyBedrockObject
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, Document
# libraries for website reader
from llama_index.readers.web import SimpleWebPageReader
# libraries for PDF reader
from PyPDF2 import PdfReader

# for self-documentation
from typing import List, Dict

def initialize_bedrock()->MyBedrockObject:
    bedrock = MyBedrockObject()
    return bedrock

## helper function to convert messages to chat messages
def convert_messages(messages:List)->List:
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

def load_data_from_url(url: str, bedrock_obj:MyBedrockObject) -> bool:
    # now create the uuid
    id = uuid.uuid4()
    # fetch the URL
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])    
    # load the documents to index
    index = VectorStoreIndex.from_documents(
        documents,        
        service_context=bedrock_obj.service_context,
        storage_context=bedrock_obj.storage_context        
    )    
    return True

def load_data_from_pdf(pdf, bedrock_obj:MyBedrockObject) -> bool:
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

    # load the documents to index
    index = VectorStoreIndex.from_documents(
        documents,        
        service_context=bedrock_obj.service_context,
        storage_context=bedrock_obj.storage_context,
        show_progress=True        
    )

    # index.storage_context.persist(persist_dir=)
    return True

def get_vector_store_index_based_chat_engine(bedrock_obj:MyBedrockObject) -> bool:
    index = VectorStoreIndex.from_vector_store(
        bedrock_obj.vector_store,
        service_context=bedrock_obj.service_context,
    )
    chat_engine = index.as_chat_engine(
            chat_mode="context", verbose=True
    )
    return chat_engine