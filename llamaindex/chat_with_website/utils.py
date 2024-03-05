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
    """Primary function to setup the Bedrock LLM, Embedding, ChromDB storage and to initialize the local vector storage.

    Returns:
        MyBedrockObject
    """
    bedrock = MyBedrockObject()
    return bedrock

def load_data_from_url(url: str, bedrock_obj:MyBedrockObject) -> bool:
    """Access, parse text contents from the URL and load to vector database.

    Args:
        url (str): URL to access and parse
        bedrock_obj (MyBedrockObject)

    Returns:
        bool: Status indicating if the data load was successful
    """
    # now create the uuid
    id = uuid.uuid4()
    result = True
    try:
        # fetch the URL
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])    
        # load the documents to index
        index = VectorStoreIndex.from_documents(
            documents,        
            service_context=bedrock_obj.service_context,
            storage_context=bedrock_obj.storage_context,
            show_progress=True        
        )    
    except Exception as e:
        print(f'Unable to load URL successfully: {e}')
        result = False
    return result

def load_data_from_pdf(pdf, bedrock_obj:MyBedrockObject) -> bool:
    """Access, parse text contents from the PDF and load to vector database.

    Args:
        pdf (_type_): PDF file data
        bedrock_obj (MyBedrockObject)

    Returns:
        bool: Status indicating if the data load was successful
    """
    # now create the uuid
    id = uuid.uuid4()
    result = True
    try:
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
    except Exception as e:
        print(f'Unable to load PDF: {e}')
        result = False

    return result