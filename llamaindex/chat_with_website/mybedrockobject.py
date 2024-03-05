from dotenv import load_dotenv
import os
import boto3

from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
# enabling debug
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.vector_stores.chroma import ChromaVectorStore
# library for database
import chromadb

class MyBedrockObject():
    """Custom object to contain all the necessary information on the Bedrock engine and LlamaIndex objects.

        Objects contained are LLM, Embedding, ServiceContext, StorageContext, VectorIndex and ChatEngine
    """
    
    def __init__(self):
        load_dotenv()
        # initialize the service and storage context
        debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager(handlers=[debug])
        # Setup bedrock
        # first get the mode - local or hosted on EC2
        if os.environ.get('RUN_MODE')=="local":
            self.llm = Bedrock(
                model=os.environ["BEDROCK_MODEL"],
                temperature=os.environ["CHAT_TEMPERATURE"],
                profile_name=os.environ['AWS_PROFILE_NAME'],
            )
            self.embedding = BedrockEmbedding(
                profile_name=os.environ['AWS_PROFILE_NAME'],
                model=os.environ['BEDROCK_EMBED_MODEL']
            )
        else:
            bedrock_runtime = boto3.client(
                service_name=os.environ['BEDROCK_RUNTIME_NAME'],
                region_name=os.environ['AWS_REGION'],
            )

            self.llm = Bedrock(
                model=os.environ["BEDROCK_MODEL"],
                temperature=os.environ["CHAT_TEMPERATURE"],
                client=bedrock_runtime,
            )
            self.embedding = BedrockEmbedding(
                client=bedrock_runtime,
                model=os.environ['BEDROCK_EMBED_MODEL']
            )

        # to read on EC2/AWS instances, use these methods instead
        # https://medium.com/@dminhk/llamaindex-q-a-over-your-data-using-amazon-bedrock-and-streamlit-1e52a096ec7c
        # create a service context using the Bedrock embeddings
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.embedding  # , callback_manager=callback_manager
        )

        dbconn = chromadb.Client(
            chromadb.config.Settings(
            is_persistent=True, persist_directory=os.environ["PERSIST_DIR"]
            )
        )
        chroma_collection = dbconn.get_or_create_collection(os.environ["COLLECTION_NAME"])
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # now setup the chat engine as well
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            service_context=self.service_context,
        )
        # Considering we want to use RAG, use the CHAT_MODE as 'context'.
        # for details on different chat modes, refer to https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/root.html
        self.chat_engine = index.as_chat_engine(
                chat_mode=os.environ.get('CHAT_MODE'), verbose=True
        )

    def get_llm_model()->str:
        """Returns the name of the bedrock model being used.

        Returns:
            str: Bedrock foundation model name
        """
        return os.environ["BEDROCK_MODEL"]
    
    def get_embed_model()->str:
        """Returns the name of the bedrock embedding model being used.

        Returns:
            str: Bedrock embedding model name
        """
        return os.environ['BEDROCK_EMBED_MODEL']