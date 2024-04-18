import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

DB_PATH='../data/my_collection'
DB_COLLECTION_NAME='multimodal_collection'

class DBObject():
    def __init__(self, collection_name:str=DB_COLLECTION_NAME, db_path:str=DB_PATH)->None:
    # this may throw errors for further library installations
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.image_loader = ImageLoader()
        self.client = chromadb.PersistentClient(path=db_path)
    
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}, # use cosine distance metric
            data_loader=self.image_loader)

    def add_object_to_collection(self, file_name:str, url:str)->bool:
        result = False
        try:
            self.collection.add(ids=[file_name], uris=[file_name], metadatas=[{"url": url}])
            result = True
        except Exception as e:
            print(e)
        return result

    def query_object_in_collection(self, file_path:str):
        result = None
        try:
            result = self.collection.query(query_uris=[file_path], n_results=3)
        except Exception as e:
            print(e)
        return result