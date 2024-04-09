import chromadb
from chromadb.db.base import NotFoundError, UniqueConstraintError
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


class DBObject():
    def __init__(self, collection_name:str='multimodal_collection', db_path:str='../data/my_collection')->None:
    # this may throw errors for further library installations
        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.image_loader = ImageLoader()
        self.client = chromadb.PersistentClient(path=db_path)

        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                data_loader=self.image_loader)
        except UniqueConstraintError as e:
            print('Collection already present')
            self.collection = self.client.get_collection(
                name = 'multimodal_collection',
                embedding_function=self.embedding_function,
                data_loader=self.image_loader
            )


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