# use this code to issue search against the vector database

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class NeuralSearcher:

    def __init__(self, collection_name:str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host="localhost", port=6333)

        # initialize encoder model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    def search(self, text:str):

        # first convert the text to a vector
        vector = self.model.encode(text).tolist()

        # now use this vector for closes match
        search_result = self.qdrant_client.query_points(
            collection_name = self.collection_name,
            query = vector,
            query_filter = None, # we don't want any filters for now
            limit = 5
        ).points

        # the search result contains both the vector and the payload that matched. We only care about the payload

        return [hit.payload for hit in search_result]
    

if __name__ == "__main__":
    searcher = NeuralSearcher(collection_name="startups")
    results = searcher.search("DevOps and Cloud computing")
    print(results)