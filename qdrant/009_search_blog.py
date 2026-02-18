from fastapi import FastAPI
# use this code to issue search against the vector database
# this should be named 004_neural_search_with_city_filter.py but, it fails in import
# hence renaming it.
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
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
            limit = 5
        ).points

        # the search result contains both the vector and the payload that matched. We only care about the payload

        return [hit.payload for hit in search_result]    

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name="blogs")

@app.get("/search")
def search_blogs(q:str):
    return {
        "result": neural_searcher.search(q)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)