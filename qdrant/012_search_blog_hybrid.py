from fastapi import FastAPI
# use this code to issue search against the vector database
# this should be named 004_neural_search_with_city_filter.py but, it fails in import
# hence renaming it.
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SparseVector, Prefetch, FusionQuery, Fusion
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

class NeuralSearcher:

    def __init__(self, collection_name:str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host="localhost", port=6333)

        # initialize encoder model
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

        # add a sparse model
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def search(self, text:str, search_type:str="dense"):        
        
        # now use this vector for closes match
        if search_type == "dense":
            # first convert the text to a vector
            vector = self.model.encode(text).tolist()
            search_result = self.qdrant_client.query_points(
                collection_name = self.collection_name,
                query = vector,
                using = "dense",
                limit = 5
            ).points
        elif search_type == "sparse":
            sparse_vector = next(self.sparse_model.embed(text)) # get the first entry from the iterator

            search_result = self.qdrant_client.query_points(
                collection_name = self.collection_name,
                query = SparseVector(indices = sparse_vector.indices, values = sparse_vector.values),
                using = "sparse",
                limit = 5
            ).points
        elif search_type == "hybrid":
            vector = self.model.encode(text)
            sparse_vector = next(self.sparse_model.embed(text)) # get the first entry from the iterator            

            search_result = self.qdrant_client.query_points(
                collection_name = self.collection_name,
                prefetch = [
                    Prefetch(
                        query=SparseVector(indices = sparse_vector.indices, values = sparse_vector.values),
                        using="sparse",
                        limit=10
                    ),
                    Prefetch(
                        query = vector,
                        using="dense",
                        limit=10
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF)
            ).points  
                      
        else:
            pass
            # this is a hybrid search

        # the search result contains both the vector and the payload that matched. We only care about the payload
    
        return [hit.payload for hit in search_result]    

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name="blogs_hybrid")

@app.get("/search")
def search_blogs(q:str,t:str="dense"):
    return {
        "result": neural_searcher.search(q,t)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)