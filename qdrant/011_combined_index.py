from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, SparseVector
import numpy as np
import json
import uuid

# copy of file 008_load_blogs but with the addition of a sparse index for the text field. This will allow us to perform a full text search on the text field and then use the vector search to find the most relevant results.
client = QdrantClient(host="localhost", port=6333)


# let's create a startups colection

if not client.collection_exists("blogs_hybrid"):
    client.create_collection(
        collection_name="blogs_hybrid",
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    

# create an iterator and load the startup data

points = []
with open('embeddings.jsonl', 'r', encoding='utf-8') as fd:    
    for row in fd:        
        row = json.loads(row.strip())
        # convert the list of vectors to a numpy array
        vector = np.array(row['vector'])
        sparse_vector = np.array(row['sparse_vector'])
        sparse_index = np.array(row['sparse_index'])

        original_id = row["id"]
        points.append(
            PointStruct(
                id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_id)), # create a unique id based on the original id              
                vector = {
                    "dense": vector,
                    "sparse": SparseVector(
                        indices = sparse_index,
                        values = sparse_vector
                    )

                },
                payload = {
                    "text": row['text'],
                    "docid": row['id'],
                    **row.get('metadata')
                }
            )
        )

# now insert all the points
client.upsert(collection_name='blogs_hybrid',points=points)
print('Records inserted successfully')    