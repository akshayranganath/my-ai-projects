from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import json
import uuid

client = QdrantClient(host="localhost", port=6333)


# let's create a startups colection

if not client.collection_exists("blogs"):
    client.create_collection(
        collection_name="blogs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    

# create an iterator and load the startup data

points = []
with open('embeddings.jsonl', 'r', encoding='utf-8') as fd:    
    for row in fd:        
        row = json.loads(row.strip())
        # convert the list of vectors to a numpy array
        vector = np.array(row['vector'])
        original_id = row["id"]
        points.append(
            PointStruct(
                id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_id)), # create a unique id based on the original id              
                vector = vector,
                payload = {
                    "text": row['text'],
                    "docid": row['id'],
                    **row.get('metadata')
                }
            )
        )

# now insert all the points
client.upsert(collection_name='blogs',points=points)
print('Records inserted successfully')    