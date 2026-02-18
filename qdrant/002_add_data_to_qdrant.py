from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np
import json

client = QdrantClient(host="localhost", port=6333)


# let's create a startups colection

if not client.collection_exists("startups"):
    client.create_collection(
        collection_name="startups",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# create an iterator and load the startup data

fd = open('startups_demo.json', 'r')
# payloads is now an interator over the data
payload = map(json.loads, fd)

# now load the vecotrs
vectors = np.load('startup_vectors.npy')


# add the records to the collection

client.upload_collection(
    collection_name = "startups",
    vectors = vectors,
    payload = payload,
    ids = None, # vecor ids will be automatically assigned
    batch_size = 256 # how many vectors will be uploaded at once?
)

print("Data added to Qdrant collection 'startups'")
