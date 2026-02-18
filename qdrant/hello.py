from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient("http://localhost:6333")

if not client.collection_exists("hello-world"):
    client.create_collection(
        collection_name="hello-world",
        vectors_config=VectorParams(size=4, distance=Distance.DOT),    
    )
    print("Collection 'hello-world' created.")
else:
    print("Collection 'hello-world' already exists.")

# we have already added the records. Commenting it out now.
# points = [
#     PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#     PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#     PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
#     PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"})
# ]


# try:
#     operation_info = client.upsert(
#         collection_name="hello-world",
#         wait=True,
#         points=points
#     )

#     print(operation_info)
# except UnexpectedResponse as e:
#     print(f"An error occurred: {e}")

# issue a basic query.
# which vector is closest to [0.2, 0.1, 0.9, 0.7]?
search_result = client.query_points(
    collection_name="hello-world",
    query=[0.2, 0.1, 0.9, 0.7],
    with_payload=True,
    #with_vectors=True,
    limit=3
)
print(search_result.points)

print("*** Adding a query filter ***")

search_result = client.query_points(
    collection_name="hello-world",
    query=[0.2, 0.1, 0.9, 0.7],
    with_payload=True,
    #with_vectors=True,
    limit=3,
    query_filter = Filter(
        must=[
            FieldCondition(
                key="city",
                match=MatchValue(value="London")
            )
        ]
    )
).points
print(search_result)