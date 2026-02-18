from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

from fastembed import SparseTextEmbedding

model = SentenceTransformer('all-MiniLM-L6-v2')

text1 = "The cat sat on the mat"
text2 = "The feline was sitting on the rug"
text3 = "The dog was barking loudly"

vectors = model.encode([text1, text2, text3])

dot1 = np.dot(vectors[0], vectors[1]) # this should be close to 1, as text1 and text2 are similar
dot2 = np.dot(vectors[0], vectors[2]) # this should be close to 0

print(f"📊 Dot product of text1 and text2 :\t{dot1:.4f}")
print(f"📊 Dot product of text1 and text3 :\t{dot2:.4f}")

cos1 = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1])) 
cos2 = np.dot(vectors[0], vectors[2]) / (norm(vectors[0]) * norm(vectors[2]))

print(f"📐 Cosine similarity of text1 and text2 :\t{cos1:.4f}")
print(f"📐 Cosine similarity of text1 and text3 :\t{cos2:.4f}")

print("📏 Norms of the vectors:")
print(f"  Vector 1 :\t{norm(vectors[0]):.4f}")
print(f"  Vector 2 :\t{norm(vectors[1]):.4f}")
print(f"  Vector 3 :\t{norm(vectors[2]):.4f}")
print("💡 Note :\tNorms are ≈1, so dot product ≈ cosine similarity")

print("=" * 50)
print("🔍 BM25 Similarity (using term frequency)")

sparse_model = SparseTextEmbedding("Qdrant/bm25")

sparse_vectors = sparse_model.embed([text1, text2, text3])
sparse_vectors = list(sparse_vectors)
print(sparse_vectors)
print(f"📊 BM25 similarity of text1 and text2 :\t{np.dot(sparse_vectors[0].values, sparse_vectors[1].values):.4f}")
print(f"📊 BM25 similarity of text1 and text3 :\t{np.dot(sparse_vectors[0].values, sparse_vectors[2].values):.4f}")