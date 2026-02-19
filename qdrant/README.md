# Blog Search with Qdrant

Semantic search over [akshayranganath.github.io](https://akshayranganath.github.io/) blog posts. Blog markdown is chunked, embedded with Sentence Transformers (dense) and FastEmbed BM25 (sparse), stored in [Qdrant](https://qdrant.tech/), and queried via a FastAPI service with a Streamlit UI. Two pipelines are supported: **dense-only** (008/009/010) and **hybrid** dense + sparse (011/012/013).

## Overview

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `007_create_embeddings_from_blogs.py` | Read markdown from `./data`, chunk, embed (dense + sparse), write `embeddings.jsonl` |
| 2a | `008_load_blogs.py` | Create Qdrant `blogs` collection (dense only) and upsert from `embeddings.jsonl` |
| 2b | `011_combined_index.py` | Create Qdrant `blogs_hybrid` collection (dense + sparse) and upsert from `embeddings.jsonl` |
| 3a | `009_search_blog.py` | FastAPI app: dense-only search against `blogs` (port 8000) |
| 3b | `012_search_blog_hybrid.py` | FastAPI app: dense / sparse / hybrid search against `blogs_hybrid` (port 8000) |
| 4a | `010_search_ui.py` | Streamlit UI for dense-only API |
| 4b | `013_search_upgraded_ui.py` | Streamlit UI with search-type selector (dense / sparse / hybrid) |

## Prerequisites

- **Qdrant** running locally (e.g. Docker: `docker run -p 6333:6333 qdrant/qdrant`)
- **Blog data**: Markdown files in `./data/` (e.g. cloned or scraped from the blog). Each file can have YAML front matter: `title`, `description`, `image`, `tag`.

## Dependencies

```text
sentence-transformers
qdrant-client
numpy
pypandoc
fastembed
fastapi
uvicorn
streamlit
requests
```

## Usage

1. **Create embeddings** (from project root, with `./data` populated):

   ```bash
   python 007_create_embeddings_from_blogs.py
   ```
   Produces `embeddings.jsonl`: chunk text, 384-d dense vectors (`all-MiniLM-L6-v2`, normalized), and sparse vectors (FastEmbed `Qdrant/bm25`). Each line includes `vector`, `sparse_vector`, and `sparse_index`.

2. **Load into Qdrant** — choose one or both:

   - **Dense-only** (original pipeline):
     ```bash
     python 008_load_blogs.py
     ```
     Creates collection `blogs` (384 dimensions, cosine) and upserts points. Use with 009 and 010.

   - **Hybrid** (dense + sparse):
     ```bash
     python 011_combined_index.py
     ```
     Creates collection `blogs_hybrid` with named vectors `dense` and `sparse`. Use with 012 and 013.

3. **Start the search API** — use the one that matches the collection you loaded:

   - Dense-only: `python 009_search_blog.py` → `GET /search?q=...`
   - Hybrid: `python 012_search_blog_hybrid.py` → `GET /search?q=...&t=dense|sparse|hybrid` (default `t=dense`). Serves at `http://0.0.0.0:8000`.

4. **Run the search UI**:

   - Dense-only: `streamlit run 010_search_ui.py`
   - Hybrid: `streamlit run 013_search_upgraded_ui.py` — includes a “Search Type” radio (dense / sparse / hybrid) and calls the API with `t=` accordingly.

## Script details

- **007**: Parses front matter, converts body to plain text (pypandoc), tokenizes and chunks (256 tokens, 32 overlap), produces **dense** embeddings (SentenceTransformer) and **sparse** embeddings (FastEmbed `Qdrant/bm25`), writes one JSON object per line to `embeddings.jsonl` with `vector`, `sparse_vector`, `sparse_index`, and metadata.
- **008**: Reads `embeddings.jsonl` (dense vector only), maps each line to a Qdrant point for collection `blogs` (UUID5 id, single vector, payload).
- **009**: `NeuralSearcher` wraps Qdrant client and SentenceTransformer; `/search` encodes the query and returns top 5 payloads from `blogs`.
- **010**: Streamlit app: text input, Search button, displays results from the dense-only API.
- **011**: Reads `embeddings.jsonl` (dense + sparse), creates collection `blogs_hybrid` with `vectors_config={"dense": ...}` and `sparse_vectors_config={"sparse": ...}`, upserts points with `vector={"dense": ..., "sparse": SparseVector(...)}`.
- **012**: `NeuralSearcher` uses both SentenceTransformer and `SparseTextEmbedding("Qdrant/bm25")`. `/search?q=...&t=dense|sparse|hybrid`: `dense` / `sparse` query the corresponding named vector; `hybrid` prefetches both (limit 10 each) and fuses with RRF, returns top 5.
- **013**: Streamlit app: text input, **Search Type** radio (dense / sparse / hybrid), calls `012` API with `t=` and displays title, text, tags, image.

## Tokenization

Blog body text is split into subword tokens using the same tokenizer as the embedding model (`all-MiniLM-L6-v2`), then sliced into overlapping chunks so each chunk fits the model’s context. This keeps chunk boundaries aligned to token boundaries and avoids splitting mid-word.

- **Tokenize**: `tokenizer.encode(markdown, add_special_tokens=False)` turns the plain-text body into token IDs.  
  → `007_create_embeddings_from_blogs.py` lines 59–60 (inside `create_chunks`). The tokenizer is the model’s tokenizer, set at line 15.
- **Chunk**: A sliding window of 256 tokens with 32-token overlap produces chunks; each slice is decoded back to text with `tokenizer.decode(chunk_ids, skip_special_tokens=True)`.  
  → `007_create_embeddings_from_blogs.py` lines 62–68 (`create_chunks`).

Chunks are then passed to the embedding step.

## Embedding

Chunk text (and at search time, the query) is turned into **dense** vectors with SentenceTransformer and **sparse** vectors with FastEmbed BM25. Dense vectors support semantic similarity; sparse vectors support lexical (keyword-style) matching.

- **Dense (index-time)**: `model.encode(chunks, batch_size=32, normalize_embeddings=True, ...)` turns each chunk into a 384-d vector.  
  → `007_create_embeddings_from_blogs.py` lines 74–78 (`get_embeddings`). Model at line 14.
- **Sparse (index-time)**: `sparse_model.embed(chunks)` produces sparse vectors (indices + values) per chunk using `Qdrant/bm25`.  
  → `007_create_embeddings_from_blogs.py` lines 16 and 81–82; results packaged at 89–91 and serialized at 108–109.
- **Search-time (dense)**: Query is encoded with the same SentenceTransformer so it lives in the same vector space.  
  → `009_search_blog.py` line 21; `012_search_blog_hybrid.py` lines 26, 43, 54 for dense and hybrid.
- **Search-time (sparse)**: Query is embedded with the same sparse model.  
  → `012_search_blog_hybrid.py` lines 19 and 34, 44 for sparse and hybrid.

Same dense and sparse models must be used for indexing (007) and search (012).

## Vectorization

Vectorization here means representing each chunk as numeric vectors and storing/querying them in Qdrant.

- **Serialization**: Each chunk’s dense vector and sparse (indices + values) are packaged with id, text, and metadata, converted to lists, and written as one JSON object per line.  
  → `007_create_embeddings_from_blogs.py` lines 85–96 and 104–112 (`get_embeddings`, `write_to_json`).
- **Storage (dense-only)**: Vectors from `embeddings.jsonl` are read (dense only), wrapped in `PointStruct`, and upserted into `blogs`.  
  → `008_load_blogs.py` lines 22–39. Collection: 384 dimensions, cosine (`008` lines 12–16).
- **Storage (hybrid)**: Dense and sparse vectors from `embeddings.jsonl` are read and upserted into `blogs_hybrid` with named vectors `dense` and `sparse`.  
  → `011_combined_index.py` lines 14–23 (collection with `vectors_config` and `sparse_vectors_config`), 29–56 (reading and building points with `vector={"dense": ..., "sparse": SparseVector(...)}`), 57–58 (upsert).
- **Search**: The query is encoded (dense and/or sparse) and sent to Qdrant; dense uses `query_points(..., using="dense")`, sparse uses `SparseVector(...)`, hybrid uses `prefetch` + `FusionQuery(fusion=Fusion.RRF)`.  
  → `009_search_blog.py` lines 24–27 (dense); `012_search_blog_hybrid.py` lines 27–60 (dense / sparse / hybrid).

## Sparse Indexing

Sparse indexing represents text as a **bag-of-words**: only the dimensions that correspond to terms present in the chunk get non-zero weights; the rest are zero. The model (e.g. BM25-style in FastEmbed) assigns higher values to important or rare terms. Search works by comparing the query’s sparse vector to each chunk’s sparse vector (e.g. dot product or BM25 similarity); chunks that share many terms with the query score higher.

**Intuitive example:** A chunk contains *“Python Lambda AWS serverless”*. Its sparse vector might look like: dimension 42 (python) → 0.8, dimension 107 (lambda) → 0.6, dimension 201 (AWS) → 0.9, dimension 55 (serverless) → 0.5, and 0 everywhere else. A query *“AWS Lambda”* has non-zero weights at dimensions 201 and 107. The chunk scores highly because those dimensions align. This gives **keyword-style** recall: exact or stemmed term overlap matters, which complements semantic (dense) search when users use specific names or phrases.

## Fusion Search

Fusion search runs **two** retrievers (dense and sparse) and **merges** their ranked lists into one. Each retriever returns its top-k (e.g. top 10); then a fusion method like **RRF (Reciprocal Rank Fusion)** combines ranks so that documents that appear in both lists, or rank well in either, get a single combined score. The final list is re-ranked by that score and trimmed (e.g. top 5).

**Intuitive example:** Dense search returns [Doc A, Doc C, Doc B, …] and sparse returns [Doc B, Doc A, Doc D, …]. With RRF, rank positions are turned into scores (e.g. 1/61, 1/62, …). Doc A gets points from being #1 on dense and #2 on sparse; Doc B from #3 on dense and #1 on sparse. So both get a boost. A doc that appears only in one list still gets points from that list. The result: you get a single ordering that favors documents that **both** semantic similarity and keyword match like, while still including strong candidates from either side.

In this project, hybrid mode in `012_search_blog_hybrid.py` does exactly this: prefetch top 10 from sparse and top 10 from dense, then fuse with `FusionQuery(fusion=Fusion.RRF)` and return the top 5.

## Notes

- **Dense**: `all-MiniLM-L6-v2` (384 dimensions). Same model for indexing (007) and search (009 or 012).
- **Sparse**: FastEmbed `Qdrant/bm25`. Used in 007 for index and in 012 for search. Hybrid collection `blogs_hybrid` requires 011 (and 012/013); original `blogs` (008/009/010) uses only dense.
- Qdrant: `localhost:6333`. API: `0.0.0.0:8000`. Change the base URL in `010_search_ui.py` or `013_search_upgraded_ui.py` if the API runs elsewhere.
