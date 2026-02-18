# Blog Search with Qdrant

Semantic search over [akshayranganath.github.io](https://akshayranganath.github.io/) blog posts. Blog markdown is chunked, embedded with Sentence Transformers, stored in [Qdrant](https://qdrant.tech/), and queried via a FastAPI service with a Streamlit UI.

## Overview

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `007_create_embeddings_from_blogs.py` | Read markdown from `./data`, chunk, embed, write `embeddings.jsonl` |
| 2 | `008_load_blogs.py` | Create Qdrant `blogs` collection and upsert vectors from `embeddings.jsonl` |
| 3 | `009_search_blog.py` | FastAPI app that encodes queries and searches Qdrant (port 8000) |
| 4 | `010_search_ui.py` | Streamlit UI that calls the API for semantic search |

## Prerequisites

- **Qdrant** running locally (e.g. Docker: `docker run -p 6333:6333 qdrant/qdrant`)
- **Blog data**: Markdown files in `./data/` (e.g. cloned or scraped from the blog). Each file can have YAML front matter: `title`, `description`, `image`, `tag`.

## Dependencies

```text
sentence-transformers
qdrant-client
numpy
pypandoc
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
   Produces `embeddings.jsonl` (chunk text + 384-d vectors from `all-MiniLM-L6-v2`, normalized for cosine similarity).

2. **Load into Qdrant**:

   ```bash
   python 008_load_blogs.py
   ```
   Creates collection `blogs` if missing (384 dimensions, cosine distance) and upserts all points.

3. **Start the search API**:

   ```bash
   python 009_search_blog.py
   ```
   Serves at `http://0.0.0.0:8000`. Example: `GET /search?q=your query` returns top 5 matches.

4. **Run the search UI**:

   ```bash
   streamlit run 010_search_ui.py
   ```
   UI calls `http://0.0.0.0:8000/search?q=...` and shows title, text snippet, tags, and image when present.

## Script details

- **007**: Parses front matter, converts body to plain text (pypandoc), tokenizes with the model’s tokenizer, chunks (256 tokens, 32 overlap), encodes with SentenceTransformer, writes one JSON object per line to `embeddings.jsonl`.
- **008**: Reads `embeddings.jsonl`, maps each line to a Qdrant point (UUID5 from doc id, vector, payload with `text`, `docid`, title, description, tags, image).
- **009**: `NeuralSearcher` wraps Qdrant client and the same SentenceTransformer model; `/search` encodes the query and returns payloads of the top 5 points.
- **010**: Single-page Streamlit app: text input, Search button, then displays each result’s title, text, tags, and optional image.

## Notes

- Embedding model: `all-MiniLM-L6-v2` (384 dimensions). Same model must be used for indexing (007) and search (009).
- Qdrant is expected at `localhost:6333`; the API at `0.0.0.0:8000`. Change the URL in `010_search_ui.py` if your API runs elsewhere.
