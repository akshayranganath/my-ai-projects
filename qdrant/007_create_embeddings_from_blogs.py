from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from tqdm.notebook import tqdm
import re
import tiktoken
from typing import List, Dict
import json
import pypandoc


# first create the model
model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = model.tokenizer
sparse_model = SparseTextEmbedding("Qdrant/bm25")

def get_metadata(markdown):

    FRONT_MATTER_RE = re.compile(
        r"^\s*\ufeff?---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
        re.DOTALL
    )

    title = description = image = ""
    tags = []

    m = FRONT_MATTER_RE.match(markdown)
    if m:
        meta_block = m.group(1)
        body = markdown[m.end():]        

        for line in meta_block.splitlines():
            if line.startswith('title'):
                title = line.split('title:')[1].strip()
            elif line.startswith('description'):
                description = line.split('description:')[1].strip()
            elif line.startswith('image'):
                image = line.split('image:')[1].strip()
            elif line.startswith('tag'):
                tags = line.split('tag:')
                if len(tags) > 1:
                    tags = tags[1].strip()
                    if tags:
                        tags = tags.split('[')[1].split(']')[0].split(',')                    
    else:
        body = markdown
    
    metadata = {
        "title": title,
        "description": description,
        "image": image,
        "tags": tags
    }
    
    return (metadata, body)


def create_chunks(markdown:str, max_tokens:int=256, overlap:int=32) -> List[str]:
    chunks = []
    start = 0
    tokens = tokenizer.encode(markdown, add_special_tokens=False) #Tokenize the text without inserting the model’s special marker tokens
    print(f'Tokens in document: {len(tokens)}')

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_ids = tokens[start:end]

        # now decode and get the chunks
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))        
        start = max(0, start + (end - overlap)) # handles errors when the document size is very small.
    
    return chunks

def get_embeddings(chunks:List[str], file_path:str, metadata:Dict)->List[Dict]:
    vectors = model.encode(
        chunks,
        batch_size=32,
        normalize_embeddings=True, # good for cosine similarity
        show_progress_bar=True
    )

    # bm_25 vectors
    sparse_vectors = sparse_model.embed(chunks)
    
    # package for upsert later
    out = []
    for i, (chunk, vec, sparse_vec) in enumerate(zip(chunks, vectors, sparse_vectors)):
        out.append({
            "id": f"{Path(file_path).name}:{i}",
            "text": chunk,
            "vector": vec,
            "sparse_vector": sparse_vec.values,
            "sparse_index": sparse_vec.indices,
            "metadata": {
                "source": str(file_path),
                "chunk_id": i,
                "title": metadata.get('title'),
                "description": metadata.get('description'),
                "tags": metadata.get('tags'),
                "image": metadata.get('image')
            }
        })
    return out

def write_to_json(embeddings:List[Dict], fd):
    for embed in embeddings:
        # convert the vecotor to a list
        embed['vector'] = embed['vector'].tolist()
        # same with sparse vectors
        embed['sparse_vector'] = embed['sparse_vector'].tolist()        
        embed['sparse_index'] = embed['sparse_index'].tolist()        
        fd.write(f"{json.dumps(embed)}\n")
    
def main():
    # first read the files
    with open('embeddings.jsonl', 'w',encoding='utf-8') as fd:
        for md_file in Path('./data').rglob('*.md'):
            markdown = md_file.read_text(encoding='utf-8', errors="ignore")  
            metadata, body = get_metadata(markdown=markdown)      
            # remove white spaces
            body = pypandoc.convert_text(body, "plain", format="md")
            body = re.sub(r"\s+", " ", body).strip()            
            embeddings = get_embeddings(create_chunks(markdown=body),
                                        file_path=md_file,
                                        metadata=metadata
            )
            write_to_json(embeddings=embeddings, fd=fd)            
            print(f"Processed: {str(md_file)}")
            
    

if __name__=="__main__":
    print('Done')
    main()