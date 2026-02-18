from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_json('startups_demo.json', lines=True)

vectors = model.encode(
    [row.alt+", "+row.description for row in df.itertuples()],
    show_progress_bar=True
)


vectors.shape
np.save('startup_vectors.npy', vectors, allow_pickle=False)