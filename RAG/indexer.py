# indexer
import faiss
import pickle
import numpy as np
import json
import os
from .chunker import load_and_chunk
from .embedder import BGEEmbedder

class FaissIndexer:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = [] 

    def add(self, vectors, texts):
        self.index.add(np.array(vectors).astype('float32'))
        self.texts.extend(texts)

    def save(self, index_path: str, text_path: str):
        faiss.write_index(self.index, index_path)
        with open(text_path, 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self, index_path: str, text_path: str):
        self.index = faiss.read_index(index_path)
        with open(text_path, 'rb') as f:
            self.texts = pickle.load(f)

    def reset(self):
        self.index = faiss.IndexFlatIP(self.index.d)
        self.texts = []

def update_index(docs_dir="docs", index_path="faiss.index", text_path="chunks.pkl", registry_path="registered_files.json", embedder=BGEEmbedder()):
    """
    Update the FAISS index by processing new or modified files in the docs directory.
    """
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(CURRENT_DIR, docs_dir)
    index_path = os.path.join(CURRENT_DIR, index_path)
    text_path = os.path.join(CURRENT_DIR, text_path)
    registry_path = os.path.join(CURRENT_DIR, registry_path)

    indexer = FaissIndexer(dim=1024)
    if os.path.exists(index_path) and os.path.exists(text_path):
        indexer.load(index_path, text_path)
    
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registered_files = json.load(f)
    else:
        registered_files = {}

    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if not os.path.isfile(filepath) or not (filepath.endswith('.pdf') or filepath.endswith('.txt')):
            continue
        
        last_modified = os.path.getmtime(filepath)
        
        if filepath in registered_files and registered_files[filepath] >= last_modified:
            continue

        print(f"Processing new file: {filepath}")
        
        chunks = load_and_chunk(filepath)
        vectors = embedder.encode(chunks)
        
        # Add new vectors to the FAISS index
        indexer.add(vectors, chunks)
        
        # Update the registered files info
        registered_files[filepath] = last_modified
    
    indexer.save(index_path, text_path)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registered_files, f, indent=2)

    print("Index update complete.")
