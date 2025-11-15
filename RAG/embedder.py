# embedder
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import os

class BGEEmbedder:
    def __init__(self, model_name="sentenceModel/bge-m3"):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_name = os.path.join(CURRENT_DIR, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.cpu().tolist()