# searcher
import faiss
import pickle
import numpy as np

class FaissSearcher:
    def __init__(self, index_path: str, text_path: str):
        """
        Initialize the FaissSearcher by loading the FAISS index and the corresponding texts.
        """
        self.index = faiss.read_index(index_path)
        with open(text_path, 'rb') as f:
            self.texts = pickle.load(f)

    def search(self, query_vector, top_k=5):
        """
        Perform a similarity search on the FAISS index.

        Args:
            query_vector (list or np.array): The query vector.
            top_k (int): The number of top results to return.

        Returns:
            list of tuple: A list of (text, similarity_score) pairs.
        """
        D, I = self.index.search(np.array([query_vector]).astype('float32'), top_k)
        results = [(self.texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results
