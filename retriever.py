"""
Simple TF-IDF retriever over a folder of .txt knowledge base documents.
Used to find the most relevant document chunk(s) before asking the LLM,
so the model answers grounded in company documentation instead of
hallucinating.
"""
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    def __init__(self, folder: str):
        self.folder = folder
        self.filenames = []
        self.chunks = []
        self._load()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks) if self.chunks else None

    def _load(self, chunk_size: int = 500):
        """Loads .txt files and splits them into rough paragraph chunks."""
        for path in sorted(glob.glob(os.path.join(self.folder, "*.txt"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                self.chunks.append(p)
                self.filenames.append(os.path.basename(path))

    def search(self, query: str, top_k: int = 2):
        if not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] > 0:
                results.append({
                    "source": self.filenames[i],
                    "text": self.chunks[i],
                    "score": float(scores[i]),
                })
        return results
