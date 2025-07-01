from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pandas as pd

class InMemoryVectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.docs = []  # stores {"embedding": ..., "text": ..., "metadata": ...}

    def chunk_excel_data(self, df: pd.DataFrame, filename: str, sheet: str) -> List[Dict[str, Any]]:
        chunks = [{
            "text": f"Sheet: {sheet}\nColumns: {', '.join(df.columns)}",
            "metadata": {"filename": filename, "sheet": sheet, "row": 1, "column": "HEADER"}
        }]
        for idx, row in df.iterrows():
            row_data = [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
            if row_data:
                chunks.append({
                    "text": f"Sheet: {sheet}, Row {idx+2}:\n" + "\n".join(row_data),
                    "metadata": {"filename": filename, "sheet": sheet, "row": idx+2, "column": "ALL"}
                })
        return chunks

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        self.docs.extend([
            {"embedding": emb, "text": text, "metadata": meta}
            for emb, text, meta in zip(embeddings, texts, metadatas)
        ])

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        query_vec = self.embedder.encode(query_text)
        scored = sorted(
            self.docs,
            key=lambda doc: self._cosine_similarity(query_vec, doc["embedding"]),
            reverse=True
        )
        return [{"text": doc["text"], "metadata": doc["metadata"]} for doc in scored[:k]]

    def clear_collection(self):
        self.docs = []

    def _cosine_similarity(self, a, b):
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

vector_store = InMemoryVectorStore()
