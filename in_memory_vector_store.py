from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class InMemoryVectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)

        self.texts.extend(texts)
        self.metadata.extend(metadatas)
        self.vectors.extend(embeddings)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not self.vectors:
            return []

        query_embedding = self.embedder.encode([query_text])
        similarities = cosine_similarity(query_embedding, self.vectors)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [
            {
                "text": self.texts[i],
                "metadata": self.metadata[i],
                "score": float(similarities[i]),
            }
            for i in top_k_indices
        ]

    def clear(self) -> None:
        self.vectors = []
        self.texts = []
        self.metadata = []
    

    def chunk_excel_data(self, df, file_name, sheet_name, chunk_size=100) -> List[Dict[str, Any]]:
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            text = chunk_df.to_csv(index=False)
            metadata = {
                "source": file_name,
                "sheet": sheet_name,
                "row_start": i + 1,
                "row_end": min(i + chunk_size, len(df))
            }
            chunks.append({"text": text, "metadata": metadata})
        return chunks


vector_store = InMemoryVectorStore()

