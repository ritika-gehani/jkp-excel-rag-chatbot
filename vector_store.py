import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pandas as pd

class ExcelVectorStore:
    def __init__(self, collection_name: str = "excel_data"):
        self.collection_name = collection_name
        self.client = chromadb.Client()  # In-memory only
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

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
        ids = [f"{m['filename']}_{m['sheet']}_{m['row']}_{i}" for i, m in enumerate(metadatas)]

        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            embedding = self.embedder.encode(query_text).tolist()
            results = self.collection.query(query_embeddings=[embedding], n_results=k)

            return [
                {"text": results["documents"][0][i], "metadata": results["metadatas"][0][i]}
                for i in range(len(results["documents"][0]))
            ]
        except Exception:
            return []

    def clear_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)

# Instantiate
vector_store = ExcelVectorStore()
