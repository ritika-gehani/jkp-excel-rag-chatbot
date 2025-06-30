# ✅ Install Required Libraries
# (No need for chromadb now)

# ✅ app.py (FAISS version)
import streamlit as st
import pandas as pd
import openpyxl
import faiss
import numpy as np
from openpyxl.utils import get_column_letter
from google.generativeai import configure, embed_content, GenerativeModel
import tempfile

# ✅ Configure Gemini API Key
configure(api_key=st.secrets["GEMINI_API_KEY"])

# ✅ Helper: Convert column index to letter
def col_to_letter(col_idx):
    return get_column_letter(col_idx)

# ✅ In-memory FAISS Index Setup
embedding_size = 768  # Size of Gemini embeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_lookup = []
metadata_lookup = []

st.title("📊 Excel Q&A Chatbot (RAG-based, FAISS)")

uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    row_id = 0
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        xls = pd.ExcelFile(tmp_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            for idx, row in df.iterrows():
                chunk_lines = []
                col_metadata = []
                for col_idx, (col_name, value) in enumerate(row.items(), start=1):
                    col_letter = col_to_letter(col_idx)
                    chunk_lines.append(f"{col_name} (Col {col_letter}): {value}")
                    col_metadata.append((col_name, col_letter, str(value)))
                chunk_text = "\n".join(chunk_lines)

                embedding = embed_content(
                    model="models/embedding-001",
                    content=chunk_text,
                    task_type="retrieval_query"
                ).get("embedding")

                index.add(np.array([embedding], dtype=np.float32))
                embedding_lookup.append(chunk_text)
                metadata_lookup.append({
                    "file": uploaded_file.name,
                    "sheet": sheet_name,
                    "row": idx + 2,
                    "columns": str(col_metadata)
                })
                row_id += 1

    st.success(f"Indexed {row_id} rows across {len(uploaded_files)} file(s).")

    question = st.text_input("Ask a question about the Excel file(s):")
    if question:
        query_embedding = embed_content(
            model="models/embedding-001",
            content=question,
            task_type="retrieval_query"
        ).get("embedding")

        D, I = index.search(np.array([query_embedding], dtype=np.float32), k=10)
        results = [(embedding_lookup[i], metadata_lookup[i]) for i in I[0] if i < len(embedding_lookup)]

        if not results:
            st.warning("No matching rows found.")
        else:
            question_lower = question.lower()
            keywords = [word.strip(".,?") for word in question_lower.split() if len(word) > 2]

            filtered = []
            for doc, meta in results:
                doc_lower = doc.lower()
                matched_keywords = [kw for kw in keywords if kw in doc_lower]
                if matched_keywords:
                    filtered.append((doc, meta, matched_keywords))

            if filtered:
                best_doc, best_meta, _ = filtered[0]
            else:
                best_doc, best_meta = results[0]

            prompt = f"""
You are an assistant answering a question using a row of an Excel file.

QUESTION: {question}

MATCHED ROW (from Excel):
{best_doc}

Provide a clear and concise answer based only on this matched row.
"""
            model = GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            st.subheader("🤖 Answer")
            st.write(response.text.strip())

            st.subheader("📁 Source Info")
            st.write(f"**File**: {best_meta['file']}")
            st.write(f"**Sheet**: {best_meta['sheet']}")
            st.write(f"**Row**: {best_meta['row']}")
