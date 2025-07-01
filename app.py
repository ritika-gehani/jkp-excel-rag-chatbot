import streamlit as st
import pandas as pd
import google.generativeai as genai
from in_memory_vector_store import vector_store

# 🧠 Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# 🧼 Initialize session
st.session_state.setdefault("messages", [])

# 📁 File processing
def process_and_preview(file):
    ext = file.name.split('.')[-1].lower()
    try:
        if ext in ['xlsx', 'xls']:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                vector_store.add_documents(vector_store.chunk_excel_data(df, file.name, sheet))
                with st.sidebar.expander(f"Preview {file.name} - {sheet}"):
                    st.dataframe(df.head(10))
            return f"Processed {file.name} ({len(xls.sheet_names)} sheets)"
        elif ext == 'csv':
            df = pd.read_csv(file)
            vector_store.add_documents(vector_store.chunk_excel_data(df, file.name, "CSV"))
            with st.sidebar.expander(f"Preview {file.name}"):
                st.dataframe(df.head(10))
            return f"Processed {file.name}"
        else:
            return f"Unsupported file: {file.name}"
    except Exception as e:
        return f"Error reading {file.name}: {e}"

# 🤖 Chat handling
def handle_chat(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not uploaded_files:
            msg = "Please upload Excel/CSV files first before asking questions."
        else:
            results = vector_store.query(prompt, k=10)
            if not results:
                msg = "No relevant data found. Try a different query or upload more data."
            else:
                context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(results)])
                try:
                    with st.spinner("Thinking..."):
                        response = model.generate_content(
                            f"Analyze this data and answer: {prompt}\n\nData:\n{context}",
                            generation_config={"max_output_tokens": 1000, "temperature": 0.3}
                        )
                        msg = response.text
                except Exception as e:
                    msg = f"Error: {e}"
        st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})

# 📤 Sidebar
st.sidebar.title("📤 Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"], accept_multiple_files=True)

# 🔄 File handling
if uploaded_files:
    vector_store.clear()
    for file in uploaded_files:
        st.sidebar.info(process_and_preview(file))

if st.sidebar.button("Clear All Data", type="primary"):
    vector_store.clear()
    st.session_state.messages = []
    st.sidebar.success("Data cleared")

# 💬 Chat display
st.title("📊 Excel/CSV Chatbot")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 📝 Chat input
if prompt := st.chat_input("Ask about your data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    handle_chat(prompt)
