import streamlit as st
import os
import uuid
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SearchParams
from qdrant_client.http.models import Filter, VectorParams
from ollama import chat

# ----------------------------
# Configuration
# ----------------------------
JSON_FOLDER = "/Users/shubhamagrawal/Documents/MS_fall_25/MS_fall_25_uni/ra/data/clinical_pharmacology/multimodal_extraction/output_all_pdfs"
COLLECTION_NAME = "clin_pharma_rag_docs"
# STORAGE_PATH = "./qdrant_storage_streamlit"    # Persistent local storage
BATCH_SIZE = 32
# qdrant = QdrantClient(url="http://localhost:6333")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en"
OLLAMA_MODEL = "mistral"

# ----------------------------
# Initialize Qdrant
# ----------------------------
qdrant = QdrantClient(url="http://localhost:6333")

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance="Cosine")  # embedding size
    )

# ----------------------------
# Initialize models
# ----------------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# ollama_client = Ollama()

# ----------------------------
# Helper functions
# ----------------------------
def load_json_docs(folder):
    all_docs = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            with open(os.path.join(folder, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    all_docs.append(data)
                elif isinstance(data, list):
                    all_docs.extend(data)
    return all_docs

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def ingest_documents(json_folder):
    docs = load_json_docs(json_folder)
    points = []
    for doc in docs:
        text = doc.get("text", "")
        for chunk in chunk_text(text):
            vector = embed_model.encode(chunk).tolist()
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "source": doc.get("id", "unknown")}
            ))
    # Batch upsert
    for i in range(0, len(points), BATCH_SIZE):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i+BATCH_SIZE]
        )
    print(f"Ingested {len(points)} chunks into Qdrant!")
    return len(points)

def retrieve(query, top_k=5):
    query_vector = embed_model.encode(query).tolist()
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]



def generate_answer(prompt):
    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    if "messages" in response and len(response["messages"]) > 0:
        return response["messages"][-1]["content"]
    else:
        return "No response from model."

@st.cache_data(show_spinner=False)
def ingest_once(json_folder):
    return ingest_documents(json_folder)

ingest_once(JSON_FOLDER)
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Clinical Pharm RAG System")



# Step 2: Query
st.subheader("Ask a Question")
user_query = st.text_area("Enter your query here:")

top_k = st.slider("Number of chunks to retrieve:", min_value=1, max_value=10, value=5)

if st.button("Get Answer") and user_query.strip():
    with st.spinner("Retrieving and generating answer..."):
        context_chunks = retrieve(user_query, top_k=top_k)
        prompt = f"Answer the question using the context below:\n{'\n'.join(context_chunks)}\nQuestion: {user_query}\nAnswer:"
        answer = generate_answer(prompt)

    st.subheader("Retrieved Chunks")
    for i, chunk in enumerate(context_chunks, 1):
        st.markdown(f"**Chunk {i}:** {chunk}")

    st.subheader("Generated Answer")
    st.write(answer)
