import json
from typing import List, Dict
import os
from multiprocessing import Pool
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm
import uuid


# ---------------- CONFIG ----------------

JSON_FOLDER = "/Users/shubhamagrawal/Documents/MS_fall_25/MS_fall_25_uni/ra/data/clinical_pharmacology/multimodal_extraction/output_all_pdfs"
COLLECTION_NAME = "clin_pharma_rag_docs"
BATCH_SIZE = 32
N_WORKERS = 4

model = SentenceTransformer("BAAI/bge-base-en")
qdrant = QdrantClient(path="./qdrant_storage1")
# qdrant = QdrantClient(":memory:")   # replace with your server

# create collection once
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"size": 768, "distance": "Cosine"}
)

# chunking
def json_to_chunks(json_path: str) -> List[Dict]:

    with open(json_path, "r") as f:
        data = json.load(f)

    document_id = data["document_metadata"]["source_file"]
    sections = data["document_structure"]["sections"]

    chunks = []

    for i, sec in enumerate(sections):

        chunk = {
            "doc_id": document_id,
            "chunk_id": f"{document_id}_sec_{i}",
            "section_level": sec.get("level", 1),
            "heading": sec.get("heading", ""),
            "page_number": sec.get("page_number", None),
            "text": sec.get("heading", ""),   # heading already contains section text in your extraction
            "metadata": {
                "source": document_id,
                "type": "section"
            }
        }

        chunks.append(chunk)

    return chunks

# ---------------- EMBEDDING ----------------

def embed_chunks(chunks: List[Dict]):

    texts = [c["text"] for c in chunks]

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vectors

# ---------------- INGESTION WORKER ----------------

def process_file(json_file):

    chunks = json_to_chunks(json_file)
    vectors = embed_chunks(chunks)

    points = []

    for chunk, vector in zip(chunks, vectors):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),       # unique ID
                vector=vector.tolist(),     # embedding vector
                payload={"text": chunk}     # store your chunk text in payload
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return f"Processed {json_file}"

# ---------------- PARALLEL DRIVER ----------------

if __name__ == "__main__":

    json_files = [
        os.path.join(JSON_FOLDER, f)
        for f in os.listdir(JSON_FOLDER)
        if f.endswith(".json")
    ]

    with Pool(N_WORKERS) as pool:
        list(tqdm(pool.imap(process_file, json_files), total=len(json_files)))

    print("Ingestion complete")

    hits = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=[0.1] * 768,
    limit=5
    )

    print(hits)

# chunks = json_to_semantic_chunks("/Users/shubhamagrawal/Documents/MS_fall_25/MS_fall_25_uni/ra/data/clinical_pharmacology/multimodal_extraction/output_all_pdfs/2017_49867_STERITALC (talc) Powder_cl_pharm_rv_205555Orig1s000ClinPharmR_extracted.json")

# print(chunks[0])