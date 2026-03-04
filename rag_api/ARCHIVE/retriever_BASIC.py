# This is the original version of the retrier funtionality from the
# Streamlit demo. Ther is no multi-language or multi category support

import os
from functools import lru_cache

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# XML processing utilities
from extractor import extract_clean_xml_from_package, convert_xml_to_documents


# -----------------------------
# Document chunking
# -----------------------------
def chunk_documents(docs, threshold=1200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for doc in docs:
        if len(doc.page_content) > threshold:
            sub_docs = splitter.split_documents([doc])
            for sub in sub_docs:
                sub.metadata.update(doc.metadata)
            chunks.extend(sub_docs)
        else:
            chunks.append(doc)

    return chunks


# -----------------------------
# Build hybrid retriever
# -----------------------------
def build_hybrid_retriever(docs, faiss_path="vectorstores/easa_airworthiness", k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load or build FAISS vectorstore
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(faiss_path)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )


# -----------------------------
# Cached retriever (global)
# -----------------------------
@lru_cache(maxsize=1)
def load_retriever(k=5):
    """
    Loads and caches the retriever.
    Called once when Django starts, then reused for all requests.
    """
    xml_path = os.path.join(
        "data",
        "Easy Access Rules for Continuing Airworthiness (Regulation (EU) No 13212014).xml"
    )
    clean_path = os.path.join("data", "easa_clean.xml")

    clean_xml = extract_clean_xml_from_package(
        xml_path,
        save_clean_path=clean_path
    )

    docs = convert_xml_to_documents(clean_xml)
    chunks = chunk_documents(docs)

    return build_hybrid_retriever(chunks, k=k)


# Expose a ready-to-use retriever instance
retriever = load_retriever()