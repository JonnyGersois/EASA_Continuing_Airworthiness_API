import os
from functools import lru_cache
from langdetect import detect, LangDetectException

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .extractor import extract_clean_xml_from_package, convert_xml_to_documents


BASE_DATA_DIR = "data"
BASE_VECTOR_DIR = "vectorstores"

DEFAULT_CATEGORY = "continuing_airworthiness"
DEFAULT_LANG = "en"

class HybridRetriever:
    def __init__(self, bm25, faiss, w_bm25=0.5, w_faiss=0.5):
        self.bm25 = bm25
        self.faiss = faiss
        self.w_bm25 = w_bm25
        self.w_faiss = w_faiss

    def get_relevant_documents(self, query: str):
        # Direct calls to the underlying engines — no callbacks, no LangChain retrievers
        bm25_docs = self.bm25.get_relevant_documents(query)
        faiss_docs = self.faiss.similarity_search(query, k=10)

        scored = {}

        for doc in bm25_docs:
            scored[id(doc)] = scored.get(id(doc), 0) + self.w_bm25

        for doc in faiss_docs:
            scored[id(doc)] = scored.get(id(doc), 0) + self.w_faiss

        combined = sorted(
            bm25_docs + faiss_docs,
            key=lambda d: scored[id(d)],
            reverse=True
        )

        seen = set()
        unique = []
        for doc in combined:
            if id(doc) not in seen:
                seen.add(id(doc))
                unique.append(doc)

        return unique
    

def infer_language(text):
    try:
        lang = detect(text)
        return lang if lang in ("en", "fr", "de", "es", "it") else DEFAULT_LANG
    except LangDetectException:
        return DEFAULT_LANG


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


def build_hybrid_retriever(docs, faiss_path, k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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

    # faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    return HybridRetriever(bm25_retriever, vectorstore)



def load_documents(category, lang):
    xml_dir = os.path.join(BASE_DATA_DIR, category, lang)
    if not os.path.exists(xml_dir):
        raise FileNotFoundError(f"Language '{lang}' not implemented for category '{category}'")

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {xml_dir}")

    xml_path = os.path.join(xml_dir, xml_files[0])
    clean_path = os.path.join(xml_dir, "clean.xml")

    clean_xml = extract_clean_xml_from_package(xml_path, save_clean_path=clean_path)
    docs = convert_xml_to_documents(clean_xml)
    return chunk_documents(docs)


@lru_cache(maxsize=None)
def load_retriever(category=DEFAULT_CATEGORY, lang=DEFAULT_LANG, k=5):
    docs = load_documents(category, lang)

    faiss_path = os.path.join(
        BASE_VECTOR_DIR,
        category,
        lang,
        "faiss_index"
    )

    return build_hybrid_retriever(docs, faiss_path, k=k)


def get_retriever_for_query(query_text, category=DEFAULT_CATEGORY):
    lang = infer_language(query_text)

    try:
        return load_retriever(category, lang), lang
    except FileNotFoundError:
        return load_retriever(category, DEFAULT_LANG), DEFAULT_LANG