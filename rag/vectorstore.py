from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.embeddings import get_embeddings
from config import CHROMA_PERSIST_DIR

def build_vectorstore():
    loader = DirectoryLoader("./data/knowledge_base/", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=CHROMA_PERSIST_DIR)
    print("Vector store built successfully!")

def load_vectorstore():
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=get_embeddings())

def add_to_vectorstore(text: str, metadata: dict = {}):
    db = load_vectorstore()
    db.add_texts([text], metadatas=[metadata])