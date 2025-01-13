from langchain.docstore import InMemoryDocstore
from faiss import IndexFlatL2
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path and database variables
pdf_path = "The-Holy-Bible-King-James-Version.pdf"
db_name = "Holy_Bible_DB"

# Helper function
def chunker(iterable, size):
    """Helper function to split iterable into smaller chunks."""
    from itertools import islice
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

@st.cache_data
def process_and_index_pdf(pdf_path, db_name, embedding_model, embedding_url, batch_size=10):
    """Extracts text from the PDF, splits it into chunks, and builds the FAISS index."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=embedding_url)

    # Create a FAISS index
    index = IndexFlatL2(embeddings.embedding_dimension)
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}

    # Initialize FAISS vector store
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    # Add documents to FAISS in batches
    for batch in chunker(chunks, batch_size):
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
        vector_store.add_texts(
            texts=[doc.page_content for doc in batch],
            metadatas=[doc.metadata for doc in batch],
            embeddings=batch_embeddings,
        )

    # Save the FAISS index
    vector_store.save_local(db_name)
    return vector_store

# Initialize or rebuild FAISS index
embedding_model = 'granite3.1-dense:latest'
embedding_url = 'http://localhost:11434'

try:
    vector_store = FAISS.load_local(
        db_name, embeddings=OllamaEmbeddings(
            model=embedding_model, base_url=embedding_url
        ), allow_dangerous_deserialization=True
    )
except Exception as e:
    st.warning("Indexing PDF file... This may take a few minutes.")
    st.write(f"Reindexing due to: {e}")
    vector_store = process_and_index_pdf(pdf_path, db_name, embedding_model, embedding_url)
