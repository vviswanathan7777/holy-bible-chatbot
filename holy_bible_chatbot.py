import os
import logging
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable Path and URL
pdf_path = st.text_input("Enter PDF file path:", "The-Holy-Bible-King-James-Version.pdf")
embedding_url = st.text_input("Enter Ollama server URL:", "http://localhost:11434")
db_name = "Holy_Bible_DB"

# Helper function
def chunker(iterable, size):
    from itertools import islice
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

@st.cache_data
def process_and_index_pdf(pdf_path, db_name, embedding_model, embedding_url, batch_size=10):
    """Extracts text from the PDF, splits it into chunks, and builds the FAISS index."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=embedding_url)
    embedding_dimension = getattr(embeddings, "embedding_dimension", 768)  # Default to 768 if attribute missing
    index = IndexFlatL2(embedding_dimension)
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}

    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    for batch in chunker(chunks, batch_size):
        try:
            # Extract text and metadata correctly
            batch_texts = [getattr(doc, 'page_content', '') for doc in batch]
            batch_metadata = [getattr(doc, 'metadata', {}) for doc in batch]

            # Embed and add to vector store
            batch_embeddings = embeddings.embed_documents(batch_texts)
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadata,
                embeddings=batch_embeddings,
            )
        except AttributeError as e:
            logger.error(f"Error processing batch: {e}")
            st.error(f"Error processing documents: {e}")
            st.stop()

    vector_store.save_local(db_name)
    return vector_store

# Embedding Model Configuration
embedding_model = 'granite3.1-dense:latest'

# Pre-check if Ollama server is reachable
try:
    response = httpx.get(embedding_url)
    response.raise_for_status()
    logger.info(f"Successfully connected to Ollama server at {embedding_url}")
except Exception as e:
    st.error(f"Cannot connect to Ollama server at {embedding_url}. Please ensure it's running.")
    logger.error(f"Ollama server connection failed: {e}")
    st.stop()

# Initialize or rebuild FAISS index
try:
    vector_store = FAISS.load_local(
        db_name, embeddings=OllamaEmbeddings(
            model=embedding_model, base_url=embedding_url
        ), allow_dangerous_deserialization=True
    )
    logger.info("Loaded existing FAISS index.")
except FileNotFoundError:
    st.warning("Indexing PDF file... This may take a few minutes.")
    logger.info("Reindexing PDF file as index file was not found.")
    vector_store = process_and_index_pdf(pdf_path, db_name, embedding_model, embedding_url)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    logger.error(f"Error loading FAISS index: {e}")
    st.stop()

st.write("Application running on AWS EC2 instance")
st.write(f"Embedding Model: {embedding_model}")
st.write(f"Ollama Embedding URL: {embedding_url}")

# Add a user input prompt field
user_input = st.text_input("Enter your question about the Holy Bible:")

if user_input:
    try:
        # Convert query to embedding
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=embedding_url)
        query_embedding = embeddings.embed_query(user_input)

        # Perform similarity search
        results = vector_store.similarity_search_by_vector(query_embedding, k=5)  # Retrieve top 5 results
        st.write("Top Matches:")
        for result in results:
            # Use `page_content` and `metadata` attributes correctly
            st.write(f"- {getattr(result, 'page_content', 'No content')} (metadata: {getattr(result, 'metadata', {})})")
    except Exception as e:
        st.error(f"Error processing your query: {e}")
        logger.error(f"Query error: {e}")

