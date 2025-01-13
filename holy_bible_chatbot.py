from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
import hashlib
from itertools import islice

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    raise ImportError("`PyPDFLoader` requires `pypdf`. Please install it using `pip install pypdf`.")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure faiss-cpu and pypdf are installed
# pip install faiss-cpu pypdf

# Step 1: Helper Functions
def calculate_file_hash(file_path):
    """Calculate the hash of a file to use as a cache key."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def chunker(iterable, size):
    """Helper function to split iterable into smaller chunks."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

# Step 2: Load and preprocess the PDF
pdf_path = "The-Holy-Bible-King-James-Version.pdf"
db_name = "Holy_Bible_DB"

@st.cache_data
def process_and_index_pdf(pdf_path, db_name, embedding_model, embedding_url, batch_size=10):
    """Extracts text from the PDF, splits it into chunks, and builds the FAISS index in batches."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings in batches and index them in FAISS
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=embedding_url
    )
    vector_store = FAISS()  # Initialize an empty FAISS vector store

    for batch in chunker(chunks, batch_size):
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
        vector_store.add_texts(
            texts=[doc.page_content for doc in batch],
            metadatas=[doc.metadata if isinstance(doc.metadata, dict) else {} for doc in batch],
            embeddings=batch_embeddings,
        )

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

# Step 3: Define Prompt and Chat History
prompt_template = """
    You are an assistant for question-answering tasks and an expert on the Holy Bible. Provide answers in concise bullet points (5 to 7) with sources cited.
    
    If the question is outside the scope of the Holy Bible, respond with: "I don't know".
    
    Question: {question}
    Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

llm = ChatOllama(model=embedding_model, base_url=embedding_url)
rag_chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key='question',
    history_messages_key='history'
)

# Step 4: Streamlit App
st.title("Holy Bible Chatbot v0.2")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

query = st.chat_input("Ask your Holy Bible question?")

if query:
    st.session_state.chat_history.append({'role': 'user', 'content': query})

    with st.chat_message("user"):
        st.markdown(query)

    try:
        if vector_store is None:
            st.error("The FAISS vector store is unavailable. Please try re-indexing the data.")
        else:
            docs = vector_store.search(query=query, k=5, search_type="similarity")
            context = "\n\n".join([doc.page_content for doc in docs])

            with st.chat_message("assistant"):
                response = st.write_stream(
                    runnable_with_history.stream(
                        {'question': query, 'context': context},
                        config={'configurable': {'session_id': 'user_id'}}
                    )
                )
                st.session_state.chat_history.append(
                    {'role': 'assistant', 'content': response}
                )
    except AssertionError as e:
        st.error("Error: Embedding dimensions do not match. Please re-index the data.")
        st.write(f"Details: {e}")
    except Exception as e:
        st.error("An unexpected error occurred while processing your query.")
        st.write(f"Details: {e}")
