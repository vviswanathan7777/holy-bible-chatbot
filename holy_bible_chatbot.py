from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    raise ImportError("`PyPDFLoader` requires `pypdf`. Please install it using `pip install pypdf`.")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure faiss-cpu and pypdf are installed
# pip install faiss-cpu pypdf

# Step 1: Load and preprocess the PDF
pdf_path = "The-Holy-Bible-King-James-Version.pdf"
db_name = "Holy_Bible_DB"

@st.cache_resource
def process_and_index_pdf(pdf_path, db_name):
    """Extracts text from the PDF, splits it into chunks, and builds the FAISS index."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and index them in FAISS
    embeddings = OllamaEmbeddings(
        model='granite3.1-dense:latest', base_url='http://localhost:11434'
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(db_name)

    return vector_store

# Initialize or rebuild FAISS index
try:
    vector_store = FAISS.load_local(
        db_name, embeddings=OllamaEmbeddings(
            model='granite3.1-dense:latest', base_url='http://localhost:11434'
        ), allow_dangerous_deserialization=True
    )
except Exception as e:
    st.warning("Indexing PDF file... This may take a few minutes.")
    st.write(f"Reindexing due to: {e}")
    vector_store = process_and_index_pdf(pdf_path, db_name)

# Step 2: Define Prompt and Chat History
prompt_template = """
    You are an assistant for question-answering tasks and an expert on the Holy Bible. Provide answers in concise bullet points (5 to 7) with sources cited.
    
    If the question is outside the scope of the Holy Bible, respond with: "I don't know".
    
    Question: {question}
    Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Step 3: Initialize ChatOllama and History
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

llm = ChatOllama(model='granite3.1-dense:latest', base_url='http://localhost:11434')

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
