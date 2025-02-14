import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# -------------------------
# Caching Functions for Performance
# -------------------------

@st.cache_data(show_spinner=False)
def save_file(uploaded_file, storage_path):
    os.makedirs(storage_path, exist_ok=True)
    file_path = os.path.join(storage_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

@st.cache_data(show_spinner=True)
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

@st.cache_data(show_spinner=True)
def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(documents)

@st.cache_resource(show_spinner=True)
def get_vector_store(embedding_model):
    return InMemoryVectorStore(embedding_model)

@st.cache_data(show_spinner=True)
def index_documents(vector_store, document_chunks):
    vector_store.add_documents(document_chunks)
    return vector_store

# -------------------------
# Custom CSS for Visual Improvements
# -------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stSidebar { 
        background-color: #1E1E1E; 
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        color: #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A;
        border: 1px solid #404040;
        color: #F0F0F0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA;
        color: #000000;
    }
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Configurations and Model Initialization
# -------------------------
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# -------------------------
# Main UI Layout
# -------------------------
st.title(" RAG Powered Document AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis"
)

if uploaded_pdf:
    # Save and process the PDF file
    with st.spinner("Saving file..."):
        saved_path = save_file(uploaded_pdf, PDF_STORAGE_PATH)
    
    with st.spinner("Loading document..."):
        raw_docs = load_pdf(saved_path)
    
    with st.spinner("Splitting document into chunks..."):
        document_chunks = split_document(raw_docs)
    
    # Initialize vector store and index documents
    vector_store = get_vector_store(EMBEDDING_MODEL)
    vector_store = index_documents(vector_store, document_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    # Chat Input Section
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            # Retrieve relevant document chunks
            relevant_docs = vector_store.similarity_search(user_input)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create and invoke the LLM prompt chain
            prompt_input = {"user_query": user_input, "document_context": context_text}
            response_chain = conversation_prompt | LANGUAGE_MODEL
            answer = response_chain.invoke(prompt_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(answer)
