import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling for a modern dark mode with new emojis and improved visuals
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    /* Custom chat message styling */
    .chat-user {
        background-color: #ffffff;
        border: 1px solid #404040;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .chat-ai {
        background-color: #ffffff;
        border: 1px solid #505050;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("DeepSeek Code Companion 💻✨")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers & Creative Flair 🌟")

# Sidebar configuration for model selection, conversation style, and conversation reset
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    conversation_style = st.selectbox(
        "Conversation Style",
        ["Debug Mode 🐞", "Creative Mode 🎨", "Optimized Mode 🚀"],
        index=0
    )
    if st.button("Clear Conversation 🧹"):
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek 🤖. How can I help you code today? 💻"}]
        st.experimental_rerun()
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert  
    - 🐞 Debugging Assistant  
    - 📝 Code Documentation  
    - 💡 Solution Design  
    - 🎯 Code Optimization  
    - 🎨 Creative Problem Solving  
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the chat engine with the selected model and configuration
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration based on the chosen conversation style
if conversation_style == "Debug Mode 🐞":
    system_prompt_text = (
        "You are an expert AI coding assistant specialized in debugging. "
        "Provide concise, correct solutions with strategic print statements for debugging. "
        "Always respond in English."
    )
elif conversation_style == "Creative Mode 🎨":
    system_prompt_text = (
        "You are a creative AI coding assistant with a knack for innovative solutions. "
        "Provide detailed, imaginative answers that blend technical correctness with creative ideas. "
        "Always respond in English."
    )
else:  # Optimized Mode 🚀
    system_prompt_text = (
        "You are an expert AI coding assistant. Provide concise and optimized solutions with clear explanations. "
        "Always respond in English."
    )

system_prompt = SystemMessagePromptTemplate.from_template(system_prompt_text)

# Initialize session state for conversation if not already set
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek 🤖. How can I help you code today? 💻"}]

# Function to build the prompt chain from the conversation history
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Function to generate AI response using the prompt chain
def generate_ai_response(prompt_chain):
    pipeline = prompt_chain | llm_engine | StrOutputParser()
    return pipeline.invoke({})

# Chat container to display conversation history
chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-user'>🧑: {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>🤖: {message['content']}</div>", unsafe_allow_html=True)

# Chat input for new queries
user_query = st.chat_input("Type your coding question here...")

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.experimental_rerun()
