from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
import os
import openai
from dotenv import load_dotenv
import warnings
import streamlit as st
from streamlit_chat import message

warnings.filterwarnings("ignore")

# Load environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "chroma"

client = openai.OpenAI(api_key=openai_api_key)

@st.cache_resource
def load_chroma_db():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

@st.cache_resource
def create_qa_system(question):
    # Load the Chroma database
    db = load_chroma_db()

    # Initialize the LLM
    llm = OpenAI(api_key=openai_api_key)
    
    # Create the QA chain
    qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    
    # Generate the answer using the QA chain
    result = qa({"query": question})
    answer = result["result"]
    
    return answer

# Initialize session state for storing chat history and input
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Add user message to chat history
        st.session_state.history.append({"sender": "User", "message": user_input})

        # Get the best answer from the QA system
        answer = create_qa_system(user_input)
        st.session_state.history.append({"sender": "Bot", "message": answer})

        # Clear input field after sending the message
        st.session_state.user_input = ""

def main():
    # Create .streamlit directory if it doesn't exist
    if not os.path.exists('.streamlit'):
        os.makedirs('.streamlit')

    # Write the config.toml file
    with open('.streamlit/config.toml', 'w') as config_file:
        config_file.write("""
    [theme]
    base="light"
    """)
    st.set_page_config(
    page_title="Frankfurt University Chatbot",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="auto",
)
    # Add title, header, and logo
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://campuas.frankfurt-university.de/pluginfile.php/1/core_admin/logocompact/300x300/1712295011/logo.png" alt="University Logo" style="width: 150px; margin: auto;">
            <h1>Frankfurt University of Applied Sciences Chatbot</h1>
            <p>Ask your questions about the university and get answers instantly!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    # Custom CSS to style the chat interface
    st.markdown(
        """
        <style>
        .chat-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: white;
            padding: 10px;
            border-top: 1px solid #e6e6e6;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-input {
            width: 80%;
            display: inline-block;
        }
        .chat-button {
            width: 10%;
            display: inline-block;
        }
        .stButton button {
            width: 100%;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            text-align: center;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display chat history
    for i, chat in enumerate(st.session_state.history):
        if chat["sender"] == "User":
            message(chat["message"], is_user=True, key=f"user_{i}")
        else:
            message(chat["message"], key=f"bot_{i}")

    # Chat input and button
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    user_input = st.text_input("", key="user_input", label_visibility="collapsed", placeholder="Type your question here...", help="Press Enter or click Send to submit")
    st.button("Send", on_click=send_message, key="send_button")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
