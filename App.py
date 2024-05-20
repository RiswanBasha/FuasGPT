import os
import streamlit as st
import faiss
import pickle
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS
import warnings
import string

warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Frankfurt University Chatbot",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPEN_API_KEY")

# Initialize the LLM and embeddings only once using Streamlit's caching
@st.cache_resource
def initialize_llm_and_embeddings():
    llm = OpenAI(api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    return llm, embeddings

llm, embeddings = initialize_llm_and_embeddings()

# Load the FAISS index and other components only once using Streamlit's caching
@st.cache_resource
def load_faiss_components():
    index = faiss.read_index("faiss_index.bin")
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)
    with open("docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    return index, docs, index_to_docstore_id, docstore

index, docs, index_to_docstore_id, docstore = load_faiss_components()

# Initialize the FAISS vector store
vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Create the retriever and QA chain only once using Streamlit's caching
@st.cache_resource
def create_qa_chain():
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=retriever)
    return qa_chain

qa_chain = create_qa_chain()

def normalize_text(text):
    # Remove punctuation and convert to lowercase
    return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

# Function to get an answer from the QA system
def get_answer(question):
    # Predefined responses for basic questions
    basic_responses = {
        "hello": "Hi! How can I assist you today?",
        "hi": "Hello! How can I help you?",
        "how are you": "I'm a chatbot, so I don't have feelings, but I'm here to help you!",
        "what is your name": "I am the Frankfurt University of Applied Sciences Chatbot.",
        "whats your name": "I am the Frankfurt University of Applied Sciences Chatbot.",
        "what's your name": "I am the Frankfurt University of Applied Sciences Chatbot.",
        "what courses are offered": "Frankfurt University of Applied Sciences offers a variety of courses in engineering, business, social sciences, and more.",
        "how to apply": "You can apply through the university's online application portal. For detailed steps, visit the admissions page on our website.",
        "contact information": "You can contact us at info@fra-uas.de or call us at +49 69 1533-0.",
        "location": "Frankfurt University of Applied Sciences is located at Nibelungenplatz 1, 60318 Frankfurt am Main, Germany.",
        "tuition fees": "The tuition fees vary by program. Please visit the tuition fees page on our website for detailed information.",
        "admission requirements": "Admission requirements vary by program. Please check the specific requirements for your program on the university's website.",
        "scholarships": "The university offers various scholarships for students. Visit the scholarships page on our website for more information.",
        "library hours": "The library is open Monday to Friday from 8 AM to 8 PM, and Saturday from 9 AM to 4 PM.",
        "campus facilities": "The campus includes facilities such as a library, gym, cafeteria, and various student support services.",
        "housing options": "The university offers several housing options for students. Visit the housing page on our website for details.",
        "international students": "We welcome international students! Check out our international office page for information on admissions, visas, and support services.",
        "research opportunities": "The university offers various research opportunities for students and faculty. Visit the research page on our website for more details."
    }

    normalized_question = normalize_text(question)
    if normalized_question in basic_responses:
        return basic_responses[normalized_question], []

    # Get the best answer from the QA system
    result = qa_chain({"question": question})
    answer = result.get("answer", "Sorry, I couldn't find an answer.")
    sources = result.get("sources", [])
    return answer, sources
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
        answer, sources = get_answer(user_input)
        
        # Split sources by space, comma or newline and format them if sources exist
        if sources:
            # Split the sources string into a list of URLs
            sources_list = sources.replace(", ", " ").replace("\n", " ").split(" ")
            # Format each source as a list item
            sources_formatted = "".join([f'<li><a href="{source}" target="_blank">{source}</a></li>' for source in sources_list if source])
            response_message = f"{answer}<br><br><strong>Sources:</strong><ul>{sources_formatted}</ul>"
        else:
            response_message = answer
        
        st.session_state.history.append({"sender": "Bot", "message": response_message})

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
            background-color: #444444;
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
        .message {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .user-message {
            justify-content: flex-end;
        }
        .bot-message {
            justify-content: flex-start;
        }
        .message-text {
            max-width: 70%;
            padding: 10px;
            border-radius: 10px;
            color: white;
        }
        .user-message .message-text {
            background-color: #444444;
        }
        .bot-message .message-text {
            background-color: #000000;
        }
        .icon {
            margin: 0 10px;
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display chat history
    for i, chat in enumerate(st.session_state.history):
        if chat["sender"] == "User":
            st.markdown(f"""
                <div class="message user-message">
                    <div class="icon">👤</div>
                    <div class="message-text">{chat['message']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            bot_message = chat['message']
            st.markdown(
                '<div class="message bot-message">'
                '<div class="icon">🤖</div>'
                f'<div class="message-text">{bot_message}</div>'
                '</div>', 
                unsafe_allow_html=True
            )

    # Chat input and button
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    user_input = st.text_input("Your message", key="user_input", label_visibility="collapsed", placeholder="Type your question here...", help="Press Enter or click Send to submit")
    st.button("Send", on_click=send_message, key="send_button")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
