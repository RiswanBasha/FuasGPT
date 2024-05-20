# Frankfurt University of Applied Sciences Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Frankfurt University of Applied Sciences. The chatbot leverages natural language processing, a document retrieval system, and predefined responses to provide accurate and relevant answers to user queries.

## Features

- **Predefined Responses**: Quickly responds to basic and common questions with predefined answers.
- **Document Retrieval**: Uses FAISS (Facebook AI Similarity Search) to retrieve relevant documents and provide accurate answers.
- **Context Preservation**: Chunks large documents into manageable pieces while retaining context through overlap.
- **Source Attribution**: Provides sources for retrieved information to ensure credibility.

## How to Run the Code

1. **Clone the Repository**
   ```bash
   https://github.com/RiswanBasha/Search_Engine.git
   cd [folder]
   ```
2. **Set Up a Virtual Environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   or
    ```
   conda create --name myenv python=3.8
   conda activate myenv
   ```
3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. **Environment Variables**

   set the OPENAI_API_KEY in your .env file.
   1. Register your account
   2. Get API Key for pay as you go service as per the usage of tokens and embeddings
      
6. **Run the Application**

   ```
   python CreateVector_FAISS.py (for creating a vector database and storage embeddings)
   
   Streamlit run App.py 
   ```
