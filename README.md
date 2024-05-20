# Frankfurt University of Applied Sciences Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Frankfurt University of Applied Sciences. The chatbot leverages natural language processing, a document retrieval system, and predefined responses to provide accurate and relevant answers to user queries.

## Features

- **Predefined Responses**: Quickly responds to basic and common questions with predefined answers.
- **Document Retrieval**: Uses FAISS (Facebook AI Similarity Search) to retrieve relevant documents and provide accurate answers.
- **Context Preservation**: Chunks large documents into manageable pieces while retaining context through overlap.
- **Source Attribution**: Provides sources for retrieved information to ensure credibility.
- **Web Scrapping from Sitemap**: From the sitemap, approximately 2500+ German links and 1400 English URLs were identified, with about 1099 valid URLs being used as the dataset for this project.


## How to Run the Code

1. **Clone the Repository**
   ```bash
   git clone https://github.com/RiswanBasha/FUAS_ChatBot.git
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

   Create a .env file in the project root and set the OPENAI_API_KEY.

   Steps to obtain the API key:
   1. Register an account on OpenAI.
   2. Get your API key for the pay-as-you-go service based on the usage of tokens and embeddings.

5. **Create FAISS Index and Document Store**

   Run the following script to create the FAISS vector database and store embeddings:
   ```
   python CreateVector_FAISS.py
   ```
      
6. **Run the Application**

   Start the Streamlit application:
   ```
   Streamlit run App.py 
   ```

## Contributing

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Open a Pull Request.

## Output

![Screenshot 2024-05-20 151429](https://github.com/RiswanBasha/FUAS_ChatBot/assets/52401793/996c3f31-9f12-4d54-9d85-500e9ee216ff)

![Screenshot 2024-05-20 151452](https://github.com/RiswanBasha/FUAS_ChatBot/assets/52401793/9e10f004-f3f9-4250-bfa1-193031f0f323)

![Screenshot 2024-05-20 151522](https://github.com/RiswanBasha/FUAS_ChatBot/assets/52401793/50dc8c01-0752-4562-b39e-042e0df0ec31)

![Screenshot 2024-05-20 151534](https://github.com/RiswanBasha/FUAS_ChatBot/assets/52401793/5c61679e-f63e-418d-b819-b3f440e474fe)

![Screenshot 2024-05-20 152407](https://github.com/RiswanBasha/FUAS_ChatBot/assets/52401793/f90d9e10-c239-4391-acd0-ad49b5a996ed)


