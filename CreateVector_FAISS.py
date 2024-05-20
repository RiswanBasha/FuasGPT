import os
import pandas as pd
from dotenv import load_dotenv
import warnings
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import pickle

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPEN_API_KEY")

# Initialize OpenAI LLM and Embeddings
llm = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load the CSV file
df = pd.read_csv("scrapped_data.csv")

# Extract valid URLs
urls = [i for i in df['URL'] if i.startswith('http')]

# Fetch content from URLs
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create FAISS vector store from documents
vector_db = FAISS.from_documents(docs, embeddings)

# Save the FAISS index to a file
faiss.write_index(vector_db.index, "faiss_index.bin")

# Save the documents and their IDs to files
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("index_to_docstore_id.pkl", "wb") as f:
    pickle.dump(vector_db.index_to_docstore_id, f)
with open("docstore.pkl", "wb") as f:
    pickle.dump(vector_db.docstore, f)
