import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Get the API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

#define the directory containing the text file and the persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir, "books","doc1.pdf")
persistent_directory=os.path.join(current_dir, "db", "chroma_db")

#check if chroma vector store already exists 
if not os.path.exists(persistent_directory):
    print("Persisitent directory does not exist. Initializing vector store")

    #Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at {file_path}")
    
    #read the text content from the file
    loader= PyPDFLoader(file_path)
    documents=loader.load()

    #split the document into chucnks
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs=text_splitter.split_documents(documents)

    #display information about the split documents
    print("\n---Document chunks information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}\n")

    #create embeddings
    print("\n---Creating embeddings---")
    embeddings=OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n---Finished creating embeddings ---")

    #creating the vector store and persist it automatically
    print("\n ---Creating vector store  ---")
    db=Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persistent_directory
    )
    print("\n---Finished creating vector store ---")

else:
    print("vector store already exists. No need to initialize")    
    
    

    