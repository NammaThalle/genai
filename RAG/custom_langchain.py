import os
import json
from typing import List
from langchain_chroma import Chroma
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LangChain :
    def __init__(self):
        self.credentials = self._read_credentials()
        os.environ["GOOGLE_API_KEY"] = self.credentials['GOOGLE_API_KEY']
        os.environ["LANGSMITH_API_KEY"] = self.credentials['LANGSMITH_API_KEY']
        os.environ["LANGSMITH_TRACING"] = "true"

        print('Loading Model')
        self.model = self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        print('Loading Embeddings')
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print('Loading Text Splitter')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 200,
            add_start_index = True,
        )

        print('Loading Vector Store')
        self.vector_store = Chroma(embedding_function=self.embeddings)
        
        print('Initialized Langchain Module successfully')
    # Read credentials from credentials.json
    def _read_credentials(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'credentials.json'), 'r') as f:
            credentials = json.load(f)
        return credentials['credentials']

    # Load documents
    def load_document(self):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nke-10k-2023.pdf")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    # Query Vector Store
    def query_vector_store(self, query):
        results = self.vector_store.similarity_search(query)
        return results

    # Query using Retriever
    def query_using_retriever(self, queries):
        vector_store = self.create_vector_store()
        @chain
        def retriever(query:str) -> List[Document]:
            return vector_store.similarity_search(query, k=1)
        
        return retriever.batch(queries)

    # Query using as_retriever
    def query_using_as_retriever(self, queries):
        vector_store = self.create_vector_store()
        retriever = vector_store.as_retriever(
            search_type = "similarity", # similarity : default | mmr: maximum marginal relevance | similarity_score_threshold: threshold doc outputs by similarity score
            search_kwargs={"k": 1} # k: number of documents to return
        )
        return retriever.batch(queries)

if __name__ == "__main__":

    library = LangChain()

    # Ask user for a query
    # query = input("Enter your query: ")

    # results = library.query_vector_store(query)
    # print(f"Output: {results}")

    # Ask for 2 queries and store in a list
    queries = []
    print("Enter 2 queries: ")
    for i in range(2):
        queries.append(input(f"Enter query {i+1}: "))

    # results = library.query_using_retriever(queries)

    results = library.query_using_as_retriever(queries)
    print(results)