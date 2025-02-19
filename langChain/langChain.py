import os
import json
import getpass

# Read credentials from credentials.json
def read_credentials():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'credentials.json'), 'r') as f:
        credentials = json.load(f)
    return credentials['credentials']

credentials = read_credentials()

os.environ["GOOGLE_API_KEY"] = credentials['GOOGLE_API_KEY']
os.environ["LANGSMITH_API_KEY"] = credentials['LANGSMITH_API_KEY']
os.environ["LANGSMITH_TRACING"] = "true"

# Load the model
def load_model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return model

# Say Hello to the model
def say_hello():
    model = load_model()
    ai_message = model.invoke('Hello')
    
    return ai_message.content

# Translate the text
def translate_text():
    model = load_model()
    from langchain_core.prompts import ChatPromptTemplate
    systemTemplate = "Translate the following text from English into {language}"
    promptTemplate = ChatPromptTemplate.from_messages(
        [("system", systemTemplate), ("user", "{text}")]
    )
    prompt = promptTemplate.invoke({"language": "Hindi with English script", "text": say_hello()})
    ai_message = model.invoke(prompt)
    print(ai_message.content)

# class LangChainDocuments:

# Load documents
def load_document():
    from langchain_community.document_loaders import PyPDFLoader
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nke-10k-2023.pdf")
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    
    # print(len(docs))
    # print(f"{docs[0].page_content[:200]}\n")
    # print(docs[0].metadata)

    return docs

# Split document
def split_document(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        add_start_index = True,
        # length_function = len,
        # is_separator_regex = False,
    )

    split_docs = text_splitter.split_documents(docs)
    
    # print(len(split_docs))
    # print(split_docs[0].page_content)
    
    return split_docs

# Create Embeddings
def create_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_splits = split_document(load_document())

    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    # print(f"Generated vectors of length {len(vector_1)}\n")

    return embeddings, all_splits

# Create Vector Store
def create_vector_store():
    embeddings, all_splits = create_embeddings()

    from langchain_chroma import Chroma

    vector_store = Chroma(embedding_function=embeddings)

    ids = vector_store.add_documents(documents=all_splits)

    return vector_store

# Query Vector Store
def query_vector_store(query):
    vector_store = create_vector_store()

    results = vector_store.similarity_search(query)

    return results[0]

if __name__ == "__main__":
    # Ask user for a query
    query = input("Enter your query: ")

    results = query_vector_store(query)
    print(f"Output: {results}")