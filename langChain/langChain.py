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
        length_function = len,
        is_separator_regex = False,
    )

    split_docs = text_splitter.split_documents(docs)
    print(len(split_docs))
    # print(split_docs[0].page_content)
    return split_docs

split_document(docs=load_document())