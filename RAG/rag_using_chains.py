import os
import bs4
import json 

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'credentials.json'), 'r') as f:
    credentials = json.load(f)['credentials']
    os.environ["GOOGLE_API_KEY"] = credentials['GOOGLE_API_KEY']
    os.environ["LANGSMITH_API_KEY"] = credentials['LANGSMITH_API_KEY']
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


#### INDEXING ####
# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vector_store = Chroma.from_documents(documents=splits,
                                    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

#### RETRIEVAL and GENERATION ####
# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
response = rag_chain.invoke("What is Task Decomposition?")
print(response)



