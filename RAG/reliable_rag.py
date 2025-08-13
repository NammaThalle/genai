import os
import json
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

class GradeDocument(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """
    score: int = Field(
        ...,
        description="Binary score for relevance check on retrieved documents. 1 for relevant, 0 for not relevant."
    )

class GradeHallucination(BaseModel):
    """
    Binary score for hallucination check on generated responses.
    """
    score: int = Field(
        ...,
        description="Answer is grounded in the retrieved documents. 1 for grounded, 0 for not grounded."
    )

def read_credentials(file_path='../credentials.json'):
    with open(file_path, 'r') as f:
        credentials = json.load(f)['credentials']
    try:
        os.environ["GOOGLE_API_KEY"] = credentials['GOOGLE_API_KEY']
        os.environ["LANGSMITH_API_KEY"] = credentials['LANGSMITH_API_KEY']
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    except KeyError as e:
        raise ValueError(f"Missing key in credentials: {e}")

def get_retrieval_grader():
        """
        Returns a retrieval grader chain for assessing document relevance.
        """

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.1,
        )

        grader_llm = llm.with_structured_output(GradeDocument)

        grade_system_prompt = """
            You are a grader assessing relevance of retrieved documents for a given question.
            If the document contains keyword(s) or semantic meaning related to the question, return a score of 1 (relevant).
            If the document does not contain relevant information, return a score of 0 (not relevant).
            The score should be binary: 1 for relevant, 0 for not relevant.
            The goal is to determine if the document is relevant to the question.
            Do not provide any additional comments or explanations.
        """

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", grade_system_prompt),
                ("human", "Retrieved document: \n\n{document}\n\nQuestion: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | grader_llm
        return retrieval_grader

def generate_rag_response(docs_to_use, question):
    """
    Generates a response to the question using the provided relevant documents.
    """
    response_system_prompt = """
        You are an AI assistant for question-answering tasks. 
        Answer the question based on the provided documents.
        Use three-to-five sentences unless stated otherwise by the human to provide a concise and informative answer.
        If the documents do not contain relevant information, respond with "I don't know."
    """

    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", response_system_prompt),
            ("human", "Retrieved documents: \n\n{documents}\n\nQuestion: {question}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    rag_chain = response_prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n".join(
            f"<doc{i+1}>:\nTitle:{doc.metadata.get('title', '')}\nSource:{doc.metadata.get('source', '')}\nContent:{doc.page_content}\n</doc{i+1}>\n"
            for i, doc in enumerate(docs)
        )

    response = rag_chain.invoke({"documents": format_docs(docs_to_use), "question": question})

    # print(f"\nQuestion: {question}")
    # print(f"Response: {response}\n")
    # print(f"Number of relevant documents used: {len(docs_to_use)}")

    return response

def hallucination_grader(documents, llm_response):
    """
    Returns a hallucination grader chain for assessing if the response is grounded in the retrieved documents.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    llm = llm.with_structured_output(GradeHallucination)

    grade_system_prompt = """
        You are a grader assessing if the LLM response is grounded/supported in the retrieved documents.
        If the response is grounded in the retrieved documents, return a score of 1 (grounded).
        If the response is not grounded in the retrieved documents, return a score of 0 (not grounded).
        The score should be binary: 1 for grounded, 0 for not grounded.
        Do not provide any additional comments or explanations.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grade_system_prompt),
            ("human", "Retrieved documents: \n\n{documents}\n\nLLM Response: {response}")
        ]
    )

    hallucination_grader = grade_prompt | llm

    response = hallucination_grader.invoke({"documents": documents, "response": llm_response})

    if response.score: # type: ignore
        return llm_response 
    else:
        return "I don't know"

# Example usage
if __name__ == "__main__":
    creds = read_credentials()

    # Docs to index
    urls = [
        "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
    ]

    # Load urls
    docs = [WebBaseLoader(url).load() for url in urls]
    # Flatten the list of documents
    docs_list = [doc for sublist in docs for doc in sublist]
    # print(f"Number of documents loaded: {len(docs_list)}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(docs_list)

    # Using the latest HuggingFace embedding model (as of August 2025)
    # Example: 'sentence-transformers/all-MiniLM-L12-v2' is newer and widely used
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    # Create a FAISS vector store
    vector_store = FAISS.from_documents(docs_split, embedding_model)

    # # Save the vector store to disk
    # vector_store.save_local("faiss_index")

    # # Load the vector store from disk
    # vector_store = FAISS.load_local("faiss_index", embedding_model)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    while True:
        # Get user input
        print("\n" + "="*50)
        print("RAG Question-Answering System")
        print("Enter 'quit' to exit")
        print("="*50)
        
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the RAG system!")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue

        retrieved_docs = retriever.invoke(question)

        # Initialize the retrieval grader
        retrieval_grader = get_retrieval_grader()

        # Filter documents based on relevance using the retrieval grader
        docs_to_use = [doc for doc in retrieved_docs if retrieval_grader.invoke({"document": doc.page_content, "question": question}).score == 1]  # type: ignore
        
        # Fallback if no relevant documents found
        if not docs_to_use:
            print("Response: I don't know - no relevant documents found.")
            continue
        
        # Generate the response using the relevant documents
        llm_response = generate_rag_response(docs_to_use, question)

        # Initialize the hallucination grader
        hallucination_grader_response = hallucination_grader(docs_to_use, llm_response)

        print(f"Response: {hallucination_grader_response}")