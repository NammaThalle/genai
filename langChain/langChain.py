import getpass
import os

if not os.environ.get["GOOGLE_API_KEY"]:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_TRACING"] = "true"

# Load the model
def load_model():
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return model
