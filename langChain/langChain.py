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

# Say Hello to the model
def say_hello():
    model = load_model()
    ai_message = model.invoke('Hello')
    print(ai_message.content)

# Translate the text
def translate_text():
    model = load_model()
    from langchain_core.prompts import ChatPromptTemplate
    systemTemplate = "Translate the following text from English into {language}"
    promptTemplate = ChatPromptTemplate.from_messages(
        [("system", systemTemplate), ("user", "{text}")]
    )
    prompt = promptTemplate.invoke({"language": "Hindi with English script", "text": ai_message.content})
    ai_message = model.invoke(prompt)
    print(ai_message.content)
