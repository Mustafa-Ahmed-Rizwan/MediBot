import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import time
from langchain_core.exceptions import OutputParserException

load_dotenv()

# Step 1: Setup LLM (Groq with ChatGroq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


def load_llm():
    try:
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=512,
            timeout=10,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        raise

# Step 2: Connect LLM with Memory (FAISS) and Create Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly and make sure the reader understands the answer like explain in simple,easy words. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Invoke with a single query and handle errors
user_query = input("Write Query Here: ")
max_attempts = 3
attempt = 1

while attempt <= max_attempts:
    try:
        response = qa_chain.invoke({'query': user_query})
        print("RESULT: ", response["result"])
        print("SOURCE DOCUMENTS: ", response["source_documents"])
        break
    except OutputParserException as e:
        print(f"Error parsing output: {e}. Retrying... ({attempt}/{max_attempts})")
        time.sleep(2)  # Wait before retrying
        attempt += 1
    except Exception as e:
        if "rate limit" in str(e).lower():
            print(f"Rate limit error: {e}. Retrying in 5 seconds... ({attempt}/{max_attempts})")
            time.sleep(5)
            attempt += 1
        else:
            print(f"Unexpected error: {e}")
            break

if attempt > max_attempts:
    print("Max retry attempts reached. Please try again later or check your API key and model availability.")
