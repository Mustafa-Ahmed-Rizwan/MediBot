import os
import chainlit as cl
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import time
import asyncio
from ui import format_response, format_sources

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# Preload embeddings model globally
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Global variable to store vectorstore
_vectorstore = None

async def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    
    try:
        _vectorstore = await asyncio.to_thread(
            FAISS.load_local,
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return _vectorstore
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=512,
            timeout=10,
            max_retries=2
        )
        return llm
    except Exception as e:
        raise Exception(f"Error initializing Groq LLM: {e}")

CUSTOM_PROMPT_TEMPLATE = """
You are MediBot, a highly knowledgeable and concise medical assistant. 
Answer the user's question using only the information provided in the context below. 
If the answer is not present in the context, respond with "I don't know based on the provided information." 
Do not fabricate or guess. Avoid small talk and focus on clarity and accuracy.

Context:
{context}

Question:
{question}

Your answer:
"""

@cl.on_chat_start
async def on_chat_start():
    # Initialize session state for messages
    cl.user_session.set("messages", [])
    
    # Load vector store
    vectorstore = await get_vectorstore()
    if vectorstore is None:
        await cl.Message(content="Failed to load the vector store").send()
        return

    # Store the vectorstore in session
    cl.user_session.set("vectorstore", vectorstore)
    
    # Welcome message with icon
    welcome_message = """
    🩺 **Welcome to MediBot!**  
    I'm here to answer your medical questions based on provided information.  
    Type your question below, and I'll respond with clear, accurate answers.  
    """
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def on_message(message: cl.Message):
    prompt = message.content
    if not prompt.strip():
        return

    # Store user message
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": prompt})
    cl.user_session.set("messages", messages)

    max_attempts = 3
    attempt = 1

    while attempt <= max_attempts:
        try:
            vectorstore = cl.user_session.get("vectorstore")
            if vectorstore is None:
                await cl.Message(content="Vector store not initialized. Please restart the chat.").send()
                return

            # Initialize QA chain only when needed
            if not cl.user_session.get("qa_chain"):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                cl.user_session.set("qa_chain", qa_chain)

            qa_chain = cl.user_session.get("qa_chain")
            response = await asyncio.to_thread(qa_chain.invoke, {'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Format response and sources
            formatted_response = format_response(result)
            formatted_sources = format_sources(source_documents)

            # Combine response and sources
            full_response = f"{formatted_response}\n\n{formatted_sources}"
            
            # Send response
            await cl.Message(content=full_response).send()
            messages.append({"role": "assistant", "content": result})
            cl.user_session.set("messages", messages)
            break

        except OutputParserException as e:
            await cl.Message(content=f"Error parsing output: {e}. Retrying... ({attempt}/{max_attempts})").send()
            time.sleep(2)
            attempt += 1
        except Exception as e:
            if "rate limit" in str(e).lower():
                await cl.Message(content=f"Rate limit error: {e}. Retrying in 5 seconds... ({attempt}/{max_attempts})").send()
                time.sleep(5)
                attempt += 1
            else:
                await cl.Message(content=f"Error: {str(e)}").send()
                break

    if attempt > max_attempts:
        await cl.Message(content="Max retry attempts reached. Please try again later or check your API key and model availability.").send()

@cl.action_callback("clear_chat")
async def on_clear_chat(action: cl.Action):
    cl.user_session.set("messages", [])
    await cl.Message(content="Chat history cleared!").send()