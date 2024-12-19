
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define configuration model with Pydantic v2 compatibility
class Config(BaseModel):
    model_path: str = Field(default="llama-2-7b-chat.ggmlv3.q8_0.bin", description="Path to the model file")
    model_type: str = Field(default="llama", description="Model type, e.g., llama")
    max_new_tokens: int = Field(default=512, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=0.5, description="Sampling temperature for generation")
    embeddings_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Name of the embeddings model")
    db_faiss_path: str = Field(default="vectorstores/db_faiss", description="Path to the FAISS vector store")

    # Optional method to serialize configuration if needed
    def as_dict(self):
        return self.model_dump()

# Instantiate configuration
config = Config()

# Custom prompt template
custom_prompt_template = """
You are a helpful and knowledgeable assistant. Use the provided context to answer the user's question accurately.

Context: {context}
Question: {question}

If the context does not provide enough information or you are unsure of the answer, respond with, "I’m sorry, I don’t have the necessary information to answer that right now."

Please give a concise and clear response that addresses the user's question.

Answer:
"""

# Set custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Load the LLM model
def load_llm():
    try:
        llm = CTransformers(
            model=config.model_path,
            model_type=config.model_type,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            client=None
        )
        logging.info("Model loaded successfully.")
        return llm
    except Exception as e:
        logging.error(f"Error loading the model: {str(e)}")
        return None

# Define Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        return qa_chain
    except Exception as e:
        logging.error(f"Error initializing Retrieval QA Chain: {str(e)}")
        return None

# QA Model Function
def qa_bot():
    try:
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embeddings_model_name,
            model_kwargs={'device': 'cuda'}
        )

        # Load FAISS vector store
        db = FAISS.load_local(config.db_faiss_path, embeddings, allow_dangerous_deserialization=True)
        
        # Load the LLM
        llm = load_llm()
        if llm is None:
            raise ValueError("Failed to load LLM model.")
        
        # Set custom prompt
        qa_prompt = set_custom_prompt()

        # Create QA chain
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        if qa is None:
            raise ValueError("Failed to initialize QA chain.")
        
        return qa
    except Exception as e:
        logging.error(f"Error in QA bot setup: {str(e)}")
        return None

# Output function to handle final result
def final_result(query):
    qa_result = qa_bot()
    if qa_result is None:
        return "Error: Unable to process the query."
    response = qa_result({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    try:
        chain = qa_bot()
        if chain is None:
            raise RuntimeError("Failed to initialize QA bot.")
        
        # Set session data
        cl.user_session.set("chain", chain)

        msg = cl.Message(content="Starting the bot...")
        await msg.send()

        msg.content = "Hi, Welcome to Medical Bot. What is your query?"
        await msg.update()
        logging.info("Bot started successfully.")

    except Exception as e:
        logging.error(f"Error during bot start: {str(e)}")
        await cl.Message(content=f"Error starting the bot: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Retrieve session
        chain = cl.user_session.get("chain")
        if chain is None:
            raise ValueError("No active session found.")
        
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        # Invoke the QA chain
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res.get("source_documents", [])

        if sources:
            answer += f"\nSources: {', '.join([str(s) for s in sources])}"
        else:
            answer += "\nNo sources found."

        await cl.Message(content=answer).send()

    except Exception as e:
        logging.error(f"Error during message handling: {str(e)}")
        await cl.Message(content=f"Error: {str(e)}").send()
