import os
from fastapi import FastAPI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langsmith import Client

from typing import Dict
import uuid
import json

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
session = {
    "session_id": str(uuid.uuid4())
}

# Initialize Langsmith client and tracer
client = Client()
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT"))

# Setup callback manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), tracer])

# Dictionary untuk menyimpan memory untuk setiap session
conversation_memories: Dict[str, ConversationBufferMemory] = {}

# Setup Langchain Azure OpenAI Client
llm = AzureChatOpenAI(
    azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT'),
    openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY'),
    openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION'),
    deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT'),
    model_name = 'gpt-4o-mini',
    temperature = 0.4,
    max_tokens = 1500,
    max_retries = 1,
    # parallel_tool_calls=False,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's in a funny way."),
    ("user", "Question: {human_input}"),
])
# output_parser = StrOutputParser()
# chain = prompt | llm | output_parser

def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create memory for a session."""
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="human_input"  # Set input_key to 'human_input' as required
        )
    return conversation_memories[session_id]

def create_chain(prompt_template: PromptTemplate, memory: ConversationBufferMemory) -> LLMChain:
    """Create a LangChain chain with memory."""
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True,
        # callback_manager=callback_manager
    )

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def chat(human_input: str):
    try:
        # Get session ID or create new one
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Get or create memory for this session
        memory = get_or_create_memory(session_id)

        prompt_template = prompt
        if not prompt_template:
            return {"error": "Prompt template could not be loaded"}, 400
        
        chain = create_chain(prompt, memory)
        # input_dict = {"human_input": human_input}
        response = chain.predict(human_input=human_input, client=client)
        return {"response": response}
    except Exception as e: 
        return {"error": str(e)}, 500