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

