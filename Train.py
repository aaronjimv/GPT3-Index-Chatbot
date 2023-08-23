import openai

# Set up the OpenAI API key
openai.api_key = "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"

# Import libraries
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage
import os
import gradio as gr 
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI

# Set up the OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = openai.api_key

# Load training data from a directory named "datos"
docs = SimpleDirectoryReader("datos").load_data()

# Define parameters
max_input = 4098
tokens = 256
chnk_size = 600
max_chnk_overlap = 0.2

# Define the training function
def entrenamiento(path):
    # Create a PromptHelper to process requests
    Prompt_helper = PromptHelper(max_input, tokens, max_chnk_overlap, chunk_size_limit=chnk_size)
    
    # Create an LLMPredictor model with a pre-trained OpenAI model
    modelo = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
    
    # Create a service context using the model and the PromptHelper
    contexto = ServiceContext.from_defaults(llm_predictor=modelo, prompt_helper=Prompt_helper)
        
    # Train a GPTVectorStoreIndex using the documents and the service context
    index_model = GPTVectorStoreIndex.from_documents(docs,service_context=contexto)
    
    # Save the trained index to disk in a directory named "Modelo"
    index_model.storage_context.persist(persist_dir= 'Modelo') 

# Call the training function with the "datos" directory
entrenamiento("datos")