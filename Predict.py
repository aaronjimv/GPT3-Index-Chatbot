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

# Define the chatbot function
def chatbot(question):
    # Load the trained index from the "Modelo" directory
    storage_context = StorageContext.from_defaults(persist_dir='Modelo')
    index = load_index_from_storage(storage_context)
    
    # Create a query engine from the loaded index
    query_engine = index.as_query_engine()
    
    # Perform a query on the index with the entered question and get the generated response
    resonse = query_engine.query(question)

    # Return the response
    return resonse

# Create a graphical interface for the chatbot using Gradio
app = gr.Interface(fn=chatbot,
                   inputs= gr.inputs.Textbox(lines=5, label="Ingrese una peticion"),
                   outputs= "text",
                   title="Chatbot")

# Launch the graphical interface
app.launch()