import openai

# Configurar la clave de la API de OpenAI
openai.api_key = "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"

from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage
import os
import gradio as gr 
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI

# Configurar la clave de la API de OpenAI en la variable de entorno
os.environ['OPENAI_API_KEY'] = openai.api_key

def chatbot(question):
    # Cargar el índice entrenado desde el directorio "Modelo"
    storage_context = StorageContext.from_defaults(persist_dir='Modelo')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    
    # Realizar una consulta al índice con la pregunta ingresada y obtener la respuesta generada
    resonse = query_engine.query(question)

    return resonse

# Crear una interfaz gráfica para el chatbot utilizando Gradio
app = gr.Interface(fn=chatbot,
                   inputs= gr.inputs.Textbox(lines=5, label="Ingrese una peticion"),
                   outputs= "text",
                   title="Chatbot")

# Lanzar la interfaz gráfica
app.launch()