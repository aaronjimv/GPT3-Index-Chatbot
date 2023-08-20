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

# Cargar datos de entrenamiento desde un directorio llamado "datos"
docs = SimpleDirectoryReader("datos").load_data()

max_input = 4098
tokens = 256
chnk_size = 600
max_chnk_overlap = 0.2

def entrenamiento(path):
    # Crear un PromptHelper para procesar las solicitudes
    Prompt_helper = PromptHelper(max_input, tokens, max_chnk_overlap, chunk_size_limit=chnk_size)
    
    # Crear un modelo de LLMPredictor con un modelo de OpenAI preentrenado
    modelo = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
    
    # Crear un contexto de servicio utilizando el modelo y el PromptHelper
    contexto = ServiceContext.from_defaults(llm_predictor=modelo, prompt_helper=Prompt_helper)
        
    # Entrenar un índice GPTVectorStoreIndex utilizando los documentos y el contexto del servicio
    index_model = GPTVectorStoreIndex.from_documents(docs,service_context=contexto)
    
    # Guardar el índice entrenado en disco en un directorio llamado "Modelo"
    index_model.storage_context.persist(persist_dir= 'Modelo') 

# Llamar a la función de entrenamiento con el directorio de datos "datos"
entrenamiento("datos")