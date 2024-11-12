import streamlit as st
import whisper
from transformers import pipeline
import os
from dotenv import load_dotenv
import librosa
import numpy as np
from io import BytesIO
import ollama

load_dotenv()

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_hf_pipeline(api_token: str):
    return pipeline("summarization", model="facebook/bart-large-cnn")

def main():
    st.set_page_config(page_title="Trasncripcion y Resumen de audio", page_icon="🎶")
    
    whisper_model = load_whisper_model()
    llm_model = load_hf_pipeline(os.getenv("HF_TOKEN"))


    st.title("Aplicacion de Transcripcion y Analisis de Audio")

    if uploaded_file := st.file_uploader("Sube un archivo de audio", type=["mp3"]):

        audio_bytes = uploaded_file.read()

        audio_np, sr = librosa.load(BytesIO(audio_bytes), sr=16000)

        with st.spinner("Transcribiendo audio..."):
            transcription = whisper_model.transcribe(audio_np)["text"]
            st.markdown("## Transcripcion del Audio:")
            st.markdown(transcription)

        with st.spinner("Procesando el texto con el modelo."):
            response = llm_model(transcription, max_length=200, do_sample=True)
            
            st.markdown("## Respuesta del modelo:")
            st.markdown(response[0]["summary_text"])