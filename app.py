# app.py dosyası

import streamlit as st
import os
import pandas as pd
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# --- 1. SABİT TANIMLAMALAR ---
# Bu kısımları Colab'den kopyalayın
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GENERATION_MODEL = "gemini-2.5-pro" 

# Model ve DataFrame'i Cache'leme (Streamlit performans optimizasyonu)
@st.cache_resource
def load_resources():
    # Veri setini yükleme
    df = pd.read_csv("erasmus_dataset.csv") 
    
    # Embedding Modelini yükleme
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Embedding'leri oluşturma (Bu kısım Colab'den gelmeli, 
    # veya verinizde zaten 'embedding' sütunu olmalı)
    documents = df['cevap'].tolist()
    document_embeddings = embedding_model.encode(documents)
    df['embedding'] = list(document_embeddings)

    return df, embedding_model

df, embedding_model = load_resources()

# --- 2. RAG FONKSİYONLARI (Hücre 3'ten kopyala) ---

def find_most_relevant_document(query: str, df: pd.DataFrame, top_k: int = 1) -> str:
    # ... (Hücre 3'teki kod) ...
    pass 

def generate_rag_answer(query: str, context: str) -> str:
    # API Anahtarını Streamlit Secrets'tan al
    api_key = st.secrets["GEMINI_API_KEY"] 
    os.environ['GEMINI_API_KEY'] = api_key
    client = genai.Client()
    
    # ... (Hücre 3'teki kodun geri kalanı) ...
    pass

# --- 3. STREAMLIT ARAYÜZÜ ---

st.title("🎓 Erasmus RAG Chatbot")
st.markdown("Bu chatbot, Erasmus veri setine dayalı sorularınızı Gemini Pro kullanarak yanıtlar.")

# Kullanıcıdan API anahtarı yerine, Streamlit Cloud'un Secret yönetimini kullanıyoruz
if not "GEMINI_API_KEY" in st.secrets:
    st.error("Lütfen Gemini API anahtarını Streamlit Secrets'a ekleyin.")
else:
    # Kullanıcıdan metin girişi alma
    user_query = st.text_input("Erasmus programı hakkında sorunuz nedir?", key="query_input")

    if user_query:
        with st.spinner("Bilgi aranıyor ve cevap üretiliyor..."):
            # A. Retrieval (Geri Çekme)
            context_text = find_most_relevant_document(user_query, df, top_k=2)
            
            # B. Generation (Cevap Üretme)
            final_answer = generate_rag_answer(user_query, context_text)
            
            # Sonucu gösterme
            st.success("Chatbot Cevabı:")
            st.write(final_answer)
            
            with st.expander("Kullanılan Bilgi Kaynağı (Context)"):
                st.markdown(context_text)
