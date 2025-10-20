# streamlit_app.py

import streamlit as st
import os
import pandas as pd
import numpy as np
# Google GenAI'yi sadece Generation adımı için kullanıyoruz
from google import genai 
# Embedding (Vektörleme) ve Kosinüs Benzerliği için
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine 
from typing import List

# --- 1. SABİT TANIMLAMALAR ---
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
GENERATION_MODEL = "gemini-2.5-pro" 
TOP_K_DOCUMENTS = 2 # Retrieval adımında çekilecek en alakalı doküman sayısı

# --- 2. STREAMLIT PERFORMANS OPTİMİZASYONU ---
# @st.cache_resource, modelin ve embedding'lerin sadece bir kez yüklenmesini sağlar.
@st.cache_resource
def load_resources():
    st.info("Kaynaklar yükleniyor: Veri seti ve Embedding Modeli...")
    
    # Veri setini yükleme (Dosyanızın aynı dizinde olması gerekir)
    try:
        df = pd.read_csv("erasmus_dataset.csv") 
    except FileNotFoundError:
        st.error("HATA: 'erasmus_dataset.csv' dosyası bulunamadı. Lütfen GitHub reponuza ekleyin.")
        return None, None
    
    # Embedding Modelini yükleme
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Embedding'leri oluşturma
    documents = df['cevap'].tolist()
    st.info(f"Toplam {len(documents)} doküman için embedding hesaplanıyor...")
    document_embeddings = embedding_model.encode(documents)
    df['embedding'] = list(document_embeddings)

    st.success("Kaynaklar başarıyla yüklendi!")
    return df, embedding_model

# --- 3. RAG FONKSİYONLARI ---

def find_most_relevant_document(query: str, df: pd.DataFrame, embedding_model, top_k: int) -> str:
    """
    Soru için embedding oluşturur ve en benzer dokümanları bulur (Retrieval).
    """
    
    # Sorunun embedding'ini Sentence Transformer ile oluşturma
    query_embedding = embedding_model.encode(query)

    # Kosinüs benzerliği hesaplama
    similarities = [1 - cosine(query_embedding, doc_embedding) for doc_embedding in df['embedding']]
    df['similarity'] = similarities
    
    # En alakalı dokümanları alma
    relevant_docs = df.sort_values(by='similarity', ascending=False).head(top_k)
    
    # Context olarak birleştirme
    context = "\n\n---\n\n".join(relevant_docs['cevap'].tolist())
    return context


def generate_rag_answer(query: str, context: str) -> str:
    """
    Context ve soru ile Gemini modelinden cevap üretir (Generation).
    """
    
    # API Anahtarını Streamlit Secrets'tan güvenli bir şekilde oku
    try:
        # Streamlit Secrets'tan anahtarı al
        api_key = st.secrets["GEMINI_API_KEY"] 
        os.environ['GEMINI_API_KEY'] = api_key
        client = genai.Client()
    except KeyError:
        # Secrets'a anahtar eklenmezse bu hata gösterilir
        st.error("API Anahtarı hatası: Lütfen 'GEMINI_API_KEY' sırrını Streamlit Secrets'a ekleyin.")
        return "API Anahtarı bulunamadığı için cevap üretilemedi."

    # RAG Sistem Talimatı (System Instruction)
    system_instruction = (
        "Sen bir Erasmus programı bilgi asistanısın. Görevin, SADECE sağlanan 'CONTEXT' içerisindeki bilgileri kullanarak "
        "kullanıcının sorusunu doğru ve yardımcı bir şekilde cevaplamaktır. "
        "Eğer CONTEXT'te soruya dair bilgi yoksa, kibarca 'Bu konuda elimde kesin bir bilgi bulunmamaktadır.' diye yanıtla. "
        "Harici bir bilgi kullanma."
    )
    
    prompt = f"""
    CONTEXT (Bilgi Kaynağı):
    ---
    {context}
    ---
    KULLANICI SORUSU:
    {query}
    """
    
    # Gemini modelini çağırma
    response = client.models.generate_content(
        model=GENERATION_MODEL, 
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )
    
    return response.text

# --- 4. STREAMLIT ANA UYGULAMA MANTIĞI ---

# Kaynakları yükle
df, embedding_model = load_resources()

st.title("🎓 Akbank GenAI Projesi: Erasmus RAG Chatbot")
st.markdown("Bu chatbot, **Custom RAG** mimarisi kullanılarak Erasmus veri setine dayalı sorularınızı yanıtlar.")

if df is None or embedding_model is None:
    st.stop() 

# Kullanıcıdan metin girişi alma
user_query = st.text_input("Erasmus programı hakkında sorunuz nedir?", placeholder="Örn: Erasmus başvuru şartları ve süresi nedir?")

if user_query:
    with st.spinner(f"Bilgi Aranıyor ve Cevap Üretiliyor..."):
        
        # A. Retrieval (Geri Çekme)
        context_text = find_most_relevant_document(user_query, df, embedding_model, TOP_K_DOCUMENTS)
        
        # B. Generation (Cevap Üretme)
        final_answer = generate_rag_answer(user_query, context_text)
        
        # Sonucu gösterme
        st.success("🤖 CHATBOT CEVABI:")
        st.info(final_answer)
        
        # Kullanılan Context'i gösterme
        with st.expander("Kullanılan Bilgi Kaynağı (RAG Context)"):
            st.markdown(context_text)

st.divider()
st.caption("Çözüm Mimarisi: Sentence-Transformers (Embedding/Retrieval) + Gemini-2.5-Pro (Generation) üzerine kurulu Custom RAG.")
