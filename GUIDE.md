# 🎓 Erasmus RAG Chatbot Kullanım Kılavuzu

Bu kılavuz, **Custom RAG** mimarisiyle geliştirilen Erasmus Chatbot projesinin kodunun çalıştırılmasına ve web arayüzünün test edilmesine dair adımları içerir.

## A. Kodun Çalışma Kılavuzu (Lokal/Geliştirme Ortamı) 

Projenin kodunu çalıştırmak için gerekli adımlar şunlardır:

1.  **Projeyi Klonlama:**
    ```bash
    git clone https://github.com/mehtapmutlubilgin/EramusChatBotCustom
    cd EramusChatBotCustom
    ```

2.  **Sanal Ortam Kurulumu:** 
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac için
    # venv\Scripts\activate   # Windows için
    ```
   *(Bu adım, gereksinimlerin ana Python ortamınıza karışmasını önler.)*

3.  **Bağımlılıkları Yükleme:**
    ```bash
    pip install -r requirements.txt
    ```
   *(Bu komut; streamlit, google-genai, sentence-transformers gibi tüm gerekli kütüphaneleri kurar.)* 

4.  **Veri Seti Kontrolü:**
    *`erasmus_dataset.csv` dosyasının projenin ana dizininde bulunduğundan emin olun. 

5.  **API Anahtarını Hazırlama:**
    * Projeyi lokalde çalıştıracaksanız, `GEMINI_API_KEY="..."` formatında bir `.env` dosyası oluşturun.
    *`streamlit_app.py` dosyası bu anahtarı kullanarak Gemini 2.5 Pro modeline erişecektir. 

6. **Uygulamayı Başlatma:** 
    ```bash
    streamlit run streamlit_app.py
    ```
    *(Uygulama, tarayıcınızda otomatik olarak açılacaktır.)*

## B. Ürün Kılavuzu (Deploy Edilmiş Arayüz) 

Deploy linki üzerinden eriştiğiniz web arayüzünde sizi bekleyen çalışma akışı aşağıdadır: 

1.  **Giriş:**
    *Sayfa açıldığında, Streamlit'in `st.cache_resource` mekanizması devreye girer ve **Sentence-Transformers** modelini yükleyerek Erasmus veri setindeki tüm cevapların vektörlerini (embedding'lerini) bellekte hesaplar.  Bu adım sadece ilk çalıştırmada zaman alır.

2.  **Soru Girişi:**
    * "Erasmus programı hakkında sorunuz nedir?" etiketli metin kutusuna sorunuzu yazın.

3.  **RAG Akışı:**
    *Sorunuzu gönderdiğinizde, sistem şu adımları izler (genellikle 1-3 saniye): 
        * **Retrieval (Geri Çekme):** Sorunuz vektörleştirilir ve veri setindeki en alakalı 2 bilgi parçası (**Context**) kosinüs benzerliği ile bulunur.
        ***Generation (Üretme):** Bulunan Context ve sorunuz, **Gemini 2.5 Pro** modeline talimatlarla birlikte gönderilir.  Model, **sadece** bu Context'i kullanarak nihai cevabı üretir.

4. **Sonuçların Görselleştirilmesi:** 
    * **Chatbot Cevabı:** Gemini tarafından üretilen nihai, net cevap gösterilir.
    * **Context'i Görün:** Cevabın hemen altında yer alan "Kullanılan Bilgi Kaynağı (RAG Context)" alanını genişleterek, chatbot'un cevabı üretmek için hangi bilgi parçalarını kullandığını (yani Retrieval adımının sonucunu) görebilirsiniz. Bu, sistemin şeffaflığını kanıtlar.

###Test Senaryoları (Projenin Kabiliyetlerini Test Etme)

Aşağıdaki soruları sorarak RAG sisteminin doğruluğunu test edebilirsiniz:

| Test Senaryosu | Beklenen Sonuç |
| :--- | :--- |
| "Erasmus nedir?" | Erasmusun tanımı |
| "Hangi belgeler gerekir?" |Başvuru formu, transkript, dil sertifikası ve motivasyon mektubunun listelenmesi.
| "Süre ne kadar ve başvuru için dil şartı var mı?" | Birden fazla konuyu kapsayan (süre ve dil şartı) iki ayrı cevabın Context'e çekilip tek bir bütünleşik cevap verilmesi. |

***

**Projenizin Web Linki:**

https://eramuschatbotcustom-mkyvbpfiuwised5jyzd5tj.streamlit.app/

**Örnek Kullanımlar:**

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/87d3ff20-c714-435d-97e7-8b1652b4d920" />
<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/9360e9f0-2779-4307-bdc3-3854c185a8e1" />
<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/4fc04847-5952-4bb7-8f49-229c2af6ae86" />




