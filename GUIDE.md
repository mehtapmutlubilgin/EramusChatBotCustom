Kodun Çalışma Kılavuzu 
Projeyi lokal makinenizde veya Colab'de çalıştırmak için aşağıdaki adımları izleyin:
1.Projeyi Klonlama:
Bash
git clone (https://github.com/mehtapmutlubilgin/EramusChatBotCustom)
cd EramusChatBotCustom
Sanal Ortam Kurulumu (Önerilen):
Bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
Bağımlılıkları Kurma:
Bash
pip install -r requirements.txt
API Anahtarını Ayarlama:
Bir .env dosyası oluşturun ve Gemini API anahtarınızı ekleyin: GEMINI_API_KEY="AIzaSy..."
Uygulamayı Başlatma:
Bash
streamlit run streamlit_app.py
Tarayıcınızda otomatik olarak Streamlit uygulamanız açılacaktır.

🌐 Web Arayüzü (Deploy Linki) 

Projemizin çalışan web arayüzüne aşağıdaki linkten erişebilirsiniz:

https://eramuschatbotcustom-mkyvbpfiuwised5jyzd5tj.streamlit.app/
