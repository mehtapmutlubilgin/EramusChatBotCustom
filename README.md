# 🎓 Akbank GenAI Bootcamp: Erasmus RAG Chatbot Projesi

## 1. Projenin Amacı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiştir.
Bu projenin temel amacı, RAG (Retrieval Augmented Generation) temelli bir chatbot geliştirerek, bu çözümü bir web arayüzü üzerinden kullanıcıya sunmaktır. Geliştirilen chatbot, Erasmus+ Programı hakkındaki soruları, Büyük Dil Modelinin (LLM) yeteneklerini özelleştirilmiş bilgi kaynağıyla birleştirerek, doğru, güvenilir ve bağlama uygun şekilde yanıtlamayı hedeflemektedir.

## 2. Veri Seti Hakkında Bilgi

* **İçerik:** Projede kullanılan veri seti (`erasmus_dataset.csv`), Erasmus+ Programı ile ilgili sık sorulan soruları (SSS) ve bunların detaylı cevaplarını içeren küçük, temiz ve yapılandırılmış bir veri kaynağıdır.
* **Rolü:** Chatbot'un bilgi kaynağı (Knowledge Base) olarak görev yapmış, `cevap` sütunundaki metinler vektörleştirilerek RAG sistemine dahil edilmiştir.

## 3. Kullanılan Yöntemler ve Çözüm Mimarisi

Bu projede, LangChain veya Haystack gibi RAG framework'leri yerine, yüksek stabilite ve tam kontrol sağlamak amacıyla temel kütüphanelerle oluşturulmuş **Custom RAG Pipeline** (Özel RAG Akışı) kullanılmıştır.

### RAG Mimarisi Adımları

| Bileşen | Kullanılan Teknoloji | Görev |
| :--- | :--- | :--- |
| **Embedding (Vektörleme)** | **Sentence-Transformers** (`paraphrase-multilingual-mpnet-base-v2`) | Veri setindeki cevap metinlerini sayısal vektörlere dönüştürme ve anlamı temsil eden vektörleri oluşturma. |
| **Vektör Arama (Retrieval)** | **NumPy / SciPy** (Kosinüs Benzerliği) | Kullanıcı sorgusu vektörünün en alakalı bilgi parçalarını (Context) bulma. *Küçük veri seti nedeniyle in-memory (bellek içi) depolama kullanılmıştır.* |
| **Generation (Üretme)** | [cite_start]**Google Gemini 2.5 Pro** [cite: 42] | Geri çekilen Context'i kullanarak nihai, doğru ve bağlama uygun cevabı üretme. |
| **Web Arayüzü** | **Streamlit** | [cite_start]Geliştirilen chatbot çözümünü bir web uygulaması olarak sunma[cite: 2]. |

### Teknik Detaylar

* **Embedding Seçimi:** Açık kaynaklı Sentence-Transformers modelinin seçilmesi, Gemini'ın kendi Embedding model API'ı ile yaşanan sürüm uyumluluk sorunlarını ortadan kaldırmıştır. Çok dilli modeli sayesinde, Türkçe bağlamı daha doğru anlamlandırmaktadır.
* **Depolama Çözümü:** Milyonlarca doküman için tasarlanan Chroma veya FAISS gibi Vektör Veritabanları yerine, veri setinin küçük olması nedeniyle Pandas DataFrame'ler in-memory (bellek içi) Vektör Deposu olarak kullanılmış, bu da deploy sürecini basitleştirmiştir.
* **LLM Kullanımı:** Gemini 2.5 Pro modeli, RAG akışında, yalnızca Retrieval aşamasında çekilen Context'i kullanması yönündeki kesin sistem talimatlarına (System Instruction) uyması için optimize edilmiştir.

#

## 4. Elde Edilen Sonuçlar Özeti

Geliştirilen RAG sistemi, Erasmus veri setindeki bilgilere dayanarak başarılı ve hızlı cevaplar üretebilmiştir:

* **Doğruluk:** Chatbot, Gemini 2.5 Pro'nun yönlendirici talimatları sayesinde, yalnızca sağlanan **Context** ile sınırlı kalmış ve uydurma (halüsinasyon) yapma riski minimize edilmiştir.
* **Stabilite:** Embedding adımının yerel bir modelle (Sentence-Transformers) çözülmesi, API uyumluluk hatalarını ortadan kaldırmıştır.
* **Kullanılabilirlik:** Streamlit ile yayımlanan web arayüzü sayesinde, son kullanıcı deneyimi için basit ve etkili bir arayüz sağlanmıştır.

## 5. Çalışma Kılavuzu

Projenin kurulum ve çalıştırma adımları için detaylı rehbere buradan ulaşabilirsiniz: [GUIDE.md](GUIDE.md)

## 6. Web Arayüzü ve Canlı Deneyim

Projemizin çalışan canlı versiyonunu buradan test edebilir ve projenin kabiliyetlerini deneyimleyebilirsiniz.

**Canlı Uygulama Linki (Deploy Link):**

https://eramuschatbotcustom-mkyvbpfiuwised5jyzd5tj.streamlit.app/
