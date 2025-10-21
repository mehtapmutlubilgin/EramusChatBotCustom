🎓 Erasmus RAG Chatbot Projesi
Projenin Amacı 
Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş ve RAG (Retrieval Augmented Generation) mimarisini temel alan bir chatbot çözümüdür.
Amacımız, kullanıcıların Erasmus+ Programı hakkındaki sorularını, geniş bir dil modelinin (Gemini Pro) yeteneklerini özelleştirilmiş bir bilgi kaynağıyla birleştirerek , doğru, güvenilir ve bağlama uygun şekilde yanıtlayabilen bir web arayüzü sunmaktır.

Veri Seti Hakkında Bilgi 

Bu projede kullanılan veri seti, Erasmus+ Programı ile ilgili sık sorulan soruları (SSS) ve bunların detaylı cevaplarını içeren küçük, temiz ve yapılandırılmış bir veri kaynağıdır.

Veri Seti Adı: erasmus_dataset.csv

İçerik: Öğrenci ve akademik konuları kapsayan Erasmus programı hakkında soru-cevap çiftleri.

Rolü: Chatbot'un bilgi kaynağı (Knowledge Base) olarak görev yapar.

Kullanılan Yöntemler (Çözüm Mimarisi Özeti) 

Proje, LangChain/Haystack gibi hazır framework'ler yerine, maksimum esneklik ve stabilite sağlamak amacıyla temel Python kütüphaneleri kullanılarak oluşturulmuş Custom RAG Pipeline (Özel RAG Akışı) mimarisini kullanır.

Embedding (Vektörleme): Veri setindeki tüm cevap metinleri, yüksek performanslı Sentence-Transformers (paraphrase-multilingual-mpnet-base-v2) açık kaynak modeli  kullanılarak sayısal vektörlere dönüştürülür.

Vektör Depolama & Arama (Retrieval): Bu küçük veri seti için NumPy dizileri ve Pandas DataFrame'ler in-memory (bellek içi) Vektör Veritabanı görevi görür. Kullanıcının sorgusu vektörleştirildikten sonra, en alakalı bilgi parçalarını bulmak için SciPy kütüphanesi ile Kosinüs Benzerliği hesaplanır.

Cevap Üretimi (Generation): Geri çekilen ilgili bilgi parçaları (context), güçlü ve yetenekli bir Büyük Dil Modeli olan Google Gemini 2.5 Pro  modeline gönderilir. Modele, yalnızca sağlanan bağlamı kullanması talimatı verilerek nihai cevap üretilir.

Elde Edilen Sonuçlar Özeti 

Geliştirilen RAG chatbot, Erasmus veri setine özel karmaşık soruları bile doğru bağlamı geri çekerek yanıtlayabilmektedir. Sistemin temel başarısı şunlardır:

Doğruluk: Sisteme beslenen bilgilere sadık kalarak, LLM'in uydurma (halüsinasyon) yapma riski minimize edilmiştir.

Bağlam Güvenilirliği: Kullanıcının sorusuyla ilgili en alakalı bilgiyi geri çekme (Retrieval) oranı oldukça yüksektir.

Hız: Sentence-Transformers'ın yerel kullanımı sayesinde, RAG akışı saniyeler içinde tamamlanmaktadır.
