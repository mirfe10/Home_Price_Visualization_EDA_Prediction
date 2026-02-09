🏠 İstanbul Konut Fiyat Tahmini – Makine Öğrenmesi Projesi

Bu projede İstanbul’daki konut ilan verileri kullanılarak ev fiyat tahmini gerçekleştirilmiştir. Veri seti üzerinde kapsamlı veri temizleme, feature engineering ve model karşılaştırma çalışmaları yapılmıştır.

🔎 Veri Ön İşleme ve Dönüşümler

Projede aşağıdaki veri temizleme ve dönüştürme işlemleri uygulanmıştır:

Fiyat değişkenindeki çarpıklığı azaltmak amacıyla log dönüşümü (log_price) uygulanmıştır.

Metin formatındaki kat bilgileri (FloorLocation) sayısal formata dönüştürülerek FloorNumber değişkeni oluşturulmuştur.

address değişkeninden mahalle bilgisi ayrıştırılmıştır.

Kategorik değişkenler analiz edilerek uygun şekilde işlenmiştir.

Eksik değerler:

Sayısal değişkenlerde medyan

Kategorik değişkenlerde en sık görülen değer
ile doldurulmuştur.

Tarih değişkenleri düzenlenmiş ve model için uygun formata getirilmiştir.

Modelin kategorik değişkenleri işleyebilmesi için:

Scikit-learn modellerinde OneHotEncoding

CatBoost modelinde ise native categorical handling kullanılmıştır.

🤖 Modelleme Süreci

Farklı regresyon algoritmaları denenmiş ve performansları karşılaştırılmıştır:

Ridge Regression (Baseline Model)

HistGradientBoostingRegressor

CatBoostRegressor (Final Model)

Boosting tabanlı modeller doğrusal modellere kıyasla daha iyi performans göstermiştir.

📊 Final Model Sonuçları (CatBoost)

R² Score: ~0.72

MAE (Log): ~0.077

MAE (TL): ~149.000 TL

Ortalama Yüzde Hata: ~%17

Model, konut fiyat varyansının yaklaşık %72’sini açıklayabilmektedir.
