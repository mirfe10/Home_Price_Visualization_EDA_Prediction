# 🏠 İstanbul Konut Fiyat Tahmini

Bu projede İstanbul’daki konut ilan verileri kullanılarak makine öğrenmesi ile ev fiyat tahmini yapılmıştır.  
Amaç, ilan özelliklerine göre fiyatı tahmin edebilen bir regresyon modeli geliştirmektir.

---

## 📌 Proje Kapsamı

- Veri temizleme ve dönüştürme
- Feature engineering
- Farklı model karşılaştırmaları
- Performans analizi

---

## 🔎 Veri Ön İşleme

Projede uygulanan başlıca işlemler:

- Fiyat değişkenine **log dönüşümü (log_price)** uygulandı.
- `FloorLocation` metin verisi sayısala çevrilerek **FloorNumber** oluşturuldu.
- `address` değişkeninden mahalle bilgisi ayrıştırıldı.
- Eksik değerler:
  - Sayısal değişkenlerde **medyan**
  - Kategorik değişkenlerde **en sık değer**
  ile dolduruldu.
- Kategorik değişkenler:
  - Scikit-learn modellerinde **OneHotEncoding**
  - CatBoost modelinde **native categorical handling**
  ile işlendi.

---

## 🤖 Kullanılan Modeller

- Ridge Regression (Baseline)
- HistGradientBoostingRegressor
- CatBoostRegressor (Final Model)

---

## 📊 Final Model Performansı (CatBoost)

- **R² Score:** ~0.72  
- **MAE (log):** ~0.077  
- **MAE (TL):** ~149.000 TL  
- **Ortalama Yüzde Hata:** ~%17  

Model, fiyat varyansının yaklaşık %72’sini açıklayabilmektedir.

---

## 🛠 Kullanılan Teknolojiler

- Python
- Pandas & NumPy
- Scikit-learn
- CatBoost
- Matplotlib

---
