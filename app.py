import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import json

st.title("🏠 İstanbul Konut Fiyat Tahmini")




# -----------------------------
# Model + Feature Columns Load
# -----------------------------
@st.cache_resource
def load_model():
    m = CatBoostRegressor()
    m.load_model("catboost_house_price.cbm")
    return m

@st.cache_data
def load_feature_columns():
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
FEATURES = load_feature_columns()

# -----------------------------
# Location + Options Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HouseData.csv")


    # district
    df["district"] = df["district"].astype(str)

    # neighborhood: address içinden çıkar
    last = df["address"].astype(str).str.extract(r"'([^']+)'\]$")[0]
    df["neighborhood"] = last.str.replace(r"\s+Satılık.*$", "", regex=True)
    df["neighborhood"] = df["neighborhood"].fillna("Unknown").astype(str)

    # bazı kolonlar string olsun (selectbox için)
    for c in ["CreditEligibility", "HeatingType", "Category", "Type", "PriceStatus", "Swap", "InsideTheSite", "NumberOfRooms"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    return df



df_loc = load_data()

# -----------------------------
# UI - Selectboxes
# -----------------------------
district = st.selectbox("İlçe", sorted(df_loc["district"].dropna().unique()))

neighborhood = st.selectbox(
    "Mahalle",
    sorted(df_loc[df_loc["district"] == district]["neighborhood"].dropna().unique())
)

gross_m2 = st.number_input("Brüt m²", min_value=10, value=90)
net_m2 = st.number_input("Net m²", min_value=10, value=80)

# Oda sayısı (CSV'den)
rooms_options = sorted(df_loc["NumberOfRooms"].dropna().unique())
rooms = st.selectbox("Oda Sayısı", rooms_options, index=rooms_options.index("2+1") if "2+1" in rooms_options else 0)

# Banyo sayısı (basit dropdown)
bathroom_options = sorted(df_loc["NumberOfBathrooms"].dropna().unique())
bathrooms = st.selectbox("Banyo Sayısı", bathroom_options, index=0)

floor_num_options= [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "36",
]

floor_num = st.selectbox("Kat Numarası", floor_num_options, index=0)

building_age_ord = st.number_input("Bina Yaşı", min_value=0, value=10)

# Diğerleri CSV'den dropdown
inside_site_options = sorted(df_loc["InsideTheSite"].dropna().unique()) if "InsideTheSite" in df_loc.columns else ["Unknown"]
inside_site = st.selectbox("Site İçinde mi?", inside_site_options)


# -----------------------------
# Build input df EXACTLY like training columns
# -----------------------------
def build_input_df(user_values: dict) -> pd.DataFrame:
    row = {c: np.nan for c in FEATURES}

    for k, v in user_values.items():
        if k in row:
            row[k] = v

    df = pd.DataFrame([row])[FEATURES]  # order guaranteed

    # modelin kategorik kolonlarını bul
    cat_idx = model.get_cat_feature_indices()
    cat_cols = [FEATURES[i] for i in cat_idx]

    # kategorikler string, boşsa Unknown
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # diğer object kolonlar da string
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].fillna("Unknown").astype(str)

    # kategorik olmayanlar numeric
    num_cols = [c for c in FEATURES if c not in obj_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df

# -----------------------------
# Predict
# -----------------------------
if st.button("Tahmin Et"):
    user_dict = {
        "district": district,
        "neighborhood": neighborhood,
        "GrossSquareMeters": gross_m2,
        "NetSquareMeters": net_m2,
        "NumberOfRooms": rooms,
        "NumberOfBathrooms": bathrooms,
        "FloorNumber": int(floor_num),
        "BuildingAge_ordinal": building_age_ord,
        "InsideTheSite": inside_site,

    }

    input_df = build_input_df(user_dict)

    prediction_log = model.predict(input_df)[0]
    predicted_price = 10 ** prediction_log  # log10 kullandıysan

    st.success(f"Tahmini Fiyat: {predicted_price:,.0f} TL")
    
    st.markdown(
    """
    ⚠️ **Bilgilendirme:** Model geçmiş dönem ilan verileri ile eğitilmiştir.  
    Enflasyon ve piyasa koşullarındaki değişimler nedeniyle tahminler güncel değerlerden sapma gösterebilir.
    """
    )
    
    st.write("R² Skoru:", 0.72)
    st.write("Ortalama Hata (MAE): 149,720 TL")
    st.write("Ortalama Yüzde Hata: %17")



