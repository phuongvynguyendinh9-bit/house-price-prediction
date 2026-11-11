import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="House Price Prediction App",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 HE THONG DU DOAN GIA NHA THONG MINH")
st.markdown("Ung dung su dung thuat toan **Random Forest Regressor** de du doan gia nha dua tren cac thue tinh dac trung.")

@st.cache_data
def load_and_train():
    try:
        df = pd.read_excel('nhom8_GiaNha_test.xlsx')
    except Exception:
        data = {
            'price': [379000, 375400, 474600, 943300, 588000, 420000, 550000, 610000, 480000, 520000],
            'bedrooms': [3, 3, 5, 2, 3, 3, 4, 4, 3, 3],
            'bathrooms': [2.5, 2.0, 3.0, 3.0, 1.0, 2.0, 2.5, 3.0, 2.0, 2.5],
            'sqft_living': [2192, 2175, 1662, 2157, 1538, 1800, 2300, 2500, 1900, 2100],
            'sqft_lot': [7191, 4268, 6000, 8868, 5348, 5000, 6000, 7000, 5500, 5800],
            'floors': [2.0, 2.0, 2.0, 1.0, 1.5, 1.0, 2.0, 2.0, 1.5, 2.0],
            'condition': [5, 3, 4, 4, 4, 3, 4, 5, 3, 4],
            'sqft_above': [2192, 1614, 1251, 1765, 1538, 1800, 2000, 2100, 1700, 1900],
            'sqft_basement': [0, 561, 411, 392, 0, 0, 300, 400, 200, 200],
            'yr_built': [1948, 2003, 1929, 1980, 1956, 1975, 1990, 2005, 1985, 1998]
        }
        df = pd.DataFrame(data)

    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above', 'sqft_basement', 'yr_built']
    X = df[features]
    y = df['price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features, df

model, features, df = load_and_train()

st.sidebar.header("🎛️ Nhap thong so ngoi nha")

input_bedrooms = st.sidebar.number_input("So phong ngu (bedrooms)", min_value=1, max_value=10, value=3, step=1)
input_bathrooms = st.sidebar.number_input("So phong tam (bathrooms)", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
input_sqft_living = st.sidebar.slider("Dien tich song (sqft_living)", 500, 10000, 2100, 50)
input_sqft_lot = st.sidebar.slider("Dien tich dat (sqft_lot)", 500, 20000, 5000, 100)
input_floors = st.sidebar.selectbox("So tang (floors)", [1.0, 1.5, 2.0, 2.5, 3.0], index=2)
input_condition = st.sidebar.slider("Tinh trang nha (condition: 1-5)", 1, 5, 4, 1)
input_sqft_above = st.sidebar.slider("Dien tich tren mat dat (sqft_above)", 500, 8000, 1800, 50)
input_sqft_basement = st.sidebar.slider("Dien tich tang ham (sqft_basement)", 0, 4000, 300, 50)
input_yr_built = st.sidebar.number_input("Nam xay dung (yr_built)", min_value=1900, max_value=2026, value=1995, step=1)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Ket qua du doan gia nha")
    user_data = pd.DataFrame([[
        input_bedrooms, input_bathrooms, input_sqft_living, input_sqft_lot,
        input_floors, input_condition, input_sqft_above, input_sqft_basement, input_yr_built
    ]], columns=features)
    
    predicted_price = model.predict(user_data)[0]
    
    st.success(f"### Gia nha uoc tinh: **${predicted_price:,.2f} USD**")
    st.info(f"Tuong duong khoang: **{predicted_price * 25400 / 1e9:,.2f} ty VND** *(Ty gia tham khao)*")

with col2:
    st.subheader("📋 Thong tin tap du lieu")
    st.metric(label="Tong so mau huan luyen", value=len(df))
    st.metric(label="Gia nha trung binh", value=f"${df['price'].mean():,.0f}")

st.markdown("---")
st.subheader("🔍 Xem truoc du lieu mau")
st.dataframe(df.head(10), use_container_width=True)
