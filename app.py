import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cấu hình trang
st.set_page_config(
    page_title="Dự Đoán Giá Nhà & Phân Cụm BĐS",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 HỆ THỐNG DỰ ĐOÁN GIÁ NHÀ & PHÂN CỤM BĐS")
st.markdown("Đồ án nhóm kết hợp 2 thuật toán: **Cây quyết định (Decision Tree)** (Dự đoán giá) và **K-Means** (Phân khúc bất động sản).")

# 1. Đọc dữ liệu và huấn luyện mô hình
@st.cache_data
def load_and_train_models():
    try:
        df = pd.read_excel('nhom8_GiaNha_test.xlsx')
    except Exception:
        # Dữ liệu dự phòng nếu file chưa sẵn sàng
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

    # Huấn luyện mô hình Cây Quyết Định (Decision Tree Regressor)
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X, y)

    # Huấn luyện mô hình K-Means để phân cụm (ví dụ dựa trên diện tích sống và giá)
    scaler = StandardScaler()
    cluster_data = df[['sqft_living', 'price']]
    X_scaled = scaler.fit_transform(cluster_data)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return dt_model, kmeans, scaler, features, df

dt_model, kmeans, scaler, features, df = load_and_train_models()

# 2. Giao diện nhập thông số bên Sidebar
st.sidebar.header("🎛️ Nhập thông số ngôi nhà")

input_bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=10, value=3, step=1)
input_bathrooms = st.sidebar.number_input("Số phòng tắm", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
input_sqft_living = st.sidebar.slider("Diện tích sống (sqft_living)", 500, 10000, 2100, 50)
input_sqft_lot = st.sidebar.slider("Diện tích đất (sqft_lot)", 500, 20000, 5000, 100)
input_floors = st.sidebar.selectbox("Số tầng (floors)", [1.0, 1.5, 2.0, 2.5, 3.0], index=2)
input_condition = st.sidebar.slider("Tình trạng nhà (condition: 1-5)", 1, 5, 4, 1)
input_sqft_above = st.sidebar.slider("Diện tích trên mặt đất (sqft_above)", 500, 8000, 1800, 50)
input_sqft_basement = st.sidebar.slider("Diện tích tầng hầm (sqft_basement)", 0, 4000, 300, 50)
input_yr_built = st.sidebar.number_input("Năm xây dựng (yr_built)", min_value=1900, max_value=2026, value=1995, step=1)

# 3. Hiển thị kết quả xử lý từ 2 thuật toán
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Kết quả dự đoán từ thuật toán Cây Quyết Định")
    
    user_data = pd.DataFrame([[
        input_bedrooms, input_bathrooms, input_sqft_living, input_sqft_lot,
        input_floors, input_condition, input_sqft_above, input_sqft_basement, input_yr_built
    ]], columns=features)
    
    # Dự đoán giá bằng Decision Tree
    predicted_price = dt_model.predict(user_data)[0]
    
    st.success(f"### Giá nhà dự đoán: **${predicted_price:,.2f} USD**")
    
    # Phân cụm ngôi nhà vừa nhập bằng K-means dựa trên diện tích và giá dự đoán
    user_cluster_scaled = scaler.transform([[input_sqft_living, predicted_price]])
    cluster_id = kmeans.predict(user_cluster_scaled)[0]
    
    cluster_names = {
        0: "Phân khúc Nhà phổ thông / Tiết kiệm",
        1: "Phân khúc Nhà trung cấp / Tiêu chuẩn",
        2: "Phân khúc Biệt thự / BĐS cao cấp"
    }
    cluster_label = cluster_names.get(cluster_id, f"Nhóm cụm số {cluster_id}")
    
    st.info(f"🏷️ **Phân khúc K-Means Clustering**: {cluster_label}")

with col2:
    st.subheader("📋 Thống kê dữ liệu")
    st.metric(label="Tổng số mẫu trong dữ liệu", value=len(df))
    st.metric(label="Giá trung bình toàn tập", value=f"${df['price'].mean():,.0f}")

st.markdown("---")
st.subheader("🔍 Dữ liệu mẫu tích hợp phân cụm K-Means")
st.dataframe(df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'Cluster']].head(10), use_container_width=True)
