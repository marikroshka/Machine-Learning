import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model5_features.joblib")


st.title("Hotel Booking Prediction")

st.markdown("""
    <style>
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)


# --- Ввод признаков ---

# Числовой ввод для lead_time
lead_time = st.number_input("Lead Time", min_value=0, max_value=500, value=50)

# Слайдер для месяца заезда
arrival_month = st.slider("Arrival Month", min_value=1, max_value=12, value=6)

# Слайдер для количества спецзапросов
no_of_special_requests = st.slider("Number of Special Requests", min_value=0, max_value=5, value=1)

# Слайдер для средней цены за номер
avg_price_per_room = st.slider("Average Price per Room", min_value=0.0, max_value=600.0, value=100.0)

# Радио для сегмента рынка
market_segment_type = st.radio(
    "Market Segment Type",
    ["Online", "Offline"]
)

# --- Преобразование в DataFrame ---
input_df = pd.DataFrame([{
    "lead_time": lead_time,
    "arrival_month": arrival_month,
    "no_of_special_requests": no_of_special_requests,
    "avg_price_per_room": avg_price_per_room,
    "market_segment_type": market_segment_type
}])

# --- Предсказание ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Booking Likely to Be Cancelled")
    else:
        st.success("✅ Booking Likely to Be Honored")

