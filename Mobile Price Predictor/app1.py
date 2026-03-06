import streamlit as st
import joblib
import pandas as pd

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Mobile Price Prediction", layout="centered")

st.title("📱 Mobile Price Prediction App")
st.write("Enter mobile specifications to predict its price")

# -----------------------
# USER INPUTS
# -----------------------
brand = st.text_input("Brand (Example: SAMSUNG, REALME)").upper()

ram = st.number_input("RAM (GB)", min_value=1, max_value=24, value=4)
rom = st.number_input("ROM (GB)", min_value=8, max_value=1024, value=64)
expandable = st.number_input("Expandable Storage (GB)", min_value=0, max_value=2048, value=0)
camera = st.number_input("Rear Camera (MP)", min_value=2, max_value=200, value=48)
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, value=5000)
rating = st.slider("User Rating", min_value=1.0, max_value=5.0, value=4.0)

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict Price 💰"):

    # Create DataFrame (IMPORTANT)
    input_df = pd.DataFrame([{
        "Brand": brand,
        "RAM": ram,
        "ROM": rom,
        "Expandable": expandable,
        "Camera": camera,
        "Battery": battery,
        "Rating": rating
    }])

    # Apply SAME encoding as training
    input_df = pd.get_dummies(input_df)

    # Align with model's training features (CRITICAL FIX)
    input_df = input_df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Mobile Price: ₹ {round(prediction, 2)}")

st.markdown("---")
st.caption("ML Project | Mobile Price Prediction | Streamlit UI")
