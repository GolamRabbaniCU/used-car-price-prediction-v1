import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Load the trained pipeline once at startup (not inside a function/callback)
# ---------------------------------------------------------------------------
pipeline = joblib.load("models/best_model.pkl")

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate the resale (selling) price of a used car based on its "
    "showroom price, mileage, age, and a few other details."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input(
        "Present Price (in Taka, lakhs)",
        min_value=0.0, max_value=100.0, value=5.0, step=0.1,
        help="Current ex-showroom price of this car model, in lakhs (৳100,000s)."
    )
    kms_driven = st.number_input(
        "Kilometres Driven",
        min_value=0, max_value=500000, value=30000, step=500
    )
    car_age = st.slider("Car Age (years)", min_value=1, max_value=20, value=5)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.divider()

if st.button("Predict Selling Price", type="primary"):
    input_df = pd.DataFrame([{
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Car_Age": car_age,
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction = max(prediction, 0)  # price can't be negative

    st.success(f"### Estimated Selling Price: ৳{prediction:.2f} Lakhs")
    st.caption(
        "This is a model estimate based on historical CarDekho listings and "
        "should be used as a reference point, not a formal valuation."
    )

st.divider()
st.caption("Model: trained on the CarDekho Used Car dataset · Built with scikit-learn + Streamlit")
