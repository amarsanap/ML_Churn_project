import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():

    with open("Customer_01Churn_model.pkl","rb") as f:
        model = pickle.load(f)

    with open("encoders.pkl","rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, encoders = load_assets()


# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer will churn or stay.")

col1, col2 = st.columns([1,2])

# ---------------- INPUT SECTION ----------------
with col1:

    st.header("Customer Information")

    gender = st.selectbox("Gender", ["Male","Female"])

    senior = st.selectbox("Senior Citizen", [0,1])

    partner = st.selectbox("Partner", ["Yes","No"])

    dependents = st.selectbox("Dependents", ["Yes","No"])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    phone = st.selectbox("Phone Service", ["Yes","No"])

    multiple = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    security = st.selectbox("Online Security", ["Yes","No","No internet service"])

    backup = st.selectbox("Online Backup", ["Yes","No","No internet service"])

    device = st.selectbox("Device Protection", ["Yes","No","No internet service"])

    support = st.selectbox("Tech Support", ["Yes","No","No internet service"])

    tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])

    movies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    contract = st.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    billing = st.selectbox("Paperless Billing", ["Yes","No"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)

    total = st.number_input("Total Charges", value=100.0)

# ---------------- PREDICTION ----------------
with col2:

    st.header("Prediction Result")

    if st.button("Predict Churn"):

        input_data = pd.DataFrame({
            'gender':[gender],
            'SeniorCitizen':[senior],
            'Partner':[partner],
            'Dependents':[dependents],
            'tenure':[tenure],
            'PhoneService':[phone],
            'MultipleLines':[multiple],
            'InternetService':[internet],
            'OnlineSecurity':[security],
            'OnlineBackup':[backup],
            'DeviceProtection':[device],
            'TechSupport':[support],
            'StreamingTV':[tv],
            'StreamingMovies':[movies],
            'Contract':[contract],
            'PaperlessBilling':[billing],
            'PaymentMethod':[payment],
            'MonthlyCharges':[monthly],
            'TotalCharges':[total]
        })


        # -------- Encode categorical columns --------
        for column, encoder in encoders.items():
            input_data[column] = encoder.transform(input_data[column])


        try:

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.metric("Churn Probability", f"{probability:.2%}")

            if prediction == 1:

                st.error("⚠️ Customer Likely to Churn")
                st.warning("Retention strategy recommended.")

            else:

                st.success("✅ Customer Likely to Stay")


            st.progress(int(probability * 100))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Amar Sanap | Customer Churn ML Project")
