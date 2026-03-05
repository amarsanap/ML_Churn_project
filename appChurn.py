import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():

    with open("churn_model.pkl","rb") as f:
        model = pickle.load(f)

    with open("encoders.pkl","rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, encoders = load_assets()


# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a telecom customer is likely to churn.")


col1, col2 = st.columns([1,2])


# ---------------- INPUT SECTION ----------------
with col1:

    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Male","Female"])

    senior = st.selectbox("Senior Citizen", [0,1])

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month","One year","Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

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


# ---------------- PREDICTION ----------------
with col2:

    st.subheader("Prediction Result")

    if st.button("Predict Churn"):

        input_data = pd.DataFrame({
            "gender":[gender],
            "SeniorCitizen":[senior],
            "tenure":[tenure],
            "Contract":[contract],
            "InternetService":[internet],
            "PaymentMethod":[payment],
            "MonthlyCharges":[monthly]
        })


        # -------- Encode categorical features --------
        for column, encoder in encoders.items():
            input_data[column] = encoder.transform(input_data[column])


        # -------- Prediction --------
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]


        st.metric("Churn Probability", f"{probability:.2%}")

        if prediction == 1:

            st.error("⚠️ Customer Likely to Churn")

            st.warning(
                "Retention strategy recommended (discount, loyalty plan)."
            )

        else:

            st.success("✅ Customer Likely to Stay")


        st.progress(int(probability * 100))


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed by Amar Sanap | Customer Churn Prediction Project")
