import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained churn prediction model (saved as .pkl file)
model = joblib.load("churn_model.pkl")
# Load the feature names used during model training (ensures input column order matches)
features = joblib.load("model_features.pkl")
# Configure the Streamlit page appearance (title, layout style)
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
# Main title displayed on the web app
st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict churn risk.")

# MANUAL ENCODING MAPPINGS
# Converting categorical text inputs to numerical values as required by ML model
gender_map = {"Female": 0, "Male": 1, "Other": 2}

state_map = {
    "Delhi": 0,
    "Gujarat": 1,
    "Karnataka": 2,
    "Madhya Pradesh": 3,
    "Maharashtra": 4,
    "Rajasthan": 5,
    "Tamil Nadu": 6,
    "Uttar Pradesh": 7,
    "West Bengal": 8,
    "Others": 9
}  # State/region encoding

operator_map = {
    "Airtel": 0,
    "BSNL": 1,
    "Jio": 2,
    "Others": 3,
    "Vi": 4
} # Telecom operator encoding

plan_map = {"Postpaid": 0, "Prepaid": 1} # Plan type encoding
port_map = {"No": 0, "Yes": 1} # Port request encoding

# Dividing input fields into 3 columns for cleaner presentation

col1, col2, col3 = st.columns(3)  # Create three equal-width columns

# ---- COLUMN 1 INPUTS ----

with col1:
    age = st.number_input("Age", 18, 70, 30)

    gender = st.selectbox(
        "Gender",
        list(gender_map.keys())
    )

    state = st.selectbox(
        "State",
        list(state_map.keys())
    )

    operator = st.selectbox(
        "Operator",
        list(operator_map.keys())
    )

    sim_tenure = st.number_input(
        "SIM Tenure Months",
        1,
        120,
        12
    )

    recharge_amount = st.number_input(
        "Recharge Amount",
        99,
        2999,
        299
    )

# ---- COLUMN 2 INPUTS ----

with col2:
    recharge_frequency = st.number_input(
        "Recharge Frequency Days",
        7,
        90,
        28
    )

    data_usage = st.number_input(
        "Data Usage GB",
        0.5,
        100.0,
        10.0
    )

    call_minutes = st.number_input(
        "Call Minutes",
        20,
        3000,
        500
    )

    sms_usage = st.number_input(
        "SMS Usage",
        0,
        500,
        50
    )

    plan_type = st.selectbox(
        "Plan Type",
        list(plan_map.keys())
    )

    network_complaints = st.number_input(
        "Network Complaints",
        0,
        10,
        2
    )

# ---- COLUMN 3 INPUTS ----

with col3:
    dropped_calls = st.number_input(
        "Dropped Calls",
        0,
        50,
        5
    )

    late_recharge = st.number_input(
        "Late Recharge Days",
        0,
        30,
        2
    )

    support_calls = st.number_input(
        "Customer Support Calls",
        0,
        8,
        1
    )

    satisfaction = st.slider(
        "Satisfaction Score",
        1,
        5,
        3
    )

    port_request = st.selectbox(
        "Port Request",
        list(port_map.keys())
    )

# CREATE INPUT DATAFRAME FOR PREDICTION
# Convert all user inputs into a single-row DataFrame

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_map[gender]],
    "State": [state_map[state]],
    "Operator": [operator_map[operator]],
    "SIMTenureMonths": [sim_tenure],
    "RechargeAmount": [recharge_amount],
    "RechargeFrequencyDays": [recharge_frequency],
    "DataUsageGB": [data_usage],
    "CallMinutes": [call_minutes],
    "SMSUsage": [sms_usage],
    "PlanType": [plan_map[plan_type]],
    "NetworkComplaints": [network_complaints],
    "DroppedCalls": [dropped_calls],
    "LateRechargeDays": [late_recharge],
    "CustomerSupportCalls": [support_calls],
    "SatisfactionScore": [satisfaction],
    "PortRequest": [port_map[port_request]]
})

# Reorder columns to exactly match the training feature order (critical for correct prediction)
input_data = input_data[features]

# PREDICTION BUTTON AND OUTPUT

if st.button("Predict Churn"):
    
    # Get binary prediction (0 = no churn, 1 = churn)
    prediction = model.predict(input_data)[0]
    # Get probability of churn (class 1 probability)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    # Display churn status with color-coded message
    if prediction == 1:
        st.error("Customer is likely to churn") # Red error box for churn
    else:
        st.success("Customer is not likely to churn") # Green success box for no churn

    # Show exact churn probability percentage
    st.write(f"Churn Probability: **{probability * 100:.2f}%**")

    # Display risk level based on probability thresholds
    if probability >= 0.70:
        st.error("Risk Level: High")    # High risk (>=70%)

    elif probability >= 0.40:
        st.warning("Risk Level: Medium")    # Medium risk (40-69%)

    else:
        st.success("Risk Level: Low")   # Low risk (<40%)