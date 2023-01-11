import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="💼",
    menu_items={
        'About': "Xyla Ramadhan batch 13"
    }
)

st.header('Churn Prediction P2 M1')
st.write("""
Xyla Ramadhan
""")

@st.cache
def fetch_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Xylverize/streamlit-example/master/new_dataset_clean1.csv')
    return df

df = fetch_data()
st.write(df)

st.sidebar.header('User Input Features')

def user_input():
    gender = st.selectbox("Customer Gender", ["Male", "Female"])
    seniorCitizen = st.selectbox("Whether the customer is a senior citizen or not", ["No", "Yes"] )
    partner = st.selectbox("Whether the customer has a partner or not", ["No", "Yes"])
    dependent = st.selectbox(" Whether the customer has dependents or not", ["No", "Yes"])
    tenure = st.number_input("Number of months the customer has stayed with the company? (Tenure)", min_value=1)
    phoneService = st.selectbox("Phone service ", ["No", "Yes"])
    multipleLines = st.selectbox("Multiple Lines ", ["No", "Yes","No phone service"])
    internetService = st.selectbox("Internet Service Provider ", ["No", "DSL", "Fiber optic"])
    onlineSecurity = st.selectbox("Online Security ", ["No", "Yes","No internet service"])
    onlineBackup = st.selectbox("Online Backup ", ["No", "Yes","No internet service"])
    deviceProtection = st.selectbox("Device Protection", ["No", "Yes","No internet service"])
    techSupport = st.selectbox("Tech Support ", ["No", "Yes","No internet service"])
    streamingTV = st.selectbox("Streaming TV ", ["No", "Yes","No internet service"])
    streamingMovies = st.selectbox("Streaming Movies ", ["No", "Yes","No internet service"])
    contract = st.selectbox("Contract ", ["Month-to-month", "One year", "Two year"])
    paperlessBilling = st.selectbox("Paperless Billing ", ["Yes", "No"])
    paymentMethod = st.selectbox("Payment Method ",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
    monthlyCharges = st.number_input("Monthly Charges", value=18.0,min_value=18.0)    
    totalCharges = st.number_input("Total Charges ",value=18.0,min_value=18.0)
    
    new_data = {
            "gender": gender,
            "SeniorCitizen": seniorCitizen,
            "Partner": partner,
            "Dependents": dependent,
            "tenure": tenure,
            "PhoneService": phoneService,
            "MultipleLines": multipleLines,
            "InternetService": internetService,
            "OnlineSecurity": onlineSecurity,
            "OnlineBackup": onlineBackup,
            "DeviceProtection": deviceProtection,
            "TechSupport": techSupport,
            "StreamingTV": streamingTV,
            "StreamingMovies": streamingMovies,
            "Contract": contract,
            "PaperlessBilling": paperlessBilling,
            "PaymentMethod": paymentMethod,
            "MonthlyCharges": monthlyCharges,
            "TotalCharges": totalCharges,
        }

    features = pd.DataFrame(new_data, index=[0])
    return features


input = user_input()

st.subheader('User Input')
st.write(input)


st.write('Based on user input, the placement model predicted: ')


pipe = pickle.load(open("preprocessor.pkl", "rb"))

prediction = pipe.transform(input)

if prediction[0][1] > 0.5:
    prediction = 'Churn'
else:
    prediction = 'Not Churn'

st.header(prediction)
