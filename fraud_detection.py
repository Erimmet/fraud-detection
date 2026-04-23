import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')

st.title('Fraud Detection App')

st.markdown('Please enter the transaction details below and use the predict button:')

st.divider()

traansaction_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input('Old Balance(sender)', min_value=0.0, value=10000.0)
newbalanceOrg = st.number_input('New Balance(sender)', min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input('Old Balance(receiver)', min_value=0.0, value=0.0)
newbalanceDest = st.number_input('New Balance(receiver)', min_value=0.0, value=0.0)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'type': [traansaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrg],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    
    prediction = model.predict(input_data)

    st.subheader(f"prediction : '{int(prediction)}'")

    if prediction[0] == 1:
        st.error('The transaction is likely to be a fraud.')
    else:
        st.success('This transaction looks legit')