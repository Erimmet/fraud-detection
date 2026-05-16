import streamlit as st
import pandas as pd
import joblib
import numpy as np


@st.cache_resource  
def load_model_package():
    package = joblib.load('fraud_detection_package.pkl')
    return package

# Load package
model_package = load_model_package()

# Extract model and threshold
model = model_package['pipeline']
THRESHOLD = model_package['threshold']
VERSION = model_package.get('version', '1.0')

# Page config
st.set_page_config(page_title="Fraud Detection System", page_icon="🚨")
st.title("🚨 Fraud Detection System")
st.markdown(f"*Model Version: {VERSION} | Decision Threshold: {THRESHOLD}*")
st.markdown("Please enter the transaction details below:")
st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        'Transaction Type', 
        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT'],
        help="Type of transaction being performed"
    )
    
    amount = st.number_input(
        'Amount', 
        min_value=0.0, 
        value=1000.0,
        help="Transaction amount in currency units"
    )
    
    oldbalanceOrg = st.number_input(
        "Sender's Old Balance", 
        min_value=0.0, 
        value=10000.0,
        help="Balance of the sender BEFORE this transaction"
    )
    
    newbalanceOrig = st.number_input(
        "Sender's New Balance", 
        min_value=0.0, 
        value=9000.0,
        help="Balance of the sender AFTER this transaction"
    )

with col2:
    oldbalanceDest = st.number_input(
        "Receiver's Old Balance", 
        min_value=0.0, 
        value=0.0,
        help="Balance of the receiver BEFORE this transaction"
    )
    
    newbalanceDest = st.number_input(
        "Receiver's New Balance", 
        min_value=0.0, 
        value=0.0,
        help="Balance of the receiver AFTER this transaction"
    )

st.divider()

# Predict button
if st.button("🔍 Predict Fraud Risk", type="primary"):

    balanceDiffOrig = newbalanceOrig - oldbalanceOrg
    balanceDiffDest = newbalanceDest - oldbalanceDest
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    is_sender_emptied = 1 if newbalanceOrig == 0 else 0
    
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'balanceDiffOrig': [balanceDiffOrig],
        'balanceDiffDest': [balanceDiffDest],
        'errorBalanceDest': [errorBalanceDest],
        'is_sender_emptied': [is_sender_emptied]
    })
    
    # Get prediction
    try:
        # Get probability of fraud (class 1)
        proba = model.predict_proba(input_data)[0, 1]
        
        # Apply your threshold
        prediction = 1 if proba >= THRESHOLD else 0
        
        # Display results
        st.divider()
        
        # Create metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fraud Probability", f"{proba:.2%}")
        
        with col2:
            st.metric("Decision Threshold", f"{THRESHOLD:.0%}")
        
        with col3:
            st.metric("Prediction", "FRAUD" if prediction == 1 else "LEGITIMATE")
        
        # Show result with appropriate styling
        if prediction == 1:
            st.error("🚨 **HIGH RISK ALERT!** This transaction is predicted to be FRAUDULENT.")
            
            # Show risk factors
            st.warning("**Risk Factors Detected:**")
            risk_factors = []
            if is_sender_emptied:
                risk_factors.append("• Sender's account was completely emptied")
            if errorBalanceDest != 0:
                risk_factors.append(f"• Balance mismatch for receiver (error: {errorBalanceDest:.2f})")
            if transaction_type in ['TRANSFER', 'CASH_OUT']:
                risk_factors.append(f"• High-risk transaction type: {transaction_type}")
            if amount > 100000:
                risk_factors.append(f"• Unusually large amount: {amount:,.2f}")
            
            if risk_factors:
                for risk in risk_factors:
                    st.write(risk)
            else:
                st.write("• Transaction patterns match known fraud indicators")
                
        else:
            st.success("✅ **LOW RISK** - This transaction appears legitimate.")
        
        # Show explanation
        with st.expander("📊 How was this calculated?"):
            st.write(f"""
            **Decision Process:**
            1. Model output: {proba:.2%} probability of fraud
            2. Threshold set at: {THRESHOLD:.0%}
            3. Since {proba:.2%} {'≥' if prediction == 1 else '<'} {THRESHOLD:.0%}, 
               prediction is **{'FRAUD' if prediction == 1 else 'LEGITIMATE'}**
            
            **Engineered Features Used:**
            - Balance Difference (Sender): {balanceDiffOrig:.2f}
            - Balance Difference (Receiver): {balanceDiffDest:.2f}
            - Balance Error (Receiver): {errorBalanceDest:.2f}
            - Sender Emptied: {'Yes' if is_sender_emptied == 1 else 'No'}
            """)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check that all inputs are valid.")

# Add sidebar with model info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Model Type:** {model_package.get('model_type', 'XGBoost')}")
    st.write(f"**Version:** {VERSION}")
    st.write(f"**Training Date:** {model_package.get('training_date', 'Unknown')}")
    st.write(f"**Decision Threshold:** {THRESHOLD}")
    
    st.divider()
    st.header("📈 Performance Metrics")
    perf = model_package.get('performance', {})
    if perf:
        st.metric("Precision", f"{perf.get('precision', 0.5525)*100:.1f}%")
        st.metric("Recall", f"{perf.get('recall', 0.9610)*100:.1f}%")
        st.metric("F1-Score", f"{perf.get('f1_score', 0.7016)*100:.1f}%")
    
    st.divider()
    st.header("⚙️ How It Works")
    st.write("""
    This system uses an XGBoost machine learning model trained on historical transaction data.
    
    **Key fraud indicators:**
    - Balance manipulation
    - Account emptying patterns
    - Unusual transaction amounts
    - High-risk transaction types
    """)

# Footer
st.divider()
st.caption("🚨 Fraud Detection System | For internal use only | Threshold: {:.0%}".format(THRESHOLD))