import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----- STEP 2: TEXT CLEANING -----
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # Converts to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Removes URLs
    text = re.sub(r'[^a-z\s]', ' ', text) # Keeps only alphabets and spaces (removes special characters)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----- STEP 3: FEATURE ENGINEERING -----
URGENCY_WORDS = ['urgent', 'immediately', 'action required', 'verify', 'now', 'limited', 'alert']
AUTHORITY_WORDS = ['bank', 'ceo', 'government', 'it department', 'manager', 'support']

def calculate_urgency_score(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in URGENCY_WORDS)

def calculate_authority_score(text):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(word) for word in AUTHORITY_WORDS)

def calculate_exclamation_count(text):
    if not isinstance(text, str):
        return 0
    return text.count("!")

# ----- MODEL TRAINING ENGINE -----
def train_and_save_model():
    data_file = "Phishing_Email.csv"
    if not os.path.exists(data_file):
        st.error(f"Dataset '{data_file}' not found. Please place it in the project root directory.")
        return False
        
    st.info("Loading dataset...")
    df = pd.read_csv(data_file)
    
    # 1. Detect automatically: Email text column, Label column
    text_col, label_col = None, None
    for col in df.columns:
        col_lower = str(col).lower()
        if text_col is None and any(kw in col_lower for kw in ['text', 'email', 'message']):
            text_col = col
        if label_col is None and any(kw in col_lower for kw in ['label', 'class', 'target', 'type']):
            label_col = col
            
    if not text_col: text_col = df.columns[0]
    if not label_col: label_col = df.columns[-1]
    
    # Rename them
    df = df.rename(columns={text_col: 'email_text', label_col: 'label'})
    
    # Drop null values and duplicates
    df = df.dropna(subset=['email_text', 'label']).drop_duplicates()
    
    # If dataset size is more than 20000 rows, randomly sample 15000 rows.
    if len(df) > 20000:
        df = df.sample(n=15000, random_state=42)
        
    # Ensure labels are binary (0 and 1)
    unique_labels = df['label'].unique()
    if set(unique_labels) != {0, 1}:
        mapping = {}
        for l in unique_labels:
            l_str = str(l).lower()
            if 'safe' in l_str or 'legitimate' in l_str or 'normal' in l_str or l_str == '0':
                mapping[l] = 0
            else:
                mapping[l] = 1
        df['label'] = df['label'].map(mapping)
    df['label'] = df['label'].astype(int)
    
    st.info("Applying text cleaning (this may take a moment)...")
    df['clean_text'] = df['email_text'].apply(clean_text)
    
    st.info("Extracting behavioral features...")
    df['urgency_score'] = df['email_text'].apply(calculate_urgency_score)
    df['authority_score'] = df['email_text'].apply(calculate_authority_score)
    df['exclamation_count'] = df['email_text'].apply(calculate_exclamation_count)
    
    st.info("Applying TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_text = vectorizer.fit_transform(df['clean_text'])
    
    # Convert sparse matrix to dense array for numpy hstack
    X_text_array = X_text.toarray()
    X_behavioral = df[['urgency_score', 'authority_score', 'exclamation_count']].values
    
    # Combine TF-IDF matrix with behavioral features using numpy hstack
    X_combined = np.hstack((X_text_array, X_behavioral))
    y = df['label'].values
    
    # Split dataset: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    st.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate and Print Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    st.success("Model trained successfully!")
    st.text(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {rec:.4f}\nF1 Score:  {f1:.4f}")
    
    # Save the trained model and vectorizer
    joblib.dump(model, 'phishing_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    return True

# ----- MAIN UI ENGINE -----
def main():
    st.set_page_config(page_title="Slingshot Secure", page_icon="🛡️", layout="centered")
    
    # Title
    st.title("🛡️ Slingshot Secure – AI Phishing Detector")
    st.markdown("Analyze an email to determine its risk of being a phishing attempt.")
    
    # Sidebar for Model Status / Retrain option
    st.sidebar.header("System Status")
    
    # Check if models exist
    model_exists = os.path.exists('phishing_model.pkl') and os.path.exists('vectorizer.pkl')
    
    if not model_exists:
        st.warning("Model and vectorizer not found. Please train the model to enable analysis.")
        st.sidebar.warning("Model Not Found")
        if st.button("Train Model Now"):
            with st.spinner("Training model, please wait..."):
                if train_and_save_model():
                    st.success("Training complete! Please analyze an email below.")
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
        return
    else:
        st.sidebar.success("Model Loaded Successfully")
        if st.sidebar.button("Retrain Model"):
            with st.spinner("Retraining model..."):
                if train_and_save_model():
                    st.sidebar.success("Retraining complete!")
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
                        
    # Load Model
    model = joblib.load('phishing_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    st.markdown("### Email Input")
    email_input = st.text_area("Paste the email content below:", height=200, placeholder="Example: URGENT! Verify your bank account immediately to avoid suspension...")
    
    if st.button("Analyze", type="primary"):
        if not email_input or not email_input.strip():
            st.error("Please enter email text to analyze.")
            return
            
        with st.spinner("Analyzing email..."):
            # Clean text
            cleaned = clean_text(email_input)
            
            # Predict
            urgency = calculate_urgency_score(email_input)
            authority = calculate_authority_score(email_input)
            exclamations = calculate_exclamation_count(email_input)
            
            # Transform
            X_text = vectorizer.transform([cleaned]).toarray()
            X_behavioral = np.array([[urgency, authority, exclamations]])
            
            # Combine
            X_input = np.hstack((X_text, X_behavioral))
            
            # Probabilities
            prob = model.predict_proba(X_input)[0]
            # Assumes 1 is Phishing, 0 is Safe
            phishing_prob = prob[1]
            threat_score = int(phishing_prob * 100)
            
            # Risk level
            if threat_score <= 30:
                risk_level = "Safe"
            elif threat_score <= 70:
                risk_level = "Suspicious"
            else:
                risk_level = "High Risk"
                
            st.markdown("---")
            st.markdown("### Analysis Results")
            
            # Display: Threat Score, Risk Level
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Threat Score", value=f"{threat_score}/100")
            with col2:
                st.metric(label="Risk Level", value=risk_level)
                
            # Threat Level Visualization
            st.progress(threat_score)
            
            # Probability percentage
            st.markdown(f"**Probability of Phishing:** {phishing_prob * 100:.2f}%")
            
            # Behavioral indicators detected
            st.markdown("#### Behavioral Indicators Detected")
            st.write(f"- **Urgency Words Found:** {urgency}")
            st.write(f"- **Authority Words Found:** {authority}")
            st.write(f"- **Exclamation Marks:** {exclamations}")
            
            # Clear explanation message
            st.markdown("#### Verdict")
            if risk_level == "Safe":
                st.success("This email appears to be SAFE and does not exhibit strong phishing indicators.")
            elif risk_level == "Suspicious":
                st.warning("This email has SUSPICIOUS characteristics. Proceed with caution and verify the sender.")
            else:
                st.error("HIGH RISK detected! Do not click on any links, open attachments, or provide personal information.")

if __name__ == "__main__":
    main()
