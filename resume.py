# Save this file as app.py
import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess resume text
def preprocess_text_simple(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load your trained TF-IDF vectorizer and model (or train inside app)
# For demo, we will train on your CSV dataset
data = pd.read_csv("resume.csv")
data['Cleaned_Resume'] = data['Resume_str'].apply(preprocess_text_simple)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['Cleaned_Resume']).toarray()
y = data['Category']

model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Streamlit UI
st.title("Resume Personality Prediction System")

uploaded_file = st.file_uploader("Upload your Resume (TXT or CSV)", type=["txt","csv"])

if uploaded_file is not None:
    try:
        # If TXT file
        if uploaded_file.name.endswith(".txt"):
            resume_text = str(uploaded_file.read(), "utf-8")
        # If CSV file
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            # Assuming text is in first column
            resume_text = str(df.iloc[0,0])
        
        # Preprocess and predict
        cleaned_text = preprocess_text_simple(resume_text)
        vector = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(vector)[0]
        
        st.subheader("Predicted Personality / Category:")
        st.success(prediction)
    
    except Exception as e:
        st.error("Error processing file: " + str(e))