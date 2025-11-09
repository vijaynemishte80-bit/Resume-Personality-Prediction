# Resume Personality Prediction System

This project is a *machine learning-based system* that predicts a candidate’s *personality or category* by analyzing their resume text. It uses *TF-IDF vectorization* to convert resume text into numerical features and a *Logistic Regression model* for classification.

---

## *Key Features*

- Preprocesses resume text by removing special characters, lowercasing, and removing stopwords.  
- Converts text into numerical features using *TF-IDF*.  
- Predicts personality or category using a *trained machine learning model*.  
- Provides a *Streamlit web app* for interactive prediction: upload a resume (TXT/CSV) and get instant results.  
- Easy to extend with *PDF support, embeddings (GloVe/BERT), or more advanced ML models*.  

---

## *Installation*

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/Resume-Personality-Prediction.git

##Running the resume.py file in cmd window
1. streamlit run "resume.py"
