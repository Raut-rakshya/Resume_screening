import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

model= joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit App
st.title("Resume Screening Prediction")

user_input = st.text_area("Paste the resume content here:")
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)[0]
        predicted_category = label_encoder.inverse_transform([prediction])[0]
        st.success(f"Predicted Job Category: **{predicted_category}**")