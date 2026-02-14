import streamlit as st
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

st.title("Fake Job Detection")

user_input = st.text_area("Enter Job Description")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.error("Fake Job Posting")
    else:
        st.success("Real Job Posting")
