# This is a sample Python script.
import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tranform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  text = [i for i in text if i.isalnum()]
  text = [a for a in text if a not in stopwords.words('english') and a not in string.punctuation]
  text = [lem.lemmatize(x) for x in text]
  return " ".join(text)

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Classification')
sms_input = st.text_area("Enter the Message")

if st.button('Predict'):
    # 1.Pre-processes
    transform_sms = tranform_text(sms_input)
    # 2.Vectorizer
    vectorizer_input = tfid.transform([transform_sms])
    # 3.Predict
    result = model.predict(vectorizer_input)[0]
    # 4.Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")