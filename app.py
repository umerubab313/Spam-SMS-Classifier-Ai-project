import streamlit as st
import pickle

import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #lower casing
    text = nltk.word_tokenize(text)  #making token of words
    y = []
    
    # removing special characters only keeping alphanumeric ones:
    for i in text:
        if i.isalnum(): #checks if alphanumeric or not
            y.append(i)
    text = y[:] #as lists are mutable so we copy them like this making shallow copy       
    y.clear()

    # removing stopwords and punctuation marks
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.set_page_config(page_title="Spam Detection", page_icon="📩")

st.title("📩 Spam Detection System")
st.write("Enter a message to check whether it is spam or not.")

# User input
message = st.text_area("Message")

# Prediction
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        #step - 1: preprocessing
        transformed_sms = transform_text(message)

        #step - 2: vectorizing
        vector_input = tfidf.transform([transformed_sms])

        # step - 3: prediction
        result = model.predict(vector_input)[0]
        # step - 4: displaying our result
        if result == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")

