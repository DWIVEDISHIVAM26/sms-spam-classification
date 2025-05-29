import streamlit as st
import pickle
import string
import nltk
import spacy
import time
nlp = spacy.load("en_core_web_sm")
from nltk.stem import PorterStemmer
from spacy.lang.en import English
from nltk.corpus import stopwords
stemmer = PorterStemmer()

def transform_text(text):
  text = text.lower()
  Docu = nlp(text)
  text = [token.text for token in Docu]

  y = []
  for i in text:
    if i.isalnum():
        y.append(i)

  text = y[:]
  y.clear()

# After for loop this is the syntax to check the stopword and punctuation:

  for i in text:
    if i not in nlp.Defaults.stop_words and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

# # Stemming

  stemmed_words = [stemmer.stem(token.text) for token in Docu]
  for i in text:
     y.append(stemmer.stem(i))
  #   # stemmed_words = [stemmer.stem(token.text) for token in i]
  #   # print(stemmer.stem(token.text))
  #   # y.append(stemmed_words.index(i))

  return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# --- Page Setup ---
st.set_page_config(page_title="Spam Classifier", layout="centered", page_icon="📨")
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #f8f8ff, #e6e6fa);
        }
        .spam {
            background-color: #ff4d4d;
            color: white;
            padding: 1em;
            border-radius: 12px;
            text-align: center;
        }
        .ham {
            background-color: #2ecc71;
            color: white;
            padding: 1em;
            border-radius: 12px;
            text-align: center;
        }
        .loader {
            border: 6px solid #f3f3f3;
            border-top: 6px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #4b0082;'>📨 Email / SMS Spam Classifier</h1>", unsafe_allow_html=True)

# --- Input Box ---
input_sms = st.text_area("Enter the message to classify", height=150)

# --- Predict Button ---
if st.button("🔍 Classify"):
    with st.spinner('Analyzing...'):
        time.sleep(1)  # simulate processing

        # Step 1: Preprocess
        transformed_sms = transform_text(input_sms)

        # Step 2: Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Step 3: Predict
        result = model.predict(vector_input)[0]

        # Step 4: Display
        if result == 1:
            st.markdown('<div class="spam">🚫 Spam</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ham">✅ Not Spam</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center;'>Made with ❤️ using Streamlit</div>
""", unsafe_allow_html=True)
