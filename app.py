import streamlit as st
import joblib
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Download nltk data
nltk.download('stopwords')

# Set page configuration at the beginning
st.set_page_config(page_title="Restaurant Reviews", page_icon="🍽️")

# Load the trained model
model = joblib.load("stacking_model.pkl")

# Load the CountVectorizer
cv = joblib.load("count_vectorizer.pkl")

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to clean and preprocess the review text
def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

# Custom CSS to make the app beautiful
custom_css = """
<style>

    .stApp {
        background-image: url(https://t3.ftcdn.net/jpg/02/34/50/46/240_F_234504608_hh2sThcrZETshikECMbKckmy3yeiGqlo.jpg);
        background-size: cover;
    }  

    [class="st-emotion-cache-1whx7iy e1nzilvr4"]{
        font-size: 50px;
    }

    body {
        background-color: #F0F2F6;
        color: #333;
    }
    h1 {
        font-size: 2.5rem;
        color: black;
        margin-left: 160px;
        margin-bottom: 50px;
    }

    h3 {
        font-size: 1.5rem;
        color: black;
        margin-left: 110px;
        margin-bottom: 20px;
    }

    strong{
        font-size: 20px;
    }

    p, div {
        font-size: 1.25rem;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        padding: 10px;
        margin-top:40px;
    }

    .stButton > button {
        background-color: #007BFF;
        border: 2px solid;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        margin-top:40px;
    }

    [data-testid="stMarkdownContainer"]{
        color: #000000;
    }

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit app
st.title("Restaurant Reviews")
st.markdown("### Predict the sentiment of a restaurant review")

# Input review text with placeholder
review_text = st.text_area("", placeholder="Enter your restaurant review here...")

if st.button("Predict Sentiment"):
    if review_text.strip():
        # Preprocess the input review
        cleaned_review = preprocess_review(review_text)
        # Transform the review using CountVectorizer
        review_vector = cv.transform([cleaned_review]).toarray()
        # Predict the sentiment
        prediction = model.predict(review_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        # Display the prediction
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.write("Please enter a review.")
