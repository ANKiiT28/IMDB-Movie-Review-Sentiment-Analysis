# Step 1 : Load the necessary libraries and the pre-trained model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load hte IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model with ReLu activation function
model = load_model('simplernn_imdb.h5')


# Step 2 : Helper functions for prediction
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review]) 


# Function to preprocess user input 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



## streamlit part
import streamlit as st

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

# User input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    ## Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    
    ## Display results
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write("Please enter a movie review.")