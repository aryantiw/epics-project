import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open('dtree.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define a function to preprocess the text input using the vectorizer
def preprocess_text(text):
    preprocessed_text = vectorizer.transform([text])
    return preprocessed_text

# Define a function to make a prediction using the trained model
def predict_sentiment(preprocessed_text):
    prediction = model.predict(preprocessed_text)
    if prediction == 1:
        return 'neutral'
    elif prediction == 0:
        return 'positive'
    else:
        return 'Negative'

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Sentiment Analysis App')

    # Get user input
    user_input = st.text_input('Enter a sentence:')

    # When the user clicks the "Predict" button
    if st.button('Predict'):
        # Preprocess the user input
        preprocessed_text = preprocess_text(user_input)

        # Make a prediction using the trained model
        prediction = predict_sentiment(preprocessed_text)

        # Show the prediction to the user
        st.write('Sentiment:', prediction)

# Run the app
if __name__ == '__main__':
    main()
