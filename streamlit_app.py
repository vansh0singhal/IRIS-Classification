# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your trained KNN model (replace 'knn_model.pkl' with the correct file path)
model = joblib.load('IRIS_model.pkl')

# Load your dataset to map predicted labels to species names (assumes it's a CSV file with species column)
df = pd.read_csv('IRIS.csv')  # Replace 'iris_dataset.csv' with the correct filename

# Ensure unique species labels
species_mapping = df['species'].unique()

# Streamlit app
st.title('Iris Flower Species Prediction')
st.write("This app predicts the *species* of Iris flowers based on input features.")

# User inputs using text inputs
st.header('Input Features')
try:
    sepal_length = float(st.text_input('Enter Sepal Length (cm)', '5.0'))
    sepal_width = float(st.text_input('Enter Sepal Width (cm)', '3.0'))
    petal_length = float(st.text_input('Enter Petal Length (cm)', '1.5'))
    petal_width = float(st.text_input('Enter Petal Width (cm)', '0.5'))
except ValueError:
    st.error("Please enter valid numeric values for all input fields.")

# Predict button
if st.button('Predict'):
    # Check if all values are valid numbers
    if all(isinstance(v, (int, float)) for v in [sepal_length, sepal_width, petal_length, petal_width]):
        # Convert inputs to numpy array for prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Prediction
        prediction = model.predict(input_data)

        # Ensure prediction is an integer index (e.g., array([0]) -> 0)
        predicted_index = int(prediction[0])

        # Map predicted numeric label to species name
        predicted_species = species_mapping[predicted_index]

        # Display results
        st.subheader('Prediction')
        st.write(f"The predicted species is: *{predicted_species}*")
    else:
        st.error("All inputs must be valid numbers.")