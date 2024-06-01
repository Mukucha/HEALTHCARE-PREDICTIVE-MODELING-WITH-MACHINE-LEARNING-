import pandas as pd
import joblib
import streamlit as st
import os

# Check if the model file exists
model_path = 'breast_cancer_detector.sav'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")

# Load the saved model
try:
    breast_cancer_detector_model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to make predictions
def predict(data):
    predictions = breast_cancer_detector_model.predict(data)
    return predictions

# Streamlit UI
def main():
    st.title("Breast Cancer Detection")
    st.write("Enter the values for the features to make predictions.")

    # Create input fields for features
    features = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]

    input_values = {}
    for feature in features:
        input_values[feature] = st.number_input(feature)

    if st.button("Predict"):
        data = pd.DataFrame([input_values])

        try:
            prediction = predict(data)
            st.write("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
