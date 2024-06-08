import pandas as pd
import joblib
import streamlit as st
import os

# Check if the model file exists
model_path = 'breast_cancer_detector.sav'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

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

    # Default values for testing purposes
    default_values = {
        'mean radius': 20.57, 'mean texture': 17.77, 'mean perimeter': 132.9, 'mean area': 1326.0,
        'mean smoothness': 0.08474, 'mean compactness': 0.07864, 'mean concavity': 0.0869,
        'mean concave points': 0.07017, 'mean symmetry': 0.1812, 'mean fractal dimension': 0.05667,
        'radius error': 0.3242, 'texture error': 1.093, 'perimeter error': 2.587, 'area error': 36.58,
        'smoothness error': 0.00749, 'compactness error': 0.00431, 'concavity error': 0.01364,
        'concave points error': 0.01016, 'symmetry error': 0.01464, 'fractal dimension error': 0.00351,
        'worst radius': 25.38, 'worst texture': 17.33, 'worst perimeter': 184.6, 'worst area': 2019.0,
        'worst smoothness': 0.1622, 'worst compactness': 0.6656, 'worst concavity': 0.7119,
        'worst concave points': 0.2654, 'worst symmetry': 0.4601, 'worst fractal dimension': 0.1189
    }

    input_values = {}
    all_fields_filled = True
    for feature in features:
        value = st.number_input(feature, min_value=0.0, step=0.1, value=default_values[feature])
        if value == 0.0:
            all_fields_filled = False
        input_values[feature] = value

    if st.button("Predict"):
        if not all_fields_filled:
            st.error("Please fill in all fields before making a prediction.")
        else:
            data = pd.DataFrame([input_values])
            st.write("Input Data:", data)

            try:
                prediction = predict(data)
                diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
                st.write("Prediction:", diagnosis)
                
                if diagnosis == "Malignant":
                    st.warning("The prediction indicates a high likelihood of breast cancer. Please consult with a healthcare provider for further evaluation.")
                    st.write("### Risk Factors")
                    st.write("""
                        - **Age**: The risk of breast cancer increases as you get older.
                        - **Genetic mutations**: Inherited changes (mutations) to certain genes, such as BRCA1 and BRCA2.
                        - **Reproductive history**: Early menstrual periods before age 12 and starting menopause after age 55.
                        - **Dense breasts**: Women with dense breasts are more likely to get breast cancer.
                        - **Family history**: A family history of breast cancer.
                        - **Previous treatment with radiation therapy**: Especially treatments to the chest area.
                        - **Lifestyle factors**: Such as alcohol consumption, obesity, and lack of physical activity.
                    """)
                    st.write("### Recommended Next Steps")
                    st.write("""
                        1. **Consult a Doctor**: Schedule an appointment with a healthcare provider for a detailed examination and diagnostic tests.
                        2. **Further Testing**: Your doctor might recommend a biopsy, mammogram, or other imaging tests.
                        3. **Genetic Counseling**: If you have a family history of breast cancer, consider genetic counseling to understand your risk better.
                        4. **Lifestyle Changes**: Adopt a healthy lifestyle to lower your risk.
                    """)
                else:
                    st.success("The prediction indicates a low likelihood of breast cancer. However, regular screenings and a healthy lifestyle are important.")
                    st.write("### General Health Tips")
                    st.write("""
                        - Maintain a healthy diet rich in fruits, vegetables, and whole grains.
                        - Engage in regular physical activity.
                        - Avoid smoking and limit alcohol consumption.
                        - Stay at a healthy weight.
                        - Perform regular self-examinations and schedule routine screenings.
                    """)
            except ValueError as e:
                st.error(f"Error in prediction: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
