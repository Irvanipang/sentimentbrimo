import streamlit as st
import joblib
import os

# Validasi file
required_files = [
    "tfidf_vectorizer_baru.sav",
    "linear_model.sav",
    "poly_model.sav",
    "rbf_model.sav",
    "sigmoid_model.sav"
]

missing_files = [file for file in required_files if not os.path.exists(file)]
if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
else:
    # Load vectorizer dan model
    try:
        tfidf_vectorizer = joblib.load("tfidf_vectorizer_baru.sav")
        linear_model = joblib.load("linear_model.sav")
        poly_model = joblib.load("poly_model.sav")
        rbf_model = joblib.load("rbf_model.sav")
        sigmoid_model = joblib.load("sigmoid_model.sav")
    except Exception as e:
        st.error(f"Error loading models or vectorizer: {e}")

# Streamlit app
st.title("Sentiment Analysis with SVM")
st.sidebar.header("Kernel Selection")
kernel_choice = st.sidebar.selectbox("Choose Kernel:", ["Linear", "Polynomial", "RBF", "Sigmoid"])

st.header("Input Text for Sentiment Analysis")
input_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if input_text.strip():
        try:
            # Transform input text
            transformed_text = tfidf_vectorizer.transform([input_text])

            # Validasi dimensi dan pilih model
            selected_model = None
            if kernel_choice == "Linear":
                selected_model = linear_model
            elif kernel_choice == "Polynomial":
                selected_model = poly_model
            elif kernel_choice == "RBF":
                selected_model = rbf_model
            elif kernel_choice == "Sigmoid":
                selected_model = sigmoid_model

            if transformed_text.shape[1] != selected_model.n_features_in_:
                st.error("Input dimensions do not match the trained model. Ensure the correct vectorizer is used.")
            else:
                # Perform prediction
                prediction = selected_model.predict(transformed_text)
                st.write(f"Prediction using {kernel_choice} Kernel: {'Positive' if prediction[0] == 1 else 'Negative'}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some text.")
