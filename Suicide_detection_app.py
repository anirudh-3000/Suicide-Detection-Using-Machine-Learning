import streamlit as st
import pickle
import numpy as np  
from sklearn.naive_bayes import GaussianNB  #GaussianNB gave highest accuracy

# Load the model and vectorizer
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf (1).pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Define the prediction function
def predict(text):
    # Preprocess the text using the loaded vectorizer
    features = vectorizer.transform([text])  # Transform returns a sparse matrix
    features = features.toarray()  # Convert to dense array if model expects it
    
    # Make prediction
    prediction = model.predict(features)  # e.g., array(['Non-suicide']) or array([0])
    prediction_proba = model.predict_proba(features)  # e.g., [[0.7, 0.3]]
    
    # Get the predicted class label
    label = prediction[0]
    
    # Find the index of the predicted class in model.classes_
    # This works regardless of the type of class labels
    class_index = np.where(model.classes_ == label)[0]
    
    if len(class_index) == 0:
        # Handle case where class label is not found
        probability = None
    else:
        class_index = class_index[0]
        # Get the probability of the predicted class
        probability = prediction_proba[0][class_index]
    
    # Map the label to the desired string
    if label == 0 or label == '0' or label == 'Non-suicide':
        label_str = "Non-suicide"
    elif label == 1 or label == '1' or label == 'Suicide':
        label_str = "Suicide"
    else:
        label_str = str(label)  # Fallback to the label's string representation
    
    return label_str, probability

# Create the Streamlit app
st.title("Suicide Detection App")
st.write("Enter text to analyze:")

# Get user input
text = st.text_area("")

# Make prediction when user clicks button
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        result, prob = predict(text)
        if prob is not None:
            st.write(f"**Prediction:** {result}")
            #st.write(f"**Confidence:** {prob*100:.2f}%")
        else:
            st.error("Unable to determine the class probability.")

