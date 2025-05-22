import streamlit as st
import joblib
import pandas as pd
import spacy
import gdown
import os

# تحميل spaCy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# تحميل النموذج من Google Drive مرة وحدة فقط
MODEL_URL = "https://drive.google.com/uc?id=1W1KbcJm5MwEQj8DeZ_CkRYEpTlJpqr33"
MODEL_PATH = "best_model_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# تحميل النموذج المدرب
model = joblib.load(MODEL_PATH)

# دالة معالجة النصوص
def preprocess_text(text):
    if pd.isnull(text) or text == "":
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# إنشاء DataFrame من الإدخال
def create_input_df(review_text, age=30, pos_feedback=0):
    return pd.DataFrame({
        'Clothing ID': [0],
        'Age': [age],
        'Title': [""],
        'Review Text': [preprocess_text(review_text)],
        'Positive Feedback Count': [pos_feedback],
        'Division Name': ["General"],
        'Department Name': ["Tops"],
        'Class Name': ["Shirts"]
    })

# واجهة Streamlit
st.title("🛍️ Product Review Recommendation Predictor")

review = st.text_area("Enter the product review:")
age = st.slider("Select your age:", 10, 100, 30)
pos_feedback = st.number_input("Positive feedback count:", min_value=0, value=0)

if st.button('Predict'):
    if review:
        try:
            input_data = create_input_df(review, age, pos_feedback)
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)

            label = "✅ Recommended" if prediction[0] == 1 else "❌ Not Recommended"
            confidence = proba[0][prediction[0]] * 100

            st.success(f"Prediction: {label} ({confidence:.1f}% confidence)")
            st.write("Class Probabilities:")
            st.write(f"- Recommended: {proba[0][1]*100:.1f}%")
            st.write(f"- Not Recommended: {proba[0][0]*100:.1f}%")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    else:
        st.warning("Please enter a review to continue.")
