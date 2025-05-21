# 🌟 Product Review Recommendation Predictor

Welcome to the **Product Review NLP Dashboard**, a machine learning project that analyzes customer reviews and predicts whether a product is **Recommended** or **Not Recommended** — all powered by Natural Language Processing (NLP) and classification techniques.

> 🧠 This project was created as part of the **Data Scientist Nanodegree** by **Udacity**. Huge thanks to them for the learning experience and support!

---

## 🚀 Getting Started

To get a local copy up and running, follow the instructions below.

### 📦 Dependencies

Before running the app, make sure you have the following installed:

- Python 3.9+
- pip
- spaCy
- Streamlit
- scikit-learn
- pandas
- numpy
- joblib
- dill
  
# Install them with:
Clone repository: ``` git clone [repository-url] cd [repository-name] ```
Create virtual environment: ```  python -m venv venv venv\Scripts\activate ```
Install dependencies: ``` pip install -r requirements.txt ```

## 🧪 Testing

This project does not include automated testing scripts, but model performance was evaluated using:

- ✅ Classification report  
- ✅ Confusion matrix  
- ✅ Accuracy, precision, recall, and F1-score metrics  

---

## 📊 Project Instructions

- Cleaned and explored real customer reviews dataset  
- Preprocessed text columns using **spaCy** (lemmatization, stopword removal)  
- Built a full `scikit-learn` pipeline combining:
  - `StandardScaler` for numerical features  
  - `OneHotEncoder` for categorical features  
  - `TfidfVectorizer` for text columns  
- Trained a `RandomForestClassifier` and optimized it using `GridSearchCV`  
- Saved the final trained pipeline using `joblib`  

---

## 🛠 Built With

| Tool        | Description                              |
|-------------|------------------------------------------|
| 🧠 scikit-learn | ML modeling and pipelines              |
| ✍️ spaCy        | Text preprocessing and lemmatization   |
| 📊 pandas, numpy | Data wrangling and transformation     |
| 🎨 matplotlib, seaborn | Data visualization             |
| 🌐 Streamlit    | Web interface for interactive prediction |

---

## 📜 License

[License](LICENSE.txt)


