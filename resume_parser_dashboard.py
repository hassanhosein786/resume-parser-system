import streamlit as st
import pandas as pd
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Tell NLTK where to look for pre-downloaded data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Preprocess the text
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z0-9\s\-\+]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)
    except Exception:
        return ""

@st.cache_data
def load_and_process_data(file):
    data = pd.read_csv(file)
    if "Resume" not in data.columns or "Category" not in data.columns:
        st.error("Your CSV must include 'Resume' and 'Category' columns.")
        return None, None
    data.dropna(subset=["Resume", "Category"], inplace=True)
    data["Resume"] = data["Resume"].astype(str)
    data["Category"] = data["Category"].astype(str).str.lower().str.strip()
    data["Cleaned_Resume"] = data["Resume"].apply(preprocess)
    data = data[data["Cleaned_Resume"].str.strip().astype(bool)]

    if data.empty:
        st.error("All resumes became empty after cleaning.")
        return None, None

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data["Cleaned_Resume"])
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(X)
    return data, vectorizer

def score_resume(resume, job_title, vectorizer):
    if not resume or not job_title:
        return 0.0
    job_cleaned = preprocess(job_title)
    try:
        job_vec = vectorizer.transform([job_cleaned])
        resume_vec = vectorizer.transform([resume])
        score = cosine_similarity(resume_vec, job_vec)[0][0]
        return round(score * 10, 2)
    except:
        return 0.0

# Streamlit Frontend
st.set_page_config(page_title="Resume Parser Dashboard", layout="wide")
st.title("📄 Resume Parser System")
st.write("Analyze and score resumes automatically using NLP and Machine Learning.")

uploaded_file = st.file_uploader("📤 Upload Resume CSV", type=["csv"])

if uploaded_file is not None:
    data, vectorizer = load_and_process_data(uploaded_file)

    if data is not None:
        st.subheader("📋 Dataset Overview")
        st.dataframe(data[["Category", "Resume"]].head())

        if st.button("⚙️ Score Resumes"):
            data["Score"] = data.apply(lambda x: score_resume(x["Cleaned_Resume"], x["Category"], vectorizer), axis=1)

            st.success("Scoring completed!")

            st.subheader("🎯 Scored Resumes")
            st.dataframe(data[["Category", "Score"]].sort_values(by="Score", ascending=False))

            st.subheader("📊 Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(data["Score"], bins=10, color="skyblue", edgecolor="black")
            ax.set_xlabel('Score')
            ax.set_ylabel('Number of Resumes')
            ax.set_title('Resume Score Distribution')
            st.pyplot(fig)

            st.subheader("📈 Average Score by Category")
            avg_score = data.groupby("Category")["Score"].mean().sort_values(ascending=False)
            fig2, ax2 = plt.subplots()
            avg_score.plot(kind='bar', color='teal', ax=ax2)
            ax2.set_ylabel('Average Score')
            ax2.set_title('Average Resume Score per Category')
            st.pyplot(fig2)

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Scored CSV",
                data=csv,
                file_name='scored_resumes.csv',
                mime='text/csv',
            )
else:
    st.info('Please upload a resume dataset file to begin.')
