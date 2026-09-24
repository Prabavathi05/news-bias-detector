import streamlit as st
import joblib
import time
import os
from google import genai

# ---------- Page setup ----------
st.set_page_config(page_title="News Article Bias Detector", page_icon="📰", layout="centered")
st.title("📰 News Article Bias Detector")
st.write("Paste a news article below to predict its political leaning (Left / Center / Right), with an AI-generated explanation.")

# ---------- Load model + vectorizer (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    clf = joblib.load("bias_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return clf, vectorizer

clf, vectorizer = load_model()

# ---------- Gemini client setup ----------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def get_gemini_client():
    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    return None

client = get_gemini_client()

# ---------- Explanation function with retry + graceful fallback ----------
def explain_bias(article_text, predicted_label, retries=2, delay=3):
    if client is None:
        return "Explanation unavailable (Gemini API key not configured)."

    prompt = f"""You are analyzing a news article that has been classified as politically "{predicted_label}"-leaning.

Article text:
\"\"\"{article_text[:2000]}\"\"\"

Task: In 3-4 sentences, explain WHY this article might be classified as {predicted_label}-leaning.
Point to specific word choices, framing, or phrases from the article itself as evidence.
Do not use generic statements — cite actual language from the text.
"""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            return response.text
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return "⚠️ Explanation temporarily unavailable — the AI service may be busy. Please try again shortly."

# ---------- Main app UI ----------
article_text = st.text_area("Paste article text here:", height=250)

if st.button("Analyze Bias"):
    if not article_text.strip():
        st.warning("Please paste some article text first.")
    else:
        with st.spinner("Analyzing..."):
            text_tfidf = vectorizer.transform([article_text])
            prediction = clf.predict(text_tfidf)[0]
            probabilities = clf.predict_proba(text_tfidf)[0]
            confidence = max(probabilities) * 100

            st.subheader(f"Predicted Bias: **{prediction.upper()}**")
            st.write(f"Confidence: {confidence:.1f}%")

            prob_dict = dict(zip(clf.classes_, probabilities))
            st.bar_chart(prob_dict)

            with st.spinner("Generating explanation..."):
                explanation = explain_bias(article_text, prediction)
            st.subheader("Why this prediction?")
            st.write(explanation)

st.markdown("---")
st.caption("Built with scikit-learn (TF-IDF + Logistic Regression) and Gemini API. Trained on the AllSides news bias dataset.")
