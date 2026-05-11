import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import matplotlib.pyplot as plt

# Download once (safe in Streamlit)
nltk.download('vader_lexicon')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Text Analyzer App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🧠 AI Text Analyzer App")

menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Analyze", "ℹ️ About"]
)

# ---------------- NLP SETUP ----------------
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ======================================================
# 🏠 HOME PAGE
# ======================================================
if menu == "🏠 Home":
    st.title("Welcome to AI Text Analyzer App 🚀")
    st.write("""
    This app helps you:
    - 🧠 Analyze sentiment of text
    - 🔍 Extract keywords
    - 📊 Visualize results
    - 📂 Upload text files
    
    Built using **Streamlit + NLTK VADER**
    """)

    st.success("Go to Analyze section from sidebar 👈")

# ======================================================
# 📊 ANALYZE PAGE
# ======================================================
elif menu == "📊 Analyze":
    st.title("📊 Text Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        text = st.text_area("✍️ Enter your text", height=250)

    with col2:
        uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

    if st.button("🚀 Analyze"):

        if not text.strip():
            st.warning("Please enter text")
        else:

            sentiment = sia.polarity_scores(text)
            compound = sentiment["compound"]

            st.subheader("📊 Results")

            # ---------------- METRICS ----------------
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentiment Score", round(compound, 2))
            c2.metric("Words", len(text.split()))
            c3.metric("Characters", len(text))
            c4.metric("Sentences", text.count("."))

            st.markdown("---")

            # ---------------- SENTIMENT LABEL ----------------
            st.subheader("🎯 Sentiment Result")

            if compound >= 0.05:
                st.success("😊 Positive Sentiment")
            elif compound <= -0.05:
                st.error("😡 Negative Sentiment")
            else:
                st.info("😐 Neutral Sentiment")

            # ---------------- PIE CHART ----------------
            st.subheader("📊 Sentiment Distribution")

            labels = ["Positive", "Neutral", "Negative"]
            values = [sentiment["pos"], sentiment["neu"], sentiment["neg"]]

            fig1, ax1 = plt.subplots()
            ax1.pie(values, labels=labels, autopct="%1.1f%%")
            st.pyplot(fig1)

            # ---------------- KEYWORDS ----------------
            st.subheader("🔍 Top Keywords")

            words = clean_text(text).split()

            stopwords = {
                "the","is","and","to","a","of","in","it","this","that",
                "for","on","with","as","are","was","were","be","been"
            }

            filtered = [w for w in words if w not in stopwords and len(w) > 2]

            freq = Counter(filtered)
            common = freq.most_common(10)

            if common:
                words_list = [w for w, _ in common]
                counts = [c for _, c in common]

                fig2, ax2 = plt.subplots()
                ax2.bar(words_list, counts)
                plt.xticks(rotation=45)
                st.pyplot(fig2)
            else:
                st.write("No strong keywords found")

            # ---------------- RAW DATA ----------------
            st.subheader("📊 Detailed Sentiment Data")
            st.json(sentiment)

# ======================================================
# ℹ️ ABOUT PAGE
# ======================================================
elif menu == "ℹ️ About":
    st.title("ℹ️ About This App")

    st.write("""
    **AI Text Analyzer App**

    Features:
    - Sentiment Analysis using NLTK VADER
    - Keyword extraction
    - Data visualization (charts)
    - File upload support

    Built by an AI Engineering student 🚀
    """)

    st.success("Great project for Resume + LinkedIn 💼")