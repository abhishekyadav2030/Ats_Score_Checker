import streamlit as st
import pdfplumber
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_data()

# ---------- Black & White CSS ----------
def set_bw_css():
    st.markdown("""
    <style>
    .stApp {
        background: #ffffff;
    }
    h1, h2, h3, p, li, span, label {
        color: #000000 !important;
    }
    .stButton button {
        background: #000000;
        color: white;
        border-radius: 20px;
        border: 1px solid #000000;
    }
    .stButton button:hover {
        background: #333333;
    }
    .stFileUploader, .stTextArea textarea {
        background: #f5f5f5;
        border: 1px solid #cccccc;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background: #000000;
    }
    .streamlit-expanderHeader {
        background: #f0f0f0;
        border-radius: 10px;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- Text extraction (same) ----------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_resume_text(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("❌ Unsupported file")
        return None

# ---------- Preprocessing ----------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def extract_keywords_from_jd(jd_text, top_n=20):
    processed = preprocess(jd_text)
    words = processed.split()
    freq_dist = nltk.FreqDist(words)
    return [word for word, _ in freq_dist.most_common(top_n)]

def keyword_match_score(resume_text, jd_keywords):
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    score = (len(matched) / len(jd_keywords)) * 100 if jd_keywords else 0
    return score, matched, missing

def cosine_similarity_score(resume_text, jd_text):
    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(jd_text)
    if not processed_resume or not processed_jd:
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100

def compute_ats_score(resume_text, jd_text):
    jd_keywords = extract_keywords_from_jd(jd_text, top_n=20)
    kw_score, matched, missing = keyword_match_score(resume_text, jd_keywords)
    cos_score = cosine_similarity_score(resume_text, jd_text)
    final_score = (0.6 * kw_score) + (0.4 * cos_score)
    return final_score, kw_score, cos_score, matched, missing

# ---------- Main ----------
def main():
    set_bw_css()
    
    st.markdown("<h1>📄 ATS Score Checker</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📎 Upload resume (PDF/DOCX)", type=["pdf", "docx"])
    jd_text = st.text_area("📝 Job description", height=200)
    
    if st.button("🔍 Check Score"):
        if not uploaded_file or not jd_text.strip():
            st.warning("⚠️ Please upload file and paste JD")
        else:
            resume_text = extract_resume_text(uploaded_file)
            if not resume_text:
                st.error("❌ Could not read text")
                st.stop()
            
            final_score, kw_score, cos_score, matched, missing = compute_ats_score(resume_text, jd_text)
            
            st.metric("🏆 ATS Score", f"{final_score:.1f}%")
            st.progress(final_score / 100)
            
            col1, col2 = st.columns(2)
            col1.metric("🎯 Keyword match", f"{kw_score:.1f}%")
            col2.metric("🧠 Similarity", f"{cos_score:.1f}%")
            
            st.subheader("✅ Matched")
            st.write(", ".join(matched) if matched else "None")
            
            st.subheader("❌ Missing")
            st.write(", ".join(missing) if missing else "None")
            
            if missing:
                st.info("💡 Add missing keywords to resume")
            else:
                st.success("🎉 Perfect!")
            
            with st.expander("📄 Preview text"):
                st.text(resume_text[:500])

if __name__ == "__main__":
    main()