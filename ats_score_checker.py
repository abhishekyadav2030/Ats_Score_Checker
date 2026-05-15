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

# Download NLTK data (run once)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_data()

# ---------- Text Extraction ----------
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
        st.error("Unsupported file format. Please upload PDF or DOCX.")
        return None

# ---------- Text Preprocessing ----------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and digits (keep letters and spaces)
    text = re.sub(f'[{re.escape(string.punctuation)}0-9]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ---------- Keyword Extraction from JD ----------
def extract_keywords_from_jd(jd_text, top_n=20):
    processed = preprocess(jd_text)
    words = processed.split()
    freq_dist = nltk.FreqDist(words)
    # Return most common words as key terms
    return [word for word, _ in freq_dist.most_common(top_n)]

# ---------- Scoring Functions ----------
def keyword_match_score(resume_text, jd_keywords):
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    score = (len(matched) / len(jd_keywords)) * 100 if jd_keywords else 0
    return score, matched, missing

def cosine_similarity_score(resume_text, jd_text):
    # Preprocess both texts
    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(jd_text)
    
    if not processed_resume or not processed_jd:
        return 0.0
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100  # as percentage

def compute_ats_score(resume_text, jd_text):
    # Extract key JD keywords
    jd_keywords = extract_keywords_from_jd(jd_text, top_n=20)
    
    # Keyword score (60% weight)
    kw_score, matched, missing = keyword_match_score(resume_text, jd_keywords)
    
    # Cosine similarity score (40% weight)
    cos_score = cosine_similarity_score(resume_text, jd_text)
    
    # Final weighted score
    final_score = (0.6 * kw_score) + (0.4 * cos_score)
    return final_score, kw_score, cos_score, matched, missing, jd_keywords

# ---------- Streamlit UI ----------
st.set_page_config(page_title="ATS Score Checker", page_icon="📄")
st.title("📄 ATS Resume Score Checker")
st.markdown("Upload your resume (PDF/DOCX) and paste the job description to get an ATS compatibility score.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📎 Upload Resume", type=["pdf", "docx"], help="PDF or DOCX only")
    
with col2:
    jd_text = st.text_area("📝 Paste Job Description", height=200, placeholder="Paste the full job description here...")

if st.button("🔍 Check ATS Score", type="primary"):
    if uploaded_file is None or not jd_text.strip():
        st.error("Please provide both a resume file and a job description.")
    else:
        with st.spinner("Analyzing your resume against the job description..."):
            # Extract resume text
            resume_text = extract_resume_text(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from the uploaded file. Ensure it contains readable text.")
                st.stop()
            
            # Compute scores
            final_score, kw_score, cos_score, matched, missing, jd_keywords = compute_ats_score(resume_text, jd_text)
            
            # Display main score
            st.subheader("📊 ATS Compatibility Score")
            st.progress(final_score / 100)
            st.metric("Overall Score", f"{final_score:.1f}%")
            
            # Detailed breakdown
            with st.expander("📈 Score Breakdown"):
                st.metric("Keyword Match Score (60%)", f"{kw_score:.1f}%")
                st.metric("Content Similarity Score (40%)", f"{cos_score:.1f}%")
            
            # Keyword analysis
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("✅ Matched Keywords")
                if matched:
                    st.write(", ".join(matched))
                else:
                    st.write("No keywords matched.")
            
            with col_b:
                st.subheader("❌ Missing Keywords")
                if missing:
                    st.write(", ".join(missing))
                else:
                    st.write("All key terms found – great!")
            
            # Improvement suggestions
            st.subheader("💡 Suggestions for Improvement")
            if missing:
                st.markdown("**Add these missing keywords to your resume:**")
                for kw in missing[:10]:
                    st.markdown(f"- `{kw}`")
                st.markdown("---")
                st.markdown("**Tips:**")
                st.markdown("1. Incorporate missing keywords naturally into your **Skills**, **Experience**, or **Summary** sections.")
                st.markdown("2. Use standard section headings like `Work Experience`, `Education`, `Technical Skills`.")
                st.markdown("3. Avoid images, tables, or complex formatting – ATS may not read them.")
            else:
                st.success("Excellent keyword coverage! Your resume is well optimised for this role.")
            
            # Show extracted text preview (optional)
            with st.expander("📄 Extracted Resume Text (first 500 chars)"):
                st.text(resume_text[:500] + ("..." if len(resume_text) > 500 else ""))

# Footer
st.markdown("---")
st.caption("🔍 ATS Score is based on keyword overlap (60%) and TF‑IDF cosine similarity (40%). For best results, use a clean text‑based resume.")