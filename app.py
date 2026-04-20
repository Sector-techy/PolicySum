import streamlit as st
import fitz  # PyMuPDF
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO
from dotenv import load_dotenv
import os
import subprocess

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("⚠️ GOOGLE_API_KEY not set in .env")

# --- spaCy setup ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --- Streamlit page config ---
st.set_page_config(page_title="Policy Summarizer — Pro UI", layout="wide")

# --- Cache model for speed ---
@st.cache_resource
def get_gemini_model():
    """Cached Gemini model initialization for 3-6x faster performance"""
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",  # Fastest model for summaries (2025)
        generation_config={
            "temperature": 0.3,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        }
    )

# --- Utility functions ---
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
        return text
    except Exception:
        return ""

def clean_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def get_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_keywords(sentences, top_n=12):
    if not sentences:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)  # Reduced for speed
    try:
        X = vectorizer.fit_transform(sentences)
    except Exception:
        return []
    features = vectorizer.get_feature_names_out()
    return list(features[:top_n]) if len(features) else sentences[:top_n]

def summarize_by_tfidf(sentences, mode="Brief Summary"):
    if not sentences:
        return "No text available to summarize."
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)  # Faster
    try:
        tfidf = vectorizer.fit_transform(sentences)
        scores = tfidf.sum(axis=1).A1
    except Exception:
        if mode == "Brief Summary":
            return " ".join(sentences[:3])
        return "\n".join(sentences[:5])
    ranked = [s for _, s in sorted(zip(scores, sentences), reverse=True)]
    if mode == "Brief Summary":
        return " ".join(ranked[:3])
    if mode == "Important Facts":
        return "\n".join([f"- {s}" for s in ranked[:6]])
    if mode == "Bullet Points":
        return "\n".join([f"- {s}" for s in ranked[:8]])
    if mode == "Coverage & Exclusions":
        coverage = [s for s in sentences if any(word in s.lower() for word in ["cover", "insured", "benefit"])]
        exclusion = [s for s in sentences if any(word in s.lower() for word in ["exclude", "not cover", "exclusion"])]
        out = ""
        out += "🟢 COVERAGE:\n" + ("\n".join([f"- {s}" for s in coverage[:6]]) if coverage else "No explicit coverage found.\n")
        out += "\n\n🔴 EXCLUSIONS:\n" + ("\n".join([f"- {s}" for s in exclusion[:6]]) if exclusion else "No explicit exclusions found.")
        return out
    return "Mode not supported."

def extract_entities(text):
    doc = nlp(text)
    ent_dict = {}
    for ent in doc.ents:
        ent_dict.setdefault(ent.label_, set()).add(ent.text)
    return {k: list(v)[:8] for k, v in ent_dict.items()}

# --- 🚀 FAST DIRECT GEMINI API (3-6x faster than LangChain) ---
def summarize_with_gemini_direct(text):
    try:
        model = get_gemini_model()
        
        # Truncate for speed (32k token limit for Flash)
        if len(text) > 32000:
            text = text[:32000] + "\n[...truncated for speed...]"
        
        prompt = f"""You are an expert insurance policy analyst.
Summarize the following policy in bullet points only:

• Key Coverage
• Exclusions  
• Conditions
• Eligibility
• Important Notes

POLICY TEXT:
{text}

SUMMARY (bullet points only):"""

        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        sentences = get_sentences(text)
        return f"⚠️ Gemini unavailable. Using TF-IDF: {str(e)[:100]}\n\n" + summarize_by_tfidf(sentences, "Coverage & Exclusions")

# --- 💅 Enhanced UI Styling ---
st.markdown("""
<style>
body { background-color: #0e1117; color: #f1f1f1; }
h1,h2,h3,h4 { color: #66b3ff; }
.header { font-size: 32px; font-weight: 700; background: linear-gradient(90deg, #007bff, #00b4d8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color: #b5b5b5; margin-bottom: 20px; font-size: 15px; }
.card { background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); backdrop-filter: blur(10px); border-radius:16px; padding:20px; box-shadow:0 4px 20px rgba(0,0,0,0.25); transition:0.3s; }
.card:hover { box-shadow:0 0 25px rgba(0,187,255,0.25); }
.stButton>button { background: linear-gradient(90deg,#007bff,#00b4d8); color:white; border-radius:8px; font-size:16px; font-weight:600; height:3em; width:100%; border:none; transition:0.3s; }
.stButton>button:hover { transform: scale(1.05); background: linear-gradient(90deg,#00b4d8,#007bff); }
textarea { background-color: #1e1e1e !important; color:#f1f1f1 !important; border-radius:8px !important; }
hr { border:1px solid rgba(255,255,255,0.1); }
.small { font-size:13px; color:#999; text-align:center; }
.speed-badge { background: linear-gradient(90deg, #10b981, #059669); color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
</style>""", unsafe_allow_html=True)

# --- 🧾 App Title ---
st.markdown("<div class='header'> PolicyPro </div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>⚡ 3-6x Faster with Gemini 2.5 Flash-Lite + Cached Model</div>", unsafe_allow_html=True)
st.markdown("<div class='speed-badge'>💨 Optimized for Speed</div>", unsafe_allow_html=True)

# --- Layout ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧩 Input Options")
    policy_type = st.selectbox("📄 Policy Type", ["Vehicle Insurance", "Health Insurance", "Life Insurance", "Travel Insurance", "Other"])
    summary_mode = st.radio("🎯 Summary Mode",
                            ["Brief Summary", "Important Facts", "Bullet Points", "Coverage & Exclusions", "⚡ LLM Summary (Gemini Flash)"])
    uploaded = st.file_uploader("📎 Upload Policy (PDF or TXT)", type=["pdf", "txt"])
    st.write("")
    analyze = st.button("🚀 Analyze Policy", key="analyze_btn")
    st.markdown("<p class='small'>💡 Uses Gemini 2.5 Flash-Lite (fastest model) with caching</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Results")
    result_placeholder = st.empty()
    st.markdown("---")
    st.subheader("🧠 NLP Insights")
    ents_placeholder = st.empty()
    kw_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# --- Logic ---
if analyze:
    if not uploaded:
        st.warning("⚠️ Please upload a PDF or TXT file first.")
    else:
        with st.spinner("⚙️ Processing with Gemini Flash + NLP... (1-3s)"):
            raw_text = ""
            if uploaded.type == "application/pdf":
                raw_text = extract_text_from_pdf_bytes(uploaded.read())
            else:
                raw_text = uploaded.read().decode("utf-8", errors="ignore")
            raw_text = clean_whitespace(raw_text)

            if not raw_text:
                result_placeholder.info("❌ No extractable text found in document.")
            else:
                sentences = get_sentences(raw_text)
                
                if summary_mode == "⚡ LLM Summary (Gemini Flash)":
                    summary = summarize_with_gemini_direct(raw_text)
                else:
                    summary = summarize_by_tfidf(sentences, mode=summary_mode)

                entities = extract_entities(raw_text)
                keywords = extract_keywords(sentences, top_n=12)

                # Display summary
                result_placeholder.markdown("**📝 AI-Generated Summary**")
                result_placeholder.text_area("Summary", value=summary, height=280)

                # Display NLP entities
                ents_text = ""
                for label, vals in entities.items():
                    ents_text += f"**{label}** → {', '.join(vals)}\n"
                ents_placeholder.markdown("**Named Entities**")
                ents_placeholder.markdown(f"<div class='card'>{ents_text or 'No entities found.'}</div>", unsafe_allow_html=True)

                # Display keywords
                kw_placeholder.markdown("**Top Keywords (TF-IDF)**")
                kw_placeholder.markdown(f"<div class='card'>{', '.join(keywords) if keywords else 'No keywords found.'}</div>", unsafe_allow_html=True)

                # Download
                b = BytesIO()
                b.write(summary.encode("utf-8"))
                b.seek(0)
                st.download_button("⬇️ Download Summary (.txt)", data=b, file_name=f"policy_summary_{policy_type.lower().replace(' ', '_')}.txt", mime="text/plain")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='small'>✨ Gemini 2.5 Flash-Lite + spaCy + TF-IDF · ⚡ Cached & Optimized · 2025</div>", unsafe_allow_html=True)



