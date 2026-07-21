import streamlit as st
import pickle
import zipfile
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Phishing Mail Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(10, 17, 40, 0.85), rgba(10, 17, 40, 0.95)), 
                    url('https://images.unsplash.com/photo-1563986768609-322da13575f3?q=80&w=1600') no-repeat center center fixed;
        background-size: cover;
    }
    .header-title {
        font-family: 'Courier New', Courier, monospace;
        color: #00ffcc;
        font-size: 3.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 15px rgba(0, 255, 204, 0.4);
        letter-spacing: -1px;
    }
    .header-subtitle {
        color: #8892b0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    div[data-baseweb="tab-list"] {
        justify-content: center !important;
        background: transparent !important;
        border: none !important;
        gap: 15px;
        margin-bottom: 20px;
    }
    button[id^="tabs-bui"] {
        background-color: rgba(17, 34, 64, 0.9) !important;
        color: #64ffda !important;
        font-weight: 600 !important;
        padding: 10px 30px !important;
        border-radius: 4px !important;
        border: 1px solid #64ffda !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }
    button[id^="tabs-bui"][aria-selected="true"] {
        background-color: #64ffda !important;
        color: #0a1128 !important;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.6);
    }
    
    .stTextArea textarea {
        color: #ffffff !important;
        background-color: #0a1128 !important;
        border: 1px solid #233554 !important;
        font-size: 1.1rem !important;
        padding: 15px !important;
        border-radius: 6px !important;
    }
    .stTextArea textarea:focus {
        border-color: #64ffda !important;
    }
    
    .stButton button {
        background-color: #64ffda !important;
        color: #0a1128 !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        padding: 15px 40px !important;
        border-radius: 6px !important;
        border: none !important;
        height: 100% !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(100, 255, 218, 0.3);
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #4cd1b0 !important;
        transform: translateY(-2px);
    }
    .result-container {
        text-align: center;
        margin: 2rem auto;
        padding: 20px;
        background: #112240;
        border-radius: 8px;
        max-width: 650px;
        border: 1px solid #233554;
    }
    .result-text {
        font-family: 'Arial', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .result-phishing { color: #ff4c4c !important; text-shadow: 0 0 10px rgba(255, 76, 76, 0.5); }
    .result-safe { color: #00ffcc !important; text-shadow: 0 0 10px rgba(0, 255, 204, 0.5); }
    
    .footer-section {
        background-color: #020c1b;
        color: #ccd6f6;
        padding: 40px;
        margin-top: 6rem;
        border-top: 1px solid #233554;
    }
    </style>
    """, unsafe_allow_html=True)

# Load pipeline models from ZIP archive
@st.cache_resource
def load_assets():
    with zipfile.ZipFile('phishing_models.zip', 'r') as z:
        with z.open('phishing_models.pkl') as f:
            models = pickle.load(f)
            
    with open('phishing_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    return models, vectorizer

try:
    models, vectorizer = load_assets()
except Exception as e:
    st.error(f"Asset loading failed: {e}. Ensure 'phishing_models.zip' and 'phishing_vectorizer.pkl' exist in your directory.")
    st.stop()

STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def preprocess_email_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([w for w in words if w not in STOPWORDS])

st.markdown('<h1 class="header-title">🛡️ PHISHING MAIL SHIELD</h1>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Ensemble Deep Learning & Meta-Feature Extraction Architecture</div>', unsafe_allow_html=True)

tab_mail, tab_logs = st.tabs(["Analyze Email Content", "System Information"])

user_input = ""
with tab_mail:
    user_input = st.text_area("Email Content Input", placeholder="Paste the complete body content of the email here to execute threat assessment...", height=220, label_visibility="collapsed")
    
    col_space, col_btn = st.columns([4, 1])
    with col_btn:
        predict_clicked = st.button("Scan Content")

with tab_logs:
    st.markdown("""
        <div style='color:#8892b0; padding:10px;'>
            <strong>Pipeline Configuration Stack:</strong><br>
            • Feature Extraction Layer: TF-IDF Vectorizer (2,500 Max Features)<br>
            • Meta Features: URL Hyperlink Tracking + High-Urgency Lexicon Scanning<br>
            • Core Classifiers: Multi-Domain Voting Consensus (Logistic Regression, Random Forest, Multi-Layer Precision Model)<br>
            • Target Framework Deployment Status: Operational
        </div>
    """, unsafe_allow_html=True)

if predict_clicked:
    if user_input.strip() == "":
        st.warning("Please provide valid text content inside the tracking block before running analysis.")
    else:
        has_url = 1 if re.search(r'https?://|www\.', str(user_input)) else 0
        has_urgency = 1 if any(w in str(user_input).lower() for w in ['urgent', 'verify', 'suspend', 'action', 'password', 'login', 'bank', 'account']) else 0

        cleaned_text = preprocess_email_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()

        meta_array = np.array([[has_url, has_urgency]])
        combined_input = np.hstack([vectorized_text, meta_array])

        votes = []
        for name in ["Logistic Regression", "Random Forest", "Neural Network"]:
            pred = models[name].predict(combined_input)[0]
            votes.append(int(pred))

        final_vote = 1 if votes.count(1) > votes.count(0) else 0

        if final_vote == 1:
            st.markdown("""
                <div class="result-container">
                    <span class="result-text">Analysis Result: <span class="result-phishing">⚠️ PHISHING DETECTED</span></span>
                    <div style='color:#8892b0; font-size:0.9rem; margin-top:10px;'>The ensemble pipeline detected structural traits matching email scams or fraudulent domain anchors.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-container">
                    <span class="result-text">Analysis Result: <span class="result-safe">✅ SAFE / LEGITIMATE</span></span>
                    <div style='color:#8892b0; font-size:0.9rem; margin-top:10px;'>No high-risk malicious vector profiles were found across the text structure consensus.</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("""
    <div class="footer-section">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 50px;">
            <div>
                <h3 style="color:#64ffda; font-size:1.4rem; margin-bottom:15px; font-family:monospace;">Developer Portfolio</h3>
                <p style="color:#8892b0; margin: 4px 0;">Principal: @ Christy Joyce A</p>
                <p style="color:#8892b0; margin: 4px 0;">contact: christyjoyce254@gmail.com</p>
            </div>
            <div>
                <h3 style="color:#64ffda; font-size:1.4rem; margin-bottom:15px; font-family:monospace;">Ensemble Methodology</h3>
                <p style="color:#8892b0; line-height:1.6; font-size:0.95rem;">
                    This defense portal monitors communication vectors by integrating vocabulary matrices alongside contextual metadata indicators. Threats are neutralized through a synchronized vote across distributed model architectures.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)