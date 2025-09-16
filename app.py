import streamlit as st
import joblib
import re
from deep_translator import GoogleTranslator
import base64

# -----------------------------
# Session State Initialization
# -----------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "users" not in st.session_state:
    st.session_state["users"] = {}  # Demo user storage
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

# -----------------------------
# Background Setup
# -----------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            position: relative;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255,255,255,0.5);
            z-index: 0;
        }}
        .stApp > div {{
            position: relative;
            z-index: 1;
        }}

        .result-box {{
            background: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            font-size: 20px;
            margin-top: 15px;
            color: #fff;
            font-weight: bold;
            opacity: 0;
            animation: fadeIn 1s forwards;
        }}

        .translate-box {{
            background: rgba(0, 123, 255, 0.15);
            border-left: 6px solid #007bff;
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
            font-size: 18px;
            font-weight: 500;
            color: #003366;
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
            opacity: 0;
            animation: fadeIn 1s forwards;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Custom CSS for Menu styling
# -----------------------------
st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #B71C1C !important;   /* গাঢ় লাল */
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Utility: text clean
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier.pkl")
    vect = joblib.load("tfidf_vectorizer.pkl")
    return model, vect

model, vectorizer = load_model()

# -----------------------------
# Languages for Translation
# -----------------------------
languages = {
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# -----------------------------
# Top Navigation (Hamburger menu)
# -----------------------------
with st.expander("☰ Menu"):
    if st.button("Home"):
        st.session_state["page"] = "home"
    if st.button("About"):
        st.session_state["page"] = "about"
    if st.button("Login"):
        st.session_state["page"] = "login"
    if st.button("Sign Up"):
        st.session_state["page"] = "signup"

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state["page"] == "home":
    st.set_page_config(page_title="Spam Classifier + Translator", layout="centered")
    add_bg_from_local("background.jpg")

    st.markdown(
        "<h1 style='text-align: center; color:#FFD700;'>🤖 SpamSens AI</h1>",
        unsafe_allow_html=True
    )
    st.write("Write a message below and click Predict to check if it is Spam or Ham.")

    if st.session_state["logged_in"]:
        st.success(f"✅ Logged in as {st.session_state['user_email']}")
        if st.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = None
            st.success("You have been logged out.")
    else:
        st.info("ℹ Please login or sign up to access all features.")

    # Input box
    user_input = st.text_area("✍ Enter your message:", height=150)

    # Predict button
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter a message first!")
        else:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            label = "🟢 Ham (Not Spam)" if pred == 0 else "🔴 Spam"

            st.markdown(f"<div class='result-box'>{label}</div>", unsafe_allow_html=True)

    # Translator
    target_lang = st.selectbox("Choose language:", list(languages.keys()))
    if st.button("Translate"):
        if not user_input.strip():
            st.warning("Please enter a message first!")
        else:
            translated = GoogleTranslator(source='auto', target=languages[target_lang]).translate(user_input)
            st.markdown(
                f"""
                <div class='translate-box'>
                    🌐 <b>{target_lang} Translation:</b><br>
                    {translated}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Credits
    st.markdown("---")
    st.write("### 📌 Credits")
    st.write("""
    - Developed by: [kartick dey]  
    - Dataset: [SMS Spam Collection Dataset (UCI/Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
    - Model: TF-IDF + Naive Bayes
    """)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif st.session_state["page"] == "about":
    st.title("ℹ About SpamSens AI")
    st.write("""
    SpamSens AI is a demo application built with Streamlit.  
    Features include:
    - Email spam classification (detect spam vs. ham)  
    - Simple text translation  
    - Login & Sign Up system (demo only, not secure for production)  

    This project is created for learning and demonstration purposes.
    """)
    if st.button("⬅ Back"):
        st.session_state["page"] = "home"

# -----------------------------
# LOGIN PAGE
# -----------------------------
elif st.session_state["page"] == "login":
    st.title("🔑 Login")

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        login_sub = st.form_submit_button("Login")

    if login_sub:
        if not email or not password:
            st.error("Please enter both email and password.")
        else:
            users = st.session_state["users"]
            if email in users and users[email] == password:
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                st.success(f"Login successful! Welcome, {email}")
                st.session_state["page"] = "home"
            else:
                st.error("Invalid email or password.")

    if st.button("⬅ Back"):
        st.session_state["page"] = "home"

# -----------------------------
# SIGN UP PAGE
# -----------------------------
elif st.session_state["page"] == "signup":
    st.title("📝 Sign Up")

    with st.form("signup_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        if not email or not password:
            st.error("Please enter both email and password.")
        elif password != password2:
            st.error("Passwords do not match.")
        elif email in st.session_state["users"]:
            st.warning("This email is already registered. Please log in instead.")
        else:
            st.session_state["users"][email] = password  # demo only
            st.success("Account created successfully! You can now log in.")
            st.info("Note: In production, passwords must be stored securely (hashed).")

    if st.button("⬅ Back"):
        st.session_state["page"] = "home"