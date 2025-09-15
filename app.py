import streamlit as st
import joblib
import re
from deep_translator import GoogleTranslator
import base64
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
            background-color: rgba(255,255,255,0.5); /* 🔥 এটাতেই হালকা হবে */
            z-index: 0;
        }}
        .stApp > div {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# -------------------
# Utility: text clean
# -------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove urls
    text = re.sub(r"[^a-z\s]", "", text)                # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------
# Load artifacts
# -------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier.pkl")
    vect = joblib.load("tfidf_vectorizer.pkl")
    return model, vect

model, vectorizer = load_model()

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Spam Classifier + Translator", layout="centered")
add_bg_from_local("background.jpg")
st.markdown(
    """
    <style>
    .top-right-buttons {
        position: absolute;
        top: 10px;
        right: 20px;
    }
    .top-right-buttons a {
        text-decoration: none;
        margin-left: 15px;
        padding: 8px 16px;
        border-radius: 8px;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    .top-right-buttons a:hover {
        background-color: #004080;
    }
    </style>

    <div class="top-right-buttons">
        <a href="#">About</a>
        <a href="#">Login</a>
        <a href="#">Sign Up</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: yellow;'>📨 Spam vs Ham Classifier</h1>", 
    unsafe_allow_html=True
)
st.write("Write a message below and click *Predict* to check if it is Spam or Ham.")

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

        # Show result
        st.subheader("Prediction Result")
        if pred == 0:
            st.success(label)
        else:
            st.error(label)
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

target_lang = st.selectbox("Choose language:", list(languages.keys()))

if st.button("Translate"):
    translated = GoogleTranslator(source='auto', target=languages[target_lang]).translate(user_input)
    st.success(f"{target_lang} Translation:** {translated}")

# -------------------
# Credits section
# -------------------
st.markdown("---")
st.write("### 📌 Credits")
st.write("""
- Developed by: *[kartick dey]*
- Dataset: [SMS Spam Collection Dataset (UCI/Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Model: TF-IDF + Naive Bayes
""")