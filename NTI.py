import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import base64

# === Page Configuration ===
st.set_page_config(page_title="Heart Disease Detector", layout="wide")

# === Custom Background ===
def set_custom_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-attachment: fixed;
        background-color: rgba(0, 0, 0, 0.65); /* Slightly darker overlay for better contrast with the electric heart background */
        background-blend-mode: overlay;
    }}
    .title-text {{
        text-align: center;
        padding: 20px;
        font-size: 60px;
        font-weight: bold;
        color: #ffffff; /* Bright white text */
        text-shadow: 4px 4px 20px rgba(0, 255, 255, 0.95), 0 0 30px rgba(0, 191, 255, 0.9); /* Enhanced neon glow to match the electric theme */
        background: linear-gradient(90deg, #00ffff, #00ccff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow-text 8s ease infinite;
    }}
    @keyframes glow-text {{
        0% {{background-position: 0% 50%;}}
        50% {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}
    .info-box {{
        background-color: rgba(255, 255, 255, 0.95); /* Opaque white for readability */
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(0, 191, 255, 0.8); /* Stronger cyan border */
        margin-bottom: 20px;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.5);
        color: #333333; /* Darker text for contrast */
    }}
    .result-box {{
        background-color: rgba(0, 128, 192, 0.95); /* Darker teal with higher opacity */
        padding: 20px;
        border-radius: 15px;
        color: #ffffff; /* Bright white text */
        border: 2px solid rgba(0, 255, 255, 0.95); /* Stronger neon cyan border */
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.7);
    }}
    .stButton>button {{
        background-color: #00ccff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.6); /* Stronger shadow for button text */
    }}
    .stButton>button:hover {{
        background-color: #0099ff;
        transform: scale(1.05);
    }}
    .stText, .stMarkdown, .stExpander {{
        color: #ffffff; /* Ensure all text is white for contrast */
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8); /* Enhanced shadow for general text */
    }}
    .stInfo {{
        background-color: rgba(0, 0, 0, 0.75); /* Darker info box background */
        color: #00ffff; /* Bright cyan text to match the theme and improve visibility */
        padding: 10px;
        border-radius: 10px;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.6), 0 0 10px rgba(0, 255, 255, 0.5); /* Added glow effect */
        border: 1px solid rgba(0, 191, 255, 0.8); /* Subtle border for definition */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Background image path (change to your actual path)
set_custom_background("C:/Users/SASOO/Downloads/aaaa.jpg")

# === Model Configuration ===
MODEL_PATH = "D:/NTI Project/Heart Beat { MobileNetV2_97% }/Heart_Beat_MobileNetV2.keras"

CLASS_NAMES = [
    "Normal",
    "Premature Beat",
    "Atrial Fibrillation",
    "Myocardial Infarction",
    "Ventricular Tachycardia"
]

REPORTS = {
    "Normal": "💚 The heart rhythm appears normal. No signs of cardiac abnormality.",
    "Premature Beat": "💛 Premature beats detected. May be benign but monitor if frequent.",
    "Atrial Fibrillation": "🧡 Irregular rhythm observed. Can increase stroke risk. Medical follow-up recommended.",
    "Myocardial Infarction": "❤️ Possible heart attack symptoms. Immediate clinical evaluation needed.",
    "Ventricular Tachycardia": "🔴 Dangerous ventricular arrhythmia detected. Requires emergency intervention."
}

# === Load the Model ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === Title ===
st.markdown("<h1 class='title-text'>CardioScan AI</h1>", unsafe_allow_html=True)

# === Upload Section ===
st.subheader("📤 Upload an ECG image:")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# === Prediction Section ===
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded ECG Image", width=350)

    resized = image.resize((224, 224))
    img_array = np.array(resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    with st.spinner("🔎 Analyzing ECG image..."):
        predictions = model.predict(img_input)[0]
        time.sleep(1)

    predicted_idx = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = round(predictions[predicted_idx] * 100, 2)
    all_confidences = [round(p * 100, 2) for p in predictions]

    # === Result Box ===
    st.markdown(f"""
        <div class="result-box">
            <h2>✅ Prediction: <u>{predicted_label}</u></h2>
            <h3>Confidence: {confidence}%</h3>
            <p>{REPORTS[predicted_label]}</p>
        </div>
    """, unsafe_allow_html=True)

    # === Confidence Chart ===
    with st.expander("📊 Show prediction chart"):
        fig, ax = plt.subplots()
        ax.barh(CLASS_NAMES, all_confidences, color="skyblue")
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Model Confidence by Class")
        st.pyplot(fig)

    # === Class-wise Reports ===
    st.subheader("📋 Class-wise Medical Insights")
    for i, label in enumerate(CLASS_NAMES):
        with st.expander(f"🔍 {label}"):
            st.info(f"Confidence: {all_confidences[i]}%")
            st.write(REPORTS[label])
else:
    st.info("📂 Please upload an ECG image to start analysis.")

# === Footer ===
st.markdown("<p style='text-align:center; color:white; text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8);'>🧠 Built by <b>Elsayed El-Sherbiny</b> using Streamlit & TensorFlow</p>", unsafe_allow_html=True)
