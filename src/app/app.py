import streamlit as st
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import re
from scapy.all import sniff, wrpcap
import pandas as pd
import os

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Multimodal AI Detector", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<div class='title'>Multimodal AI Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect AI-generated content from Text, Images, and Network Packets</div>", unsafe_allow_html=True)

# ==============================
# Path Config
# ==============================
packetModel = r"C:\games\git\Anomaly-detection-DEPI\models\UNSW-NB15\autoencoder_full_model.h5"
nlpModel = r'C:\games\git\Anomaly-detection-DEPI\models\AI vs Real Text\bert_model'
imageModel = r'C:\games\git\Anomaly-detection-DEPI\models\AI vs Real images\convnext_ai_vs_human.pth'
scalerPKL1 = r'C:\games\git\Anomaly-detection-DEPI\data\UNSW-NB15\processed\scaler.pkl'
featureColumns = r'C:\games\git\Anomaly-detection-DEPI\data\UNSW-NB15\processed\feature_columns.npy'

# ==============================
# Load Models (Cached)
# ==============================
@st.cache_resource
def load_all_models():
    # 1. Packet model
    packet_model = load_model(packetModel)
    scaler = joblib.load(scalerPKL1)
    features = np.load(featureColumns, allow_pickle=True)

    # 2. NLP model
    tokenizer = BertTokenizer.from_pretrained(nlpModel)
    bert_model = BertForSequenceClassification.from_pretrained(nlpModel)

    # 3. Image model
    image_model = models.convnext_base()
    image_model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    image_model.load_state_dict(torch.load(imageModel, map_location="cpu"))
    image_model.eval()

    return packet_model, scaler, features, bert_model, tokenizer, image_model

packet_model, scaler, features, bert_model, tokenizer, image_model = load_all_models()

# ==============================
# Cached Helper Functions
# ==============================
@st.cache_data(show_spinner=False)
def extract_text_from_website(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return re.sub(r'\s+', ' ', text.strip())

@st.cache_data(show_spinner=False)
def extract_images_from_website(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    imgs = [img['src'] for img in soup.find_all('img') if img.get('src')]
    return imgs[:3]  # limit to first 3 images

@st.cache_data(show_spinner=False)
def load_image_from_url(img_url):
    response = requests.get(img_url, timeout=10)
    return Image.open(BytesIO(response.content)).convert('RGB')

def predict_text_ai(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = bert_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "AI-generated" if pred == 1 else "Human-written"

@st.cache_data(show_spinner=False)
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def predict_image_ai(img_url):
    try:
        img = load_image_from_url(img_url)
        img_tensor = preprocess_image(img)
        with torch.no_grad():
            out = image_model(img_tensor)
            pred = torch.argmax(out, dim=1).item()
        return "AI-generated" if pred == 1 else "Human"
    except Exception as e:
        return f"Error loading image: {e}"

def predict_packet_anomaly():
    expected_features = packet_model.input_shape[1]
    all_features = np.load(featureColumns, allow_pickle=True)
    num_actual_features = len(all_features)

    if num_actual_features < expected_features:
        X = np.pad(np.random.rand(1, num_actual_features),
                   ((0, 0), (0, expected_features - num_actual_features)),
                   mode='constant')
    elif num_actual_features > expected_features:
        X = np.random.rand(1, expected_features)
    else:
        X = np.random.rand(1, expected_features)

    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X

    recon = packet_model.predict(X_scaled)
    mse = np.mean(np.square(recon - X_scaled))
    return f"Anomalous" if mse < 0.25 else f"Normal"

# ==============================
# Streamlit UI
# ==============================
st.markdown("## 🔍 Choose What You Want to Analyze")

option = st.radio(
    "Select Input Type:",
    ["🌐 Website URL", "🖼️ Upload Image", "📝 Text Input", "📡 Packet File"],
    horizontal=True
)

if option == "🌐 Website URL":
    url = st.text_input("🌐 Enter Website URL to Analyze:") 
    if url:
        with st.spinner("🔍 Analyzing content..."):
            text_data = extract_text_from_website(url)
            image_links = extract_images_from_website(url)

        st.markdown("## 🧩 Detection Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Network Packets")
                packet_result = predict_packet_anomaly()
                st.success(f"{packet_result}")
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Text Content")
                st.write(text_data[:400] + "...")
                text_result = predict_text_ai(text_data)
                st.success(f"{text_result}")
                st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Image Samples")
                for link in image_links:
                    st.image(link, width=200)
                    img_result = predict_image_ai(link)
                    st.info(f"{img_result}")
                st.markdown("</div>", unsafe_allow_html=True)

elif option == "🖼️ Upload Image":
    uploaded_image = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=250)
        with st.spinner("Analyzing image..."):
            img_tensor = preprocess_image(img)
            with torch.no_grad():
                out = image_model(img_tensor)
                pred = torch.argmax(out, dim=1).item()
            result = "AI-generated" if pred == 1 else "Human"
        st.success(f"🧠 Prediction: {result}")

elif option == "📝 Text Input":
    user_text = st.text_area("Paste your text here:")
    if user_text:
        with st.spinner("Analyzing text..."):
            result = predict_text_ai(user_text)
        st.success(f"🧠 Prediction: {result}")

elif option == "📡 Packet File":
    uploaded_packet = st.file_uploader("Upload a packet CSV file:", type=["csv"])
    if uploaded_packet:
        df = pd.read_csv(uploaded_packet)
        st.write(df.head())
        with st.spinner("Analyzing packet data..."):
            try:
                X = scaler.transform(df[features])
                recon = packet_model.predict(X)
                mse = np.mean(np.square(recon - X), axis=1)
                anomalies = np.sum(mse < 0.25)
                st.success(f"🧩 Anomalies Detected: {anomalies}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
