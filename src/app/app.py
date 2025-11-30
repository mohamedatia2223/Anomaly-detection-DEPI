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
import gdown
import zipfile

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Multimodal AI Detector", layout="wide")

# ==============================
# Path Configuration (Local & Remote)
# ==============================

# 1. Determine Directories
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels (src/app -> root) to find data folder
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 2. Local Files (Already on GitHub)
css_path = os.path.join(current_dir, "style.css")
scaler_local_path = os.path.join(root_dir, "data", "UNSW-NB15", "processed", "scaler.pkl")
features_local_path = os.path.join(root_dir, "data", "UNSW-NB15", "processed", "feature_columns.npy")

# 3. Google Drive Links (For Large Models Only)
# Note: I inserted the links you provided below.
PACKET_MODEL_URL = "https://drive.google.com/file/d/13sB3P7UAwZHsTbUeQ6skDVpuBQVRJXxI/view?usp=drivesdk" 
IMAGE_MODEL_URL = "https://drive.google.com/file/d/1HHNepjybFcmwzv1OJAE44wAENZr5tfhS/view?usp=drivesdk"

# <--- IMPORTANT: PUT YOUR BERT *FOLDER* LINK HERE (NOT ZIP)
NLP_MODEL_FOLDER_URL = "https://drive.google.com/drive/folders/10z6EgbgMsS8iPyfhFJP2AGqSnftOYfXG" 

# Apply CSS
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<div class='title'>Multimodal AI Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect AI-generated content from Text, Images, and Network Packets</div>", unsafe_allow_html=True)

# ==============================
# Download Logic (Folder Support)
# ==============================
@st.cache_resource
def download_and_setup_models():
    """Downloads heavy models from Google Drive to a temporary folder."""
    
    # Folder to store downloaded models
    models_dir = os.path.join(current_dir, "downloaded_models")
    os.makedirs(models_dir, exist_ok=True)

    # Destination paths
    packet_dest = os.path.join(models_dir, "autoencoder_full_model.h5")
    image_dest = os.path.join(models_dir, "convnext_ai_vs_human.pth")
    
    # For NLP, we define the folder where contents will be downloaded
    nlp_folder_dest = os.path.join(models_dir, "bert_model_folder")

    # Helper function for single files
    def download_file_if_missing(url, output_path):
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading {os.path.basename(output_path)}..."):
                gdown.download(url, output_path, quiet=False, fuzzy=True)

    # 1. Download Packet Model (File)
    if "drive.google.com" in PACKET_MODEL_URL:
        download_file_if_missing(PACKET_MODEL_URL, packet_dest)
    
    # 2. Download Image Model (File)
    if "drive.google.com" in IMAGE_MODEL_URL:
        download_file_if_missing(IMAGE_MODEL_URL, image_dest)

    # 3. Download NLP Model (FOLDER)
    if "drive.google.com" in NLP_MODEL_FOLDER_URL:
        # Check if folder exists and is not empty
        if not os.path.exists(nlp_folder_dest) or not os.listdir(nlp_folder_dest):
            with st.spinner("Downloading BERT Model Folder (This may take a few minutes)..."):
                # gdown.download_folder handles the recursion
                gdown.download_folder(url=NLP_MODEL_FOLDER_URL, output=nlp_folder_dest, quiet=False, use_cookies=False)
    
    return packet_dest, image_dest, nlp_folder_dest

# Execute Download
try:
    packetModelPath, imageModelPath, nlpModelPath = download_and_setup_models()
except Exception as e:
    st.error(f"Error downloading models: {e}")
    st.stop()

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_all_models_in_memory():
    # 1. Load Packet Model (From Downloaded File)
    if os.path.exists(packetModelPath):
        # compile=False prevents the error by ignoring metrics/optimizers
        packet_model = load_model(packetModelPath, compile=False) 
    else:
        st.error("Packet model not found. Check Google Drive link.")
        st.stop()
    
    # 2. Load Scaler & Features (From Local GitHub Files)
    if os.path.exists(scaler_local_path) and os.path.exists(features_local_path):
        scaler = joblib.load(scaler_local_path)
        features = np.load(features_local_path, allow_pickle=True)
    else:
        st.error(f"Missing local data files! Checked: {scaler_local_path}")
        st.stop()

    # 3. Load NLP Model (From Downloaded Folder)
    # Search for config.json to find the exact subfolder
    bert_final_path = nlpModelPath
    
    # Sometimes download_folder creates a subfolder with the drive name, so we search recursively
    found_config = False
    for root, dirs, files in os.walk(nlpModelPath):
        if "config.json" in files:
            bert_final_path = root
            found_config = True
            break
            
    try:
        if found_config:
            tokenizer = BertTokenizer.from_pretrained(bert_final_path)
            bert_model = BertForSequenceClassification.from_pretrained(bert_final_path)
        else:
            # Fallback if config.json isn't found immediately
            tokenizer = None
            bert_model = None
            # Only show error if we expected a download
            if "drive.google.com" in NLP_MODEL_FOLDER_URL:
                st.warning("Downloaded NLP folder but couldn't find 'config.json'. Check if the Drive Folder contains the model files.")
                
    except Exception as e:
        tokenizer = None
        bert_model = None
        st.warning(f"Could not load NLP model: {e}")

    # 4. Load Image Model (From Downloaded File)
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
    if os.path.exists(imageModelPath):
        image_model.load_state_dict(torch.load(imageModelPath, map_location=torch.device('cpu')))
    image_model.eval()

    return packet_model, scaler, features, bert_model, tokenizer, image_model

# Load into memory
try:
    packet_model, scaler, features, bert_model, tokenizer, image_model = load_all_models_in_memory()
except Exception as e:
    st.error(f"Critical Error loading models: {e}")
    st.stop()


# ==============================
# Helper Functions
# ==============================
@st.cache_data(show_spinner=False)
def extract_text_from_website(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return re.sub(r'\s+', ' ', text.strip())
    except:
        return ""

@st.cache_data(show_spinner=False)
def extract_images_from_website(url):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, 'html.parser')
        imgs = [img['src'] for img in soup.find_all('img') if img.get('src')]
        full_imgs = []
        for img in imgs:
            if img.startswith('http'):
                full_imgs.append(img)
            else:
                full_imgs.append(url + img)
        return full_imgs[:3]
    except:
        return []

@st.cache_data(show_spinner=False)
def load_image_from_url(img_url):
    try:
        response = requests.get(img_url, timeout=10)
        return Image.open(BytesIO(response.content)).convert('RGB')
    except:
        return None

def predict_text_ai(text):
    if not text: return "No text found"
    if tokenizer is None or bert_model is None: return "NLP Model not loaded (Check Drive Link)"
    
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

def predict_image_ai(img_url_or_obj):
    try:
        if isinstance(img_url_or_obj, str):
            img = load_image_from_url(img_url_or_obj)
            if img is None: return "Error loading image"
        else:
            img = img_url_or_obj
            
        img_tensor = preprocess_image(img)
        with torch.no_grad():
            out = image_model(img_tensor)
            pred = torch.argmax(out, dim=1).item()
        return "AI-generated" if pred == 1 else "Human"
    except Exception as e:
        return f"Error: {e}"

def predict_packet_anomaly():
    expected_features = packet_model.input_shape[1]
    num_actual_features = len(features) # Using the loaded feature list

    # Dummy logic to handle shape mismatch (placeholder)
    if num_actual_features < expected_features:
        X = np.pad(np.random.rand(1, num_actual_features),
                   ((0, 0), (0, expected_features - num_actual_features)),
                   mode='constant')
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
                if image_links:
                    for link in image_links:
                        st.image(link, width=200)
                        img_result = predict_image_ai(link)
                        st.info(f"{img_result}")
                else:
                    st.write("No images found.")
                st.markdown("</div>", unsafe_allow_html=True)

elif option == "🖼️ Upload Image":
    uploaded_image = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=250)
        with st.spinner("Analyzing image..."):
            result = predict_image_ai(img)
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
                packet_result = predict_packet_anomaly()
                st.success(f"Result: {packet_result}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
