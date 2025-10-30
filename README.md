# Multimodal AI Anomaly Detector

### Detect anomalies in **Text**, **Images**, and **Network Packets** using Machine Learning and Deep Learning

---

## Project Overview

The **Multimodal AI Anomaly Detector** is a unified system designed to detect **anomalies across multiple data types** — textual, visual, and network — using advanced AI models.

Originally developed as part of an **anomaly detection project**, this tool extends the concept beyond traditional network security to include **linguistic and visual anomalies**, making it a versatile system capable of identifying **AI-generated or abnormal content** across different modalities.

The app uses:

* **Text Anomaly Detection** — Detects unusual or AI-generated language using a fine-tuned **BERT model**.
* **Image Anomaly Detection** — Identifies AI-generated images using a **ConvNeXt classifier**.
* **Network Anomaly Detection** — Detects malicious or abnormal packet behavior using a **Sparse Autoencoder** trained on the **UNSW-NB15 dataset**.

---

## Key Features

* **Website Analysis:** Extracts and analyzes text and image data from any website to detect anomalies.
* **Text Detector:** Determines whether a piece of text is AI-generated or written by a human.
* **Image Detector:** Identifies whether an image is synthetic (AI-generated) or real.
* **Network Packet Detector:** Detects suspicious or anomalous network activity using reconstruction errors.
* **Unified Anomaly Framework:** Combines three AI domains — NLP, Computer Vision, and Cybersecurity — into a single pipeline.

---

## Project Architecture

```
Multimodal-AI-Anomaly-Detector/
│   .gitignore
│   environment.yml
│   README.md
│   requirements.txt
│   setup.py
│
├───data
│   ├───AI vs Human Text
│   │       ---.txt
│   │
│   ├───AI vs Real images
│   │       --.txt
│   │
│   └───UNSW-NB15
│       │   --.txt
│       │
│       └───processed
│               feature_columns.npy
│               labels.npy
│               scaler.pkl
│               threshold.npy
│
├───docs
│   ├───AI vs Real images
│   │       AI_vs_Human_Image_Detection_Report.docx
│   │
│   ├───AI vs Real Text
│   │       Human_vs_AI_Text_Detection_Report.docx
│   │
│   └───UNSW-NB15
│           Autoencoder_UNSW_Report.docx
│
├───notebooks
│   ├───AI vs Human Text
│   │       human_vs_ai_text_detection.ipynb
│   │
│   ├───AI vs Real images
│   │       convnext-classifier.ipynb
│   │
│   └───UNSW-NB15
│           01_exploration.ipynb
│           02_preprocessing.ipynb
│           03_training.ipynb
│           04_evaluation.ipynb
│
├───results
│   ├───AI vs Human Text
│   │       Screenshot 2025-10-23 193442.png
│   │
│   └───UNSW-NB15
│           Figure_1.png
│           Figure_2.png
│           Figure_3.png
│
└───src
    ├───app
    │       app.py
    │       style.css
    │
    └───models
            convnext_classifier.py
            human_vs_ai_text_detection.py
            sparse_autoencoder.py

```
## Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mohamedatia2223/Anomaly-detection-DEPI
cd Multimodal-AI-Anomaly-Detector
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\\Scripts\\activate    # On Windows
source venv/bin/activate # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Trained Models

Place your trained models in the following directories:

```
models/
 ├── AI vs Real Text/bert_model/
 ├── AI vs Real Images/convnext_ai_vs_human.pth
 └── UNSW-NB15/autoencoder_full_model.h5
```

---

## ▶️ Run the Application

```bash
streamlit run src/app/app.py
```

Once launched, the interface allows you to:

* 🌐 Enter a website URL for text, image and packets anomaly detection
* 📝 Paste text directly into the input box
* 🖼️ Upload an image file
* 📡 Upload a packet CSV file

---

## 🧠 Models Used

| Modality            | Model              | Framework              | Task                        |
| ------------------- | ------------------ | ---------------------- | --------------------------- |
| **Network Packets** | Sparse Autoencoder | TensorFlow/Keras       | Detect network anomalies    |
| **Text**            | Bert + OneClassSVM | PyTorch                | Detect linguistic anomalies |
| **Image**           | ConvNeXt           | PyTorch                | Detect visual anomalies     |
ConvNeXt
---

## 🎨 User Interface

The web app uses **Streamlit** with a custom **dark-themed UI** styled via `style.css`, including:

* Glowing gradient buttons
* Rounded analysis cards
* Clean layout for multimodal analysis

---

## 🧾 Example Output

* **Text:** “AI-generated” if the text shows synthetic patterns
* **Image:** “Human” or “AI-generated” based on classification
* **Packets:** “Anomalous” if reconstruction MSE < 0.25

---
