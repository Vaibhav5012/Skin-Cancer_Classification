# 🌟 ECLIPSE — Skin Lesion Classifier

Standalone Windows application for skin-lesion classification using DenseNet169 + custom preprocessing pipeline.
The system analyzes dermoscopic images and predicts the probability distribution across 7 major skin cancer types, along with visual explanations and confidence metrics.

## 📌 Overview

Skin cancer is one of the fastest-growing cancers globally, and early detection is crucial.
ECLIPSE provides a fast, offline, and easy-to-use diagnostic support tool built using:

Python

Tkinter (CustomTkinter UI)

DenseNet169 deep learning model

ONNX for optimized inference

Matplotlib for visual analytics

The app processes an image, performs model inference locally, and displays:

Top predicted class

Confidence distribution

Donut chart visualization

Log of every computation step

Optional heatmap view

## 🧠 Model & Classes
Model Used

DenseNet169

Pretrained on ImageNet

Fine-tuned using dermoscopic image datasets

Input size: 224 × 224 px

Output: 7-class softmax probability vector

Classification Labels
Abbreviation	Full Name
akiec	Actinic Keratoses / Bowen’s Disease
bcc	Basal Cell Carcinoma
bkl	Benign Keratosis-like Lesions
df	Dermatofibroma
mel	Melanoma
nv	Melanocytic Nevi
vasc	Vascular Lesions

These follow standard ISIC dataset categories.

## ✨ Features
✔ Offline Desktop App (No Internet Required)

Fully local inference using ONNX — data remains private.

✔ Real-Time Prediction

Shows a full probability breakdown across 7 classes.

✔ Visual Analytics

Horizontal bar chart (confidence %)

Donut chart of predictions

Live logs of the entire process pipeline

✔ Heatmap Mode

Highlights attention regions if Grad-CAM is enabled (optional).

✔ Export Options

Export predictions as CSV

Save full logs as PDF

✔ Clean, Modern UI

Built using CustomTkinter with teal-themed aesthetics.

## 🚀 How It Works

User uploads a dermoscopic image

Image is preprocessed (resize → normalize)

Tensor is passed to the DenseNet169 ONNX model

Model outputs a 7-dimensional probability vector

UI displays:

Top predicted class

Full distribution

Pie chart

Log details

## 📸 Application Interface


<img width="1362" height="723" alt="image" src="https://github.com/user-attachments/assets/6bcff941-bc06-406f-b45c-0aa37a3b4c79" />


## 🛠 Installation Guide
🔹 1. Create a Virtual Environment
python -m venv venv

🔹 2. Activate the Environment

Windows:

venv\Scripts\activate

🔹 3. Install Requirements
pip install -r requirements.txt

🔹 4. Run the App
python main.py

## 📂 Repository Structure
ECLIPSE_PYTHON/
│── assets/
│── models/
│   ├── densenet169_unet_final.h5
│   ├── final.onnx
│── main.py
│── launcher.py
│── inspect_model.py
│── README.md
│── requirements.txt

## Team

Anagha P Kulkarni

Debabrata Kuiry

B Chiru Vaibhav

## 📄 License

Distributed under the MIT License.
