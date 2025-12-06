<!-- PROJECT BANNER -->
<p align="center">
  <img src="<img width="1741" height="730" alt="Banner" src="https://github.com/user-attachments/assets/5deffe11-4e7c-4bee-8e53-9e5841f5e330" />" alt="Project Banner" width="80%">

</p>

<h1 align="center">ECLIPSE – Offline Skin Lesion Classification</h1>

<p align="center">
  <b>Fully Offline · Standalone MSI Installer · Parallel Swin Encoder</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows-blue" />
  <img src="https://img.shields.io/badge/Installer-MSI-orange" />
  <img src="https://img.shields.io/badge/Offline-Yes-success" />
  <img src="https://img.shields.io/badge/Framework-.NET%20WPF-blueviolet" />
  <img src="https://img.shields.io/badge/Model-ONNX-yellow" />
</p>

---

## 🚀 Overview

**ECLIPSE** is a **standalone**, **fully offline** skin lesion classification system combining  
**Swin Transformer + U-Net** through a parallel encoder architecture.

The application runs **locally**, requires **no internet**, and is packaged as a  
**Windows MSI installer** for seamless installation on any device.

---

## 🧠 System Architecture

### 🔄 Workflow  
![Workflow](assets/workflow.png)

### 🏗️ Parallel Encoder (Swin Transformer + U-Net)  
![Parallel Encoder](assets/parallel-encoder.png)

---

## 💻 Application Screenshots

### Input View  
![Input View](assets/input-view.png)

### Prediction Output  
![Output View](assets/output-visualisation.png)

### Classification UI  
![Classification UI](assets/classification-ui.png)

---

## ⭐ Features

- 🔌 **100% Offline** — no cloud, no API calls, no data leaves device  
- 📦 **MSI Installer** — install like a standard Windows application  
- 🧠 ONNX model loaded locally for instant inference  
- 🎯 Benign / Malignant classification with confidence %  
- 📊 Optional CSV export  
- 🔒 User images stay **secure & local**  
- 🖥️ Clean and simple WPF UI  

---

## 📥 Installation (MSI)

1. Download **`ECLIPSE_Setup.msi`**  
2. Run the installer  
3. Follow the installation wizard  
4. Launch the app from:  
   **Start Menu → ECLIPSE – Skin Lesion Classifier**

_No dependencies required. No internet needed._

---

## 📘 Usage

1. Open the ECLIPSE app  
2. Click **Browse** → Select dermoscopic image  
3. Click **Predict**  
4. View:  
   - Classification result (Benign/Malignant)  
   - Confidence percentage  
5. Export results (optional)

All computations are handled **on-device** using the embedded ONNX model.

---

## 📂 Project Structure

```text
.
├── installer/
│   └── ECLIPSE_Setup.msi
├── models/
│   └── model_final.onnx
├── src/
│   ├── training/            # Python training pipeline
│   └── ECLIPSE.App/         # Offline WPF application
├── assets/                  # Images, diagrams, screenshots
└── README.md
