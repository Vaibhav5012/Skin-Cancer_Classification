<!-- PROJECT BANNER -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/eef4e5b6-6f58-4bb3-9614-e39d7bd3c6e7" alt="Project Banner" width="50%">
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
<img width="1000" height="212" alt="Workflow Diag" src="https://github.com/user-attachments/assets/c803fde4-61f3-414c-933a-fdeed9c706f9" />


### 🏗️ Parallel Encoder (Swin Transformer + U-Net)  
<img width="1024" height="1024" alt="SYS arch nobg" src="https://github.com/user-attachments/assets/704ebf80-6271-42bb-911c-103877aa3b52" />


---

## 💻 Application Screenshots

### Input View  
<img width="568" height="739" alt="input" src="https://github.com/user-attachments/assets/bec33f01-50ac-44fa-b243-e86f92bffe86" />


### Prediction Output  
<img width="1919" height="1015" alt="App testing" src="https://github.com/user-attachments/assets/bd69d7d7-66a8-4f26-8790-2b68dc45b446" />


### Classification UI  
<img width="1275" height="745" alt="output" src="https://github.com/user-attachments/assets/b1cc3b32-22e6-4e0a-997f-feb44714d553" />

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

