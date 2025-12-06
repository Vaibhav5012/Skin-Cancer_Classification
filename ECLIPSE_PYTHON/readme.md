## 📌 Overview

Skin cancer is one of the fastest-growing cancers globally, and early detection is crucial.  
**ECLIPSE (Python Edition)** is a fast, fully offline diagnostic support tool built with:

- **Python**
- **CustomTkinter** (Modern desktop UI)
- **Hybrid deep-learning model (ONNX Runtime)**
- **Matplotlib** for visual analytics

The application processes dermoscopic images locally and provides:

- Top predicted class  
- Full probability distribution  
- Donut chart visualization  
- Computation logs  
- Optional Grad-CAM heatmap

---

## ✨ Features

### ✔ Fully Offline Desktop App
- 100% local inference using **ONNX Runtime**  
- No internet needed — complete privacy

### ✔ Real-Time Predictions
- Displays top class + 7-class probability vector  
- Instant model inference

### ✔ Visual Analytics
- Horizontal bar chart (confidence distribution)  
- Donut chart representation  
- Live computation logs

### ✔ Heatmap Mode (Optional)
- Grad-CAM based heatmaps to visualize attention regions

### ✔ Export Options
- Export predictions as **CSV**  
- Save complete logs as **PDF**

### ✔ Modern UI
- Built using **CustomTkinter**  
- Clean teal-themed interface  

---

## 🚀 How It Works

1. User uploads a dermoscopic image  
2. Image is preprocessed  
   - Resize  
   - Normalize  
3. Tensor is fed to the ONNX model  
4. Model outputs a **7-class probability vector**  
5. UI displays:  
   - Predicted class  
   - Confidence distribution  
   - Donut/pie chart  
   - Operation logs  
   - Optional heatmap  

---

## 📸 Application Interface

<p align="center">
  <img src="https://github.com/user-attachments/assets/6bcff941-bc06-406f-b45c-0aa37a3b4c79"
       alt="App Interface"
       style="max-width: 90%; border-radius: 10px;">
</p>

---

## 🛠 Installation Guide (Windows OS)

### 🔹 1. Create Virtual Environment
python -m venv venv

### 🔹 2. Activate Environment
venv\Scripts\activate

### 🔹 3. Install Dependencies
pip install -r requirements.txt


### 🔹 4. Run the Application
python main.py

