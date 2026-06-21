# 🧠 FauxSight  
### AI-Generated Image & Video Detection System

## 📌 Overview
FauxSight is a deep learning-based system that detects whether an image or video frame is **real** or **AI-generated** using a **Convolutional Neural Network (CNN)**.
It addresses the growing challenge of deepfakes and synthetic media created by modern generative AI models.

## 🎯 Objective

To build an AI system that automatically classifies media into:
- 🟢 Real Content  
- 🔴 AI-Generated Content  

## 📊 Dataset

- DeepFake Detection Challenge (DFDC)
- Real and AI-manipulated videos  
- Frames extracted and labeled for training  

**Preprocessing:**
- Frame extraction  
- Resizing & normalization  
- Train/validation/test split  


## 🧠 Model

- Custom CNN architecture
- Feature extraction using convolution layers
- Binary classification (Sigmoid output)
  
## ⚙️ Workflow

Upload Media → Frame Extraction → Preprocessing → CNN Model → Prediction (Real / AI-Generated)

## 📈 Evaluation

- Accuracy  
- Precision  
- Recall  
- F1 Score
  
## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Google Colab (GPU)

## 🚀 Features

- Image + video deepfake detection  
- End-to-end ML pipeline  
- Custom CNN model  
- Real-world AI safety application  

