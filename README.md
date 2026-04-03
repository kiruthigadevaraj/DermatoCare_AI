# 🩺 DermatoCare AI

DermatoCare AI is an AI-powered skin lesion analysis system developed using Deep Learning and Computer Vision techniques.

This project helps in the early detection of skin cancer and skin lesion abnormalities by performing:

- Multi-class disease classification
- Lesion segmentation / boundary detection
- Confidence score prediction
- Professional web interface using Streamlit

---

## 📌 Project Overview

This system is designed to assist in the screening and analysis of skin lesions using dermoscopic images.

The application allows the user to:

1. Upload a skin lesion image
2. Analyze the lesion using AI
3. View predicted disease type
4. Visualize lesion segmentation
5. Check confidence score

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- PIL
- Streamlit

---

## 📂 Dataset Used

This project uses the HAM10000 dataset  
(Human Against Machine with 10000 training images)

The dataset contains real dermoscopic skin lesion images.

### Detectable Conditions
- Actinic Keratoses
- Basal Cell Carcinoma
- Benign Keratosis
- Dermatofibroma
- Melanoma
- Melanocytic Nevi
- Vascular Lesions

---

## ⚙️ Model Details

- CNN Model → Skin disease classification
- Segmentation Model → Lesion area detection
- OpenCV Thresholding → Boundary highlighting
- Confidence Scoring → Prediction probability

---

## 🚀 How to Run

Install required packages:

pip install streamlit tensorflow opencv-python pillow matplotlib numpy

Run the project:

streamlit run app.py

---

## 📸 Output Features

- Uploaded image preview
- Predicted disease
- Confidence score
- Lesion boundary / dark red segmentation
- Probability distribution graph

---

## 🎯 Future Enhancements

- Improved segmentation accuracy
- U-Net based medical segmentation
- Mobile application support
- Cloud deployment
- Hospital clinical integration

---

## ⚠️ Disclaimer

This system is developed for educational and assistive analysis purposes only.

It should not replace professional medical consultation.

---

