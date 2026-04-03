import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="DermatoCare AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
classification_model = load_model("skin_cancer_model.h5")

IMG_SIZE = 128

class_names = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 DermatoCare AI")
st.sidebar.markdown("### Smart Skin Lesion Detection")

menu = st.sidebar.radio(
    "🧭 Navigation",
    ["🏠 Home", "📤 Upload Image", "📊 Model Details", "ℹ About System", "📌 Future Scope"]
)

# ---------------- HOME ----------------
if menu == "🏠 Home":

    st.markdown(
        "<div class='main-title'>AI-Powered Skin Lesion Analysis</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Professional Clinical Decision Support System</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Step 1**\n\nUpload skin lesion image")

    with col2:
        st.info("**Step 2**\n\nAI analyzes lesion pattern")

    with col3:
        st.info("**Step 3**\n\nGet prediction and segmentation")

    st.markdown("---")

    st.subheader("🧬 Detectable Conditions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("• Actinic Keratoses")
        st.write("• Basal Cell Carcinoma")
        st.write("• Benign Keratosis")
        st.write("• Dermatofibroma")

    with col2:
        st.write("• Melanoma")
        st.write("• Melanocytic Nevi")
        st.write("• Vascular Lesions")

# ---------------- UPLOAD IMAGE ----------------
elif menu == "📤 Upload Image":

    st.title("📤 Upload Skin Image")

    uploaded_file = st.file_uploader(
        "Choose a skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        with col1:
            st.image(image, caption="Uploaded Image", width=350)

        img_class = image.resize((IMG_SIZE, IMG_SIZE))
        img_class = img_to_array(img_class) / 255.0
        img_class = np.expand_dims(img_class, axis=0)

        if st.button("🔍 Analyze Image"):

            # -------- CLASSIFICATION --------
            prediction = classification_model.predict(img_class)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100

            # -------- RELIABLE SEGMENTATION --------
            original_np = np.array(image)

            gray = cv2.cvtColor(
                original_np,
                cv2.COLOR_RGB2GRAY
            )

            blur = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            _, thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            kernel = np.ones((3, 3), np.uint8)

            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_OPEN,
                kernel
            )

            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_CLOSE,
                kernel
            )

            overlay = original_np.copy()
            overlay[thresh == 255] = [139, 0, 0]

            final_output = cv2.addWeighted(
                original_np,
                0.7,
                overlay,
                0.5,
                0
            )

            # -------- DISPLAY --------
            with col2:
                st.subheader("🩺 Diagnosis Result")
                st.success(
                    f"Prediction: {class_names[predicted_class]}"
                )

                st.progress(int(confidence))

                st.write(
                    f"Confidence Score: {confidence:.2f}%"
                )

                st.subheader("🎯 Lesion Segmentation")
                st.image(
                    final_output,
                    caption="Detected Lesion Area",
                    width=350
                )

                st.subheader("📈 Probability Distribution")

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(class_names, prediction)
                ax.set_xlabel("Probability")
                st.pyplot(fig)

# ---------------- MODEL DETAILS ----------------
elif menu == "📊 Model Details":

    st.title("📊 Model Details")

    st.write("""
    • CNN based skin lesion classification  
    • HAM10000 real dataset used  
    • 7 class disease prediction  
    • Confidence score output  
    • Otsu threshold based segmentation  
    • Real-time clinical decision support  
    """)

# ---------------- ABOUT SYSTEM ----------------
elif menu == "ℹ About System":

    st.title("ℹ About System")

    st.write("""
    DermatoCare AI is an intelligent medical support system
    for early detection of skin lesions.

    Features:
    • Disease classification  
    • Lesion segmentation  
    • Confidence scoring  
    • Real-time image analysis  
    • Professional medical interface  
    """)

# ---------------- FUTURE SCOPE ----------------
elif menu == "📌 Future Scope":

    st.title("📌 Future Scope")

    st.write("""
    • Better U-Net segmentation  
    • Mobile application support  
    • Hospital integration  
    • Cloud deployment  
    • Higher accuracy model  
    """)