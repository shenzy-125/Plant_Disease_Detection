import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

#  Page config 
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# Styling 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f7faf7; }

    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-block h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #1a3d1a;
        margin-bottom: 0.25rem;
    }
    .title-block p {
        color: #5a7a5a;
        font-size: 0.95rem;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #d4e8d4;
        border-left: 5px solid #2e7d32;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
    }
    .result-card.healthy { border-left-color: #2e7d32; }
    .result-card.disease { border-left-color: #c62828; }

    .disease-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a3d1a;
        margin-bottom: 0.25rem;
    }
    .confidence {
        font-size: 0.85rem;
        color: #5a7a5a;
        margin-bottom: 1rem;
    }

    .info-section h4 {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888;
        margin: 0.75rem 0 0.25rem 0;
    }
    .info-section p {
        font-size: 0.9rem;
        color: #333;
        margin: 0;
    }

    .healthy-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

#  Disease info 
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Pepper Bell – Bacterial Spot",
        "cause": "Caused by the bacterium Xanthomonas campestris pv. vesicatoria.",
        "symptoms": "Small, water-soaked spots on leaves that turn brown with yellow halos. Spots may also appear on fruits.",
        "treatment": "Use copper-based bactericides. Remove infected plant parts. Avoid overhead irrigation and use disease-free seeds."
    },
    "Pepper__bell___healthy": {
        "display": "Pepper Bell – Healthy",
        "cause": None,
        "symptoms": None,
        "treatment": None
    },
    "Potato___Early_blight": {
        "display": "Potato – Early Blight",
        "cause": "Caused by the fungus Alternaria solani.",
        "symptoms": "Dark brown circular spots with concentric rings (target-board appearance) on older leaves. Leaves yellow and drop early.",
        "treatment": "Apply fungicides like chlorothalonil or mancozeb. Practice crop rotation and remove infected debris."
    },
    "Potato___Late_blight": {
        "display": "Potato – Late Blight",
        "cause": "Caused by the water mold Phytophthora infestans.",
        "symptoms": "Water-soaked, pale green lesions that turn dark brown. White mold may appear on the underside of leaves in humid conditions.",
        "treatment": "Apply systemic fungicides promptly. Destroy infected plants. Use resistant varieties and avoid waterlogging."
    },
    "Potato___healthy": {
        "display": "Potato – Healthy",
        "cause": None,
        "symptoms": None,
        "treatment": None
    },
    "Tomato_Bacterial_spot": {
        "display": "Tomato – Bacterial Spot",
        "cause": "Caused by Xanthomonas species bacteria.",
        "symptoms": "Small, dark, water-soaked spots on leaves, stems, and fruits. Spots have yellow halos and may cause defoliation.",
        "treatment": "Use copper sprays combined with mancozeb. Avoid working with wet plants and use certified disease-free seeds."
    },
    "Tomato_Early_blight": {
        "display": "Tomato – Early Blight",
        "cause": "Caused by the fungus Alternaria solani.",
        "symptoms": "Dark spots with concentric rings on lower/older leaves. Yellowing around lesions and premature leaf drop.",
        "treatment": "Apply fungicides (mancozeb, chlorothalonil). Remove affected leaves, mulch around plants, and practice crop rotation."
    },
    "Tomato_Late_blight": {
        "display": "Tomato – Late Blight",
        "cause": "Caused by Phytophthora infestans.",
        "symptoms": "Greasy, irregular, grey-green spots on leaves that turn brown rapidly. White sporulation visible in humid conditions.",
        "treatment": "Apply fungicides immediately. Remove and destroy infected plants. Avoid overhead watering."
    },
    "Tomato_Leaf_Mold": {
        "display": "Tomato – Leaf Mold",
        "cause": "Caused by the fungus Passalora fulva (formerly Cladosporium fulvum).",
        "symptoms": "Pale green or yellow spots on upper leaf surfaces with olive-green to brown velvety mold on the underside.",
        "treatment": "Improve air circulation, reduce humidity. Apply fungicides like chlorothalonil. Remove infected leaves."
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Tomato – Septoria Leaf Spot",
        "cause": "Caused by the fungus Septoria lycopersici.",
        "symptoms": "Small, circular spots with dark borders and lighter centers, often with tiny black dots (pycnidia) inside.",
        "treatment": "Remove infected leaves promptly. Apply fungicides (mancozeb, copper). Avoid wetting foliage when watering."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Tomato – Spider Mites (Two-spotted)",
        "cause": "Infestation by Tetranychus urticae (two-spotted spider mite).",
        "symptoms": "Tiny yellow or white stippling on leaves, fine webbing on underside, bronzing and leaf drop in severe cases.",
        "treatment": "Spray with miticides or neem oil. Increase humidity. Introduce predatory mites as biological control."
    },
    "Tomato__Target_Spot": {
        "display": "Tomato – Target Spot",
        "cause": "Caused by the fungus Corynespora cassiicola.",
        "symptoms": "Brown lesions with concentric rings resembling a target on leaves, stems, and fruits. Leads to defoliation.",
        "treatment": "Apply fungicides (azoxystrobin, chlorothalonil). Improve spacing and airflow. Remove plant debris after harvest."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Tomato – Yellow Leaf Curl Virus",
        "cause": "Caused by Tomato Yellow Leaf Curl Virus (TYLCV), transmitted by whiteflies.",
        "symptoms": "Severe leaf curling, yellowing, stunted growth, and reduced fruit set. Leaves cup upward.",
        "treatment": "Control whitefly populations using insecticides or sticky traps. Use resistant varieties. Remove infected plants early."
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Tomato – Mosaic Virus",
        "cause": "Caused by Tomato Mosaic Virus (ToMV), spread by contact and contaminated tools.",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves, leaf distortion, and reduced fruit quality.",
        "treatment": "No cure once infected. Remove and destroy infected plants. Disinfect tools regularly. Use resistant seeds."
    },
    "Tomato_healthy": {
        "display": "Tomato – Healthy",
        "cause": None,
        "symptoms": None,
        "treatment": None
    }
}

CLASS_NAMES = list(DISEASE_INFO.keys())

#  Load model 
@st.cache_resource
def load_trained_model():
    return load_model("best_model.keras")

model = load_trained_model()

#  UI 
st.markdown("""
<div class="title-block">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf image to identify the disease and get treatment guidance.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_array = np.expand_dims(img_resized.astype("float32"), axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        preds = model.predict(img_array)

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    pred_class = CLASS_NAMES[pred_idx]
    info = DISEASE_INFO[pred_class]
    is_healthy = info["cause"] is None

    # Result card
    card_class = "healthy" if is_healthy else "disease"
    st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)

    st.markdown(f'<div class="disease-name">{info["display"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)

    if is_healthy:
        st.markdown('<span class="healthy-badge">✅ No disease detected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)
        st.markdown(f'<h4>Cause</h4><p>{info["cause"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<h4>Symptoms</h4><p>{info["symptoms"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<h4>Treatment</h4><p>{info["treatment"]}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info(" Upload a JPG or PNG image of a plant leaf to get started.")
