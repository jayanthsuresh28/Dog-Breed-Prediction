import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
from io import BytesIO

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🐾 Dog Breed Predictor",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
        background-color: #fdf6ec;
    }

    .main {
        background-color: #fdf6ec;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #2c1810;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #7a5c45;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #fff8f0, #ffe8cc);
        border-left: 5px solid #e07b39;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(224,123,57,0.15);
    }

    .breed-name {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #2c1810;
        margin-bottom: 0.3rem;
    }

    .confidence-label {
        font-size: 0.95rem;
        color: #7a5c45;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .top-breeds-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        color: #2c1810;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .stProgress > div > div > div > div {
        background-color: #e07b39 !important;
    }

    .upload-hint {
        text-align: center;
        color: #b07050;
        font-size: 0.9rem;
        margin-top: -1rem;
        margin-bottom: 1.5rem;
    }

    .footer {
        text-align: center;
        color: #b07050;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e8d5c0;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #e07b39 !important;
        border-radius: 12px !important;
        background-color: #fff8f0 !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ImageNet Dog Breed Labels (120 breeds)
# ─────────────────────────────────────────────
DOG_BREEDS = {
    151: "Chihuahua", 152: "Japanese Spaniel", 153: "Maltese Dog",
    154: "Pekinese", 155: "Shih-Tzu", 156: "Blenheim Spaniel",
    157: "Papillon", 158: "Toy Terrier", 159: "Rhodesian Ridgeback",
    160: "Afghan Hound", 161: "Basset", 162: "Beagle",
    163: "Bloodhound", 164: "Bluetick", 165: "Black-and-Tan Coonhound",
    166: "Walker Hound", 167: "English Foxhound", 168: "Redbone",
    169: "Borzoi", 170: "Irish Wolfhound", 171: "Italian Greyhound",
    172: "Whippet", 173: "Ibizan Hound", 174: "Norwegian Elkhound",
    175: "Otterhound", 176: "Saluki", 177: "Scottish Deerhound",
    178: "Weimaraner", 179: "Staffordshire Bull Terrier",
    180: "American Staffordshire Terrier", 181: "Bedlington Terrier",
    182: "Border Terrier", 183: "Kerry Blue Terrier", 184: "Irish Terrier",
    185: "Norfolk Terrier", 186: "Norwich Terrier", 187: "Yorkshire Terrier",
    188: "Wire-Haired Fox Terrier", 189: "Lakeland Terrier", 190: "Sealyham Terrier",
    191: "Airedale", 192: "Cairn", 193: "Australian Terrier",
    194: "Dandie Dinmont", 195: "Boston Bull", 196: "Miniature Schnauzer",
    197: "Giant Schnauzer", 198: "Standard Schnauzer", 199: "Scotch Terrier",
    200: "Tibetan Terrier", 201: "Silky Terrier", 202: "Soft-Coated Wheaten Terrier",
    203: "West Highland White Terrier", 204: "Lhasa",
    205: "Flat-Coated Retriever", 206: "Curly-Coated Retriever",
    207: "Golden Retriever", 208: "Labrador Retriever",
    209: "Chesapeake Bay Retriever", 210: "German Short-Haired Pointer",
    211: "Vizsla", 212: "English Setter", 213: "Irish Setter",
    214: "Gordon Setter", 215: "Brittany Spaniel", 216: "Clumber",
    217: "English Springer", 218: "Welsh Springer Spaniel",
    219: "Cocker Spaniel", 220: "Sussex Spaniel", 221: "Irish Water Spaniel",
    222: "Kuvasz", 223: "Schipperke", 224: "Groenendael",
    225: "Malinois", 226: "Briard", 227: "Kelpie",
    228: "Komondor", 229: "Old English Sheepdog", 230: "Shetland Sheepdog",
    231: "Collie", 232: "Border Collie", 233: "Bouvier des Flandres",
    234: "Rottweiler", 235: "German Shepherd", 236: "Doberman",
    237: "Miniature Pinscher", 238: "Greater Swiss Mountain Dog",
    239: "Bernese Mountain Dog", 240: "Appenzeller",
    241: "EntleBucher", 242: "Boxer", 243: "Bull Mastiff",
    244: "Tibetan Mastiff", 245: "French Bulldog", 246: "Great Dane",
    247: "Saint Bernard", 248: "Eskimo Dog", 249: "Malamute",
    250: "Siberian Husky", 251: "Affenpinscher", 252: "Basenji",
    253: "Pug", 254: "Leonberg", 255: "Newfoundland",
    256: "Great Pyrenees", 257: "Samoyed", 258: "Pomeranian",
    259: "Chow", 260: "Keeshond", 261: "Brabancon Griffon",
    262: "Pembroke", 263: "Cardigan", 264: "Toy Poodle",
    265: "Miniature Poodle", 266: "Standard Poodle",
    267: "Mexican Hairless", 268: "Dingo", 269: "Dhole",
    270: "African Hunting Dog",
}

# ─────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────
def predict_breed(image: Image.Image, model, top_k=5):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]  # shape: (1000,)

    # Filter only dog classes (indices 151–270)
    dog_preds = {idx: float(preds[idx]) for idx in DOG_BREEDS}
    total = sum(dog_preds.values())

    if total < 1e-6:
        # Fallback: normalize across all
        sorted_preds = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)
        results = []
        for idx, conf in sorted_preds[:top_k]:
            name = DOG_BREEDS.get(idx, f"Class {idx}")
            results.append({"breed": name, "confidence": float(conf)})
        return results, False

    # Normalize among dog classes only
    results = sorted(dog_preds.items(), key=lambda x: x[1], reverse=True)[:top_k]
    output = []
    for idx, raw_conf in results:
        normalized = raw_conf / total
        output.append({"breed": DOG_BREEDS[idx], "confidence": normalized})
    return output, True


# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🐾 Dog Breed Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Upload a dog photo and discover the breed instantly using AI</div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading AI model..."):
    model = load_model()

# Upload
uploaded_file = st.file_uploader(
    "Upload a dog image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
st.markdown('<div class="upload-hint">Supports JPG, PNG, WEBP · Best results with clear, centered dog photos</div>', unsafe_allow_html=True)

# Or try sample
st.markdown("**Or try a sample image:**")
sample_cols = st.columns(4)
SAMPLES = {
    "🐕 Golden": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Shyla.jpg/320px-Golden_Retriever_Shyla.jpg",
    "🐩 Poodle": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Full_attention_%288067543690%29.jpg/320px-Full_attention_%288067543690%29.jpg",
    "🐾 Husky": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    "🦮 Labrador": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/320px-Labrador_on_Quantock_%282175262184%29.jpg",
}

sample_choice = None
for col, (label, url) in zip(sample_cols, SAMPLES.items()):
    with col:
        if st.button(label, use_container_width=True):
            sample_choice = url

# Determine image source
image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif sample_choice:
    try:
        resp = requests.get(sample_choice, timeout=8)
        image = Image.open(BytesIO(resp.content))
    except Exception:
        st.warning("Could not load sample image. Please upload your own.")

# Run prediction
if image:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing breed..."):
            predictions, is_dog = predict_breed(image, model)

        if predictions:
            top = predictions[0]
            confidence_pct = top["confidence"] * 100

            st.markdown(f"""
            <div class="prediction-card">
                <div class="breed-name">🐶 {top['breed']}</div>
                <div class="confidence-label">Top Prediction</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Confidence: {confidence_pct:.1f}%**")
            st.progress(min(top["confidence"], 1.0))

            st.markdown('<div class="top-breeds-header">Top 5 Matches</div>', unsafe_allow_html=True)
            for i, pred in enumerate(predictions):
                pct = pred["confidence"] * 100
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                st.markdown(f"{medal} **{pred['breed']}** — {pct:.1f}%")
                st.progress(min(pred["confidence"], 1.0))

        if not is_dog:
            st.warning("⚠️ The model is not confident this is a dog. Results may be inaccurate.")

# ─────────────────────────────────────────────
# About Section
# ─────────────────────────────────────────────
with st.expander("ℹ️ About this app"):
    st.markdown("""
    **Model:** MobileNetV2 (pretrained on ImageNet — includes 120 dog breeds)
    
    **How it works:**
    - Your image is resized to 224×224 pixels
    - Passed through MobileNetV2 neural network
    - Output is filtered and normalized across dog breed classes (ImageNet indices 151–270)
    - Top 5 breed matches are shown with confidence scores
    
    **Tips for best results:**
    - Use a clear, well-lit photo
    - Dog should be centered and clearly visible
    - Avoid heavily filtered or cartoon images
    
    **Breeds supported:** 120 dog breeds from the ImageNet dataset including Golden Retriever, Labrador, Poodle, Husky, German Shepherd, Bulldog, Beagle, and many more.
    """)

st.markdown('<div class="footer">Built with Streamlit + TensorFlow · MobileNetV2 · ImageNet Weights</div>', unsafe_allow_html=True)
