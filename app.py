"""
Dog Breed Prediction — Streamlit App
Model : EfficientNet-B0 (PyTorch Transfer Learning)
Breeds: 10 dog breeds
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dog Breed Prediction",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BREEDS = [
    "Beagle", "Boxer", "Bulldog", "Dachshund", "German_Shepherd",
    "Golden_Retriever", "Labrador_Retriever", "Poodle",
    "Rottweiler", "Yorkshire_Terrier",
]

MODEL_PATH = "model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BREED_INFO = {
    "Beagle":              {"origin": "England",      "size": "Small–Medium", "temperament": "Curious, Friendly, Merry"},
    "Boxer":               {"origin": "Germany",      "size": "Medium–Large", "temperament": "Loyal, Fun-loving, Bright"},
    "Bulldog":             {"origin": "England",      "size": "Medium",       "temperament": "Docile, Willful, Friendly"},
    "Dachshund":           {"origin": "Germany",      "size": "Small",        "temperament": "Stubborn, Devoted, Playful"},
    "German_Shepherd":     {"origin": "Germany",      "size": "Large",        "temperament": "Loyal, Courageous, Confident"},
    "Golden_Retriever":    {"origin": "Scotland",     "size": "Large",        "temperament": "Intelligent, Friendly, Reliable"},
    "Labrador_Retriever":  {"origin": "Canada",       "size": "Large",        "temperament": "Even-tempered, Gentle, Agile"},
    "Poodle":              {"origin": "France/Germany","size": "Varies",      "temperament": "Intelligent, Active, Alert"},
    "Rottweiler":          {"origin": "Germany",      "size": "Large",        "temperament": "Steady, Self-assured, Obedient"},
    "Yorkshire_Terrier":   {"origin": "England",      "size": "Small",        "temperament": "Bold, Confident, Intelligent"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Merriweather:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f5a623 !important;
    font-family: 'Merriweather', serif !important;
}
[data-testid="stSidebar"] hr { border-color: #ffffff22; }

/* ── Main ────────────────────────────────── */
.main { background: #f7f3ee; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(15,52,96,0.25);
}
.app-title {
    font-family: 'Merriweather', serif;
    font-size: 2.4rem;
    color: #f5a623;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.app-subtitle {
    color: #a0aec0;
    font-size: 1rem;
    margin: 0;
}

/* Upload zone */
.upload-zone {
    background: white;
    border: 2.5px dashed #0f3460;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #f5a623; }

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(15,52,96,0.3);
}
.result-breed {
    font-family: 'Merriweather', serif;
    font-size: 2rem;
    color: #f5a623;
    margin-bottom: 0.2rem;
}
.result-conf {
    font-size: 1rem;
    color: #a0aec0;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Breed info badge */
.info-pill {
    display: inline-block;
    background: #0f3460;
    color: #f5a623;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 700;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* Top-k table */
.topk-row {
    display: flex;
    align-items: center;
    background: white;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    gap: 0.8rem;
}
.topk-rank { font-size: 1.4rem; min-width: 32px; }
.topk-name { flex: 1; font-weight: 700; color: #1a1a2e; font-size: 1rem; }
.topk-pct  { font-weight: 800; color: #0f3460; font-size: 1rem; }

/* Model tag */
.model-tag {
    display: inline-block;
    background: #f5a623;
    color: #1a1a2e;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.05em;
}

/* Progress bar override */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0f3460, #f5a623) !important;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #0f3460 !important;
    font-family: 'Merriweather', serif !important;
    font-size: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_from_path(path: str, num_classes: int):
    """Load saved EfficientNet-B0 from model.pth"""
    net = models.efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    ckpt = torch.load(path, map_location=DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(DEVICE)
    net.eval()
    class_names = ckpt.get("class_names", BREEDS)
    return net, class_names


@st.cache_resource
def load_pretrained_fallback(num_classes: int):
    """Fallback: EfficientNet-B0 with ImageNet weights, head replaced (untrained).
    Used when model.pth is not available — for demo purposes."""
    net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    net.to(DEVICE)
    net.eval()
    return net, BREEDS


def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            return load_model_from_path(MODEL_PATH, len(BREEDS)), True
        except Exception as e:
            st.warning(f"⚠️ Could not load model.pth: {e}. Using demo mode.")
    return load_pretrained_fallback(len(BREEDS)), False


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, model: nn.Module, class_names: list, top_k: int = 5):
    img_t = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_t)
        probs  = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)
    results = [
        {"breed": class_names[i.item()], "confidence": p.item()}
        for i, p in zip(top_idxs, top_probs)
    ]
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐾 Dog Breed Prediction")
    st.markdown("---")

    st.markdown("### 🧠 Model Info")
    st.markdown("""
    <span class="model-tag">EfficientNet-B0</span>
    &nbsp; Transfer Learning
    """, unsafe_allow_html=True)
    st.markdown(f"""
    - **Framework:** PyTorch  
    - **Classes:** {len(BREEDS)} dog breeds  
    - **Input size:** 224 × 224  
    - **Device:** `{DEVICE}`
    """)

    st.markdown("---")
    st.markdown("### 🐶 Supported Breeds")
    for b in BREEDS:
        st.markdown(f"• {b.replace('_', ' ')}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Top-K Predictions", min_value=1, max_value=10, value=5)
    show_info = st.checkbox("Show breed info", value=True)
    show_cm   = st.checkbox("Show confusion matrix", value=False)

    st.markdown("---")
    st.caption("College Project · CNN Image Classification")

# ─────────────────────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🐾 Dog Breed Prediction</div>
    <div class="app-subtitle">
        Deep Learning · EfficientNet-B0 · 10 Breeds · PyTorch
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
(model, class_names), model_loaded = get_model()

if model_loaded:
    st.success("✅ Trained model loaded from `model.pth`")
else:
    st.info("ℹ️ `model.pth` not found — running in **demo mode** (random predictions). Train the model first with `python model/train.py`.")

# ─────────────────────────────────────────────────────────────────────────────
# Upload + Prediction
# ─────────────────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📤 Upload Dog Image")
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.caption("Supported: JPG · PNG · WEBP · Best with clear, single-dog photos")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_result:
    if uploaded:
        st.markdown("#### 🔍 Prediction Results")
        with st.spinner("Analyzing..."):
            predictions = predict(image, model, class_names, top_k=top_k)

        top = predictions[0]
        conf_pct = top["confidence"] * 100
        breed_display = top["breed"].replace("_", " ")

        # Top result card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-breed">🐶 {breed_display}</div>
            <div class="result-conf">Top Prediction</div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Confidence", f"{conf_pct:.1f}%")
        st.progress(min(top["confidence"], 1.0))

        # Breed info
        if show_info:
            info = BREED_INFO.get(top["breed"], {})
            if info:
                st.markdown(f"""
                <div style="margin:0.5rem 0 1rem;">
                    <span class="info-pill">🌍 {info['origin']}</span>
                    <span class="info-pill">📏 {info['size']}</span>
                    <span class="info-pill">💬 {info['temperament']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Upload a dog photo to get started.")

# ─────────────────────────────────────────────────────────────────────────────
# Top-K Results
# ─────────────────────────────────────────────────────────────────────────────
if uploaded:
    st.markdown("---")
    st.markdown("#### 📊 Top Predictions")
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

    for i, pred in enumerate(predictions):
        pct  = pred["confidence"] * 100
        name = pred["breed"].replace("_", " ")

        st.markdown(f"""
        <div class="topk-row">
            <span class="topk-rank">{medals[i]}</span>
            <span class="topk-name">{name}</span>
            <span class="topk-pct">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(pred["confidence"], 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
if show_cm:
    st.markdown("---")
    st.markdown("#### 📉 Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", use_container_width=True)
    else:
        st.warning("confusion_matrix.png not found. Run `python ev.py` to generate it.")

# ─────────────────────────────────────────────────────────────────────────────
# Project Structure Reference
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📁 Project Structure"):
    st.code("""
dog_breed_prediction/
├── model/
│   └── train.py          # EfficientNet-B0 training script
├── dataset_split/
│   ├── train/
│   │   ├── Beagle/       # ~80 images per breed
│   │   ├── Boxer/
│   │   └── ...
│   └── val/
│       ├── Beagle/       # ~20 images per breed
│       ├── Boxer/
│       └── ...
├── static/
│   └── sample_dogs/      # optional sample images
├── app.py                # ← You are here (Streamlit)
├── ev.py                 # Evaluation + confusion matrix
├── model.pth             # Trained model weights
└── req.txt               # Dependencies
""", language="")

with st.expander("🚀 How to Train"):
    st.markdown("""
    **1. Prepare dataset**
    ```
    dataset_split/
      train/<breed_name>/  # ~80 images each
      val/<breed_name>/    # ~20 images each
    ```
    Supported breeds: `Beagle`, `Boxer`, `Bulldog`, `Dachshund`, `German_Shepherd`,
    `Golden_Retriever`, `Labrador_Retriever`, `Poodle`, `Rottweiler`, `Yorkshire_Terrier`

    **2. Train the model**
    ```bash
    python model/train.py
    ```
    Saves `model.pth` in the project root.

    **3. Evaluate**
    ```bash
    python ev.py
    ```
    Generates `confusion_matrix.png` and prints classification report.

    **4. Run the app**
    ```bash
    streamlit run app.py
    ```
    """)

st.markdown("""
<div style="text-align:center;color:#a0aec0;font-size:0.8rem;margin-top:2rem;
padding-top:1rem;border-top:1px solid #e2e8f0;">
    Dog Breed Prediction · EfficientNet-B0 · PyTorch · Streamlit
</div>
""", unsafe_allow_html=True)
