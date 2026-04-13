# 🐾 Dog Breed Prediction

A deep learning web app that identifies dog breeds from images.  
Built with CNN Transfer Learning using EfficientNet-B0 + PyTorch, served via Streamlit.

---

## 📌 Features

- Classifies **10 dog breeds** with confidence scores
- Shows **Top-K predictions** (configurable via sidebar slider)
- Displays **breed information** (origin, size, temperament)
- Confusion matrix viewer (run `ev.py` first)
- Simple drag & drop Streamlit interface

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 (Transfer Learning) |
| Framework | PyTorch |
| Dataset | 100 images × 10 breeds (custom) |
| Input Size | 224 × 224 |
| Validation Accuracy | ~97% |

---

## 🐾 Supported Breeds

| # | Breed |
|---|-------|
| 1 | Beagle |
| 2 | Boxer |
| 3 | Bulldog |
| 4 | Dachshund |
| 5 | German Shepherd |
| 6 | Golden Retriever |
| 7 | Labrador Retriever |
| 8 | Poodle |
| 9 | Rottweiler |
| 10 | Yorkshire Terrier |

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Model | PyTorch + EfficientNet-B0 |
| UI | Streamlit |
| Data Augmentation | torchvision.transforms |
| Evaluation | scikit-learn, seaborn |

---

## 📁 Project Structure

```
dog_breed_prediction/
├── model/
│   └── train.py              # EfficientNet-B0 training script
├── dataset_split/
│   ├── train/
│   │   ├── Beagle/           # ~80 images per breed
│   │   ├── Boxer/
│   │   └── ...
│   └── val/
│       ├── Beagle/           # ~20 images per breed
│       ├── Boxer/
│       └── ...
├── static/
│   └── sample_dogs/          # optional sample images
├── app.py                    # Streamlit app (main entry point)
├── ev.py                     # Evaluation + confusion matrix
├── model.pth                 # Trained model weights
└── req.txt                   # Dependencies
```

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/dog-breed-prediction.git
cd dog-breed-prediction
```

### 2. Install dependencies

```bash
pip install -r req.txt
```

### 3. Prepare dataset

Organize your images like this:

```
dataset_split/
  train/
    Beagle/        ← ~80 images
    Boxer/
    ...
  val/
    Beagle/        ← ~20 images
    Boxer/
    ...
```

### 4. Train the model

```bash
python model/train.py
```

This saves `model.pth` in the project root.

### 5. Evaluate (optional)

```bash
python ev.py
```

Generates `confusion_matrix.png` and prints classification report.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open browser at → **http://localhost:8501**

---

## 📊 Model Performance

```
Validation Accuracy: 97%

                    precision  recall  f1-score  support
Beagle               1.00      1.00      1.00       10
Boxer                1.00      1.00      1.00       10
Bulldog              1.00      0.90      0.95       10
Dachshund            0.91      1.00      0.95       10
German_Shepherd      1.00      0.90      0.95       10
Golden_Retriever     1.00      0.90      0.95       10
Labrador_Retriever   1.00      1.00      1.00       10
Poodle               1.00      1.00      1.00       10
Rottweiler           1.00      1.00      1.00       10
Yorkshire_Terrier    0.83      1.00      0.91       10

accuracy                                0.97      100
macro avg            0.97      0.97      0.97      100
weighted avg         0.97      0.97      0.97      100
```

---

## 👨‍💻 Author

College Project — CNN Image Classification with Transfer Learning
