# ⭐ App Store Review Sentiment Classification

An end-to-end NLP pipeline to classify Google Play Store user reviews into **Positive**, **Neutral**, or **Negative** sentiment using BERT, BiLSTM, RAG, and a local LLM — no API key needed.

## 🚀 Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/michaelmedhat20/app-store-sentiment-analysis)

> Try it live → [huggingface.co/spaces/michaelmedhat20/app-store-sentiment-analysis](https://huggingface.co/spaces/michaelmedhat20/app-store-sentiment-analysis)

---

## 📌 Project Overview

| Property | Details |
|---|---|
| **Task** | 3-class Sentiment Classification |
| **Classes** | Negative · Neutral · Positive |
| **Dataset** | Google Play Store Reviews (~37k reviews) |
| **Best Model** | BERT fine-tuned (~82% Macro F1) |
| **Runtime** | Google Colab (GPU recommended) |

---

## 🏗️ Pipeline Architecture

```
Raw Review
    │
    ├── Preprocessing → TF-IDF → ML Classifiers (5 models)
    │
    ├── Token Sequences → Embedding → RNN/LSTM/GRU/BiLSTM/BiGRU
    │
    ├── BERT Tokenizer → Fine-tuned BERT → label + confidence
    │                         │
    └── SentenceTransformer → FAISS Index → top-k similar reviews
                                    │
                              TinyLlama (local LLM)
                                    │
                              RAG Explanation
                                    │
                              Gradio UI
```

---

## 📊 Model Comparison

| Layer | Model | Macro F1 |
|---|---|---|
| ML | Logistic Regression | ~0.72 |
| ML | Linear SVM | ~0.74 |
| DL | BiLSTM | ~0.76 |
| DL | BiGRU | ~0.76 |
| Transformer | **BERT fine-tuned** | **~0.82** |

---

## 🧠 Key Features

- **5 ML models** compared (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting)
- **5 Deep Learning architectures** compared (SimpleRNN, LSTM, GRU, BiLSTM, BiGRU)
- **BERT fine-tuned** for 3-class sentiment classification
- **FAISS vector store** for fast similarity search across 37k reviews
- **TinyLlama** local LLM for RAG-augmented explanations — no API key needed
- **Negation handling** via data augmentation ("I don't hate it" → Positive)
- **Gradio UI** deployed on Hugging Face Spaces

---

## 📁 Project Structure

```
app-store-sentiment-analysis/
├── App_Store_Review_Classification.ipynb   ← Full training pipeline
├── README.md
├── app.py                                  ← Gradio app (loads from HF Hub)
├── googleplaystore_user_reviews.csv        ← Dataset
└── requirements.txt                        ← Dependencies
```

---

## ⚙️ How to Run

### Option 1 — Live Demo (No setup needed)
Click the live demo link above ☝️

### Option 2 — Run locally

```bash
git clone https://github.com/michaelmedhat20/app-store-sentiment-analysis
cd app-store-sentiment-analysis
pip install -r requirements.txt
python app.py
```

### Option 3 — Run the full training notebook
1. Open `App_Store_Review_Classification.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Upload `googleplaystore_user_reviews.csv`
4. Run all cells

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | [Google Play Store Apps — Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps) |
| **File** | `googleplaystore_user_reviews.csv` |
| **Total rows** | ~64,000 reviews |
| **After cleaning** | ~37,400 rows |
| **Classes** | Positive (64%) · Negative (22%) · Neutral (14%) |

---

## 🤗 Model on Hugging Face

Fine-tuned BERT model available at:
[huggingface.co/michaelmedhat20/app-store-sentiment-bert](https://huggingface.co/michaelmedhat20/app-store-sentiment-bert)

---

## 🛠️ Tech Stack

- **Python** — TensorFlow, PyTorch, Scikit-learn
- **NLP** — HuggingFace Transformers, BERT, TinyLlama
- **RAG** — FAISS, Sentence-Transformers
- **Data** — Pandas, NumPy, Matplotlib, Seaborn
- **UI** — Gradio
- **Platform** — Google Colab, Hugging Face Spaces

---

## 👨‍💻 Author

**Michael Medhat**
- LinkedIn: [linkedin.com/in/michael-medhat-74a243306](https://linkedin.com/in/michael-medhat-74a243306)
- GitHub: [github.com/michaelmedhat20](https://github.com/michaelmedhat20)
