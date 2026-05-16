import os
import torch
import faiss
import numpy as np
import pandas as pd
import gradio as gr
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           pipeline)
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# -- Config -------------------------------------------------------------------
MODEL_ID    = 'michaelmedhat20/app-store-sentiment-bert'
BERT_MAXLEN = 128
LABEL_NAMES = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
SENTIMENT_EMOJI = {'Positive': '✅', 'Neutral': '😐', 'Negative': '🚨'}

# -- Load BERT ----------------------------------------------------------------
print('Loading BERT model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bert_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
bert_model.to(device).eval()
print('✅ BERT loaded.')

# -- Load FAISS index and corpus ----------------------------------------------
print('Loading FAISS index...')
faiss_path  = hf_hub_download(repo_id=MODEL_ID, filename='faiss.index')
corpus_path = hf_hub_download(repo_id=MODEL_ID, filename='corpus.csv')
index       = faiss.read_index(faiss_path)
corpus_df   = pd.read_csv(corpus_path)
corpus_texts  = corpus_df['text'].tolist()
corpus_labels = corpus_df['label'].tolist()
print(f'✅ FAISS loaded — {index.ntotal:,} vectors.')

# -- Load SentenceTransformer -------------------------------------------------
print('Loading embedder...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Embedder loaded.')

# -- Load TinyLlama -----------------------------------------------------------
print('Loading TinyLlama...')
explainer = pipeline(
    'text-generation',
    model  = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    device = 0 if torch.cuda.is_available() else -1,
)
print('✅ TinyLlama loaded.')

# -- BERT Predict -------------------------------------------------------------
def bert_predict(text: str):
    enc = bert_tokenizer(text, return_tensors='pt',
                         truncation=True, padding=True,
                         max_length=BERT_MAXLEN).to(device)
    with torch.no_grad():
        logits = bert_model(**enc).logits
    probs     = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    label_idx = int(np.argmax(probs))
    return LABEL_NAMES[label_idx], float(probs[label_idx])

# -- Retrieve Similar ---------------------------------------------------------
def retrieve_similar(query: str, k: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'text'  : corpus_texts[idx],
            'label' : corpus_labels[idx],
            'score' : float(score),
        })
    return results

# -- RAG Explain --------------------------------------------------------------
RAG_PROMPT = """<|system|>
You are an app store review analysis assistant.</s>
<|user|>
Review: {text}
Prediction: {label} ({score:.0%} confidence)

Similar reviews:
{examples}

Explain why this review is classified as {label}.</s>
<|assistant|>"""

def rag_explain(text: str, label: str, score: float) -> str:
    similar  = retrieve_similar(text, k=3)
    ex_block = '\n'.join(
        [f'- [{s["label"].upper()}] {s["text"]}' for s in similar]
    )
    prompt = RAG_PROMPT.format(text=text, label=label,
                                score=score, examples=ex_block)
    output = explainer(
        prompt,
        max_new_tokens     = 300,
        temperature        = 0.3,
        do_sample          = True,
        repetition_penalty = 1.2,
    )
    return output[0]['generated_text'].split('<|assistant|>')[-1].strip()

# -- Full Pipeline ------------------------------------------------------------
def full_pipeline(review_text: str):
    if not review_text.strip():
        return 'Please enter a review.', '', ''

    label, conf = bert_predict(review_text)
    emoji       = SENTIMENT_EMOJI.get(label, '')
    pred_str    = f'{emoji} {label.upper()} ({conf:.1%})'

    similar = retrieve_similar(review_text, k=3)
    sim_str = '\n'.join(
        [f'• [{s["label"].upper()}] {s["text"]}' for s in similar]
    )

    explanation = rag_explain(review_text, label, conf)

    return pred_str, sim_str, explanation

# -- Gradio UI ----------------------------------------------------------------
with gr.Blocks(title='App Store Review Classifier', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# ⭐ App Store Review Sentiment Classifier\n> BERT + RAG + TinyLlama — No API needed')

    with gr.Row():
        review_input = gr.Textbox(label='Enter App Review',
                                   placeholder='Paste a user review here...',
                                   lines=3, scale=4)
        predict_btn  = gr.Button('Analyse', variant='primary', scale=1)

    pred_output = gr.Textbox(label='Prediction', interactive=False)
    sim_output  = gr.Textbox(label='Similar Reviews', lines=4, interactive=False)
    exp_output  = gr.Textbox(label='LLM Explanation', lines=8, interactive=False)

    predict_btn.click(fn=full_pipeline,
                       inputs=[review_input],
                       outputs=[pred_output, sim_output, exp_output])

    gr.Examples(
        examples=[
            ['This app is absolutely amazing! Smooth, fast, and no ads.'],
            ['App keeps crashing every time I open it. Very frustrating.'],
            ['It works fine I guess. Nothing special but does the job.'],
        ],
        inputs=review_input,
    )

demo.launch()
