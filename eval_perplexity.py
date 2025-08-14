
import os, re, json, random
import os
import re
import json
import math
import numpy as np
import pandas as pd
from typing import List

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model


# ---------- Config ----------
DATASET_PATH = "Reviews.csv"
MODEL_PATH = "workable_model.keras"
MAX_REVIEWS_TO_PROCESS = 20000
VALIDATION_SPLIT = 0.1
SEQUENCE_LENGTH = 50
EOR_TOKEN = "<EOR>"
BATCH_SIZE = 128
OOV_TOKEN = "<UNK>"


# ---------- Utils ----------
def clean_text(text: str) -> str:
    """Same cleaning as training: remove HTML, keep letters + [.,?!], lowercase, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s\.,?!]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextSequenceGenerator(Sequence):
    """Yields batches of (X, y) where X is a window of `sequence_length` token IDs and y is the next token ID."""
    def __init__(self, data_sequence: List[int], sequence_length: int, batch_size: int):
        self.data_sequence = data_sequence
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data_sequence) - self.sequence_length)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.data_sequence) - self.sequence_length) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X_batch, y_batch = [], []
        L = len(self.data_sequence)
        S = self.sequence_length
        for i in batch_indices:
            if i + S + 1 <= L:
                X_batch.append(self.data_sequence[i:i+S])
                y_batch.append(self.data_sequence[i+S])
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def main():
    # ---------- Load dataset ----------
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Place it next to this script.")
    print(f"[info] Loading dataset: {DATASET_PATH} (nrows={MAX_REVIEWS_TO_PROCESS})")
    df = pd.read_csv(DATASET_PATH, nrows=MAX_REVIEWS_TO_PROCESS)
    texts = df["Text"].tolist()
    cleaned_texts = [clean_text(t) + " " + EOR_TOKEN for t in texts if clean_text(t)]
    print(f"[info] Cleaned reviews: {len(cleaned_texts)}")

    # ---------- Load model ----------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    print(f"[info] Loading model: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH, compile=True)
    except Exception as e:
        # If compile info wasn't saved, compile with defaults
        print(f"[warn] Could not load compiled model: {e}. Re-loading without compile and compiling with defaults...")
        model = load_model(MODEL_PATH, compile=False)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Discover model vocab size from final Dense layer
    try:
        model_vocab_size = model.output_shape[-1]
    except Exception:
        # Fallback: last layer units
        model_vocab_size = getattr(model.layers[-1], "units", None)
    if not isinstance(model_vocab_size, int):
        raise RuntimeError("Could not determine model's output vocabulary size (last Dense units).")
    print(f"[info] Model output vocab size: {model_vocab_size}")

    # ---------- Rebuild tokenizer capped to model vocab ----------
    # Using same settings as training, but cap num_words to model_vocab_size
    tokenizer = Tokenizer(num_words=model_vocab_size, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(cleaned_texts)
    # Optional: sanity print
    # print(f"[debug] tokenizer.word_index size: {len(tokenizer.word_index)} (capped at {model_vocab_size})")

    # ---------- Build integer token stream ----------
    int_sequence: List[int] = []
    for txt in cleaned_texts:
        seq = tokenizer.texts_to_sequences([txt])[0]
        int_sequence.extend(seq)
    print(f"[info] Integer token stream length: {len(int_sequence)}")
    if len(int_sequence) <= SEQUENCE_LENGTH + 1:
        raise RuntimeError("Token stream too short for given SEQUENCE_LENGTH. Increase data or lower SEQUENCE_LENGTH.")

    # ---------- Validation split (time-based) ----------
    val_split_index = int(len(int_sequence) * VALIDATION_SPLIT)
    train_sequence = int_sequence[val_split_index:]
    val_sequence   = int_sequence[:val_split_index]
    if len(val_sequence) <= SEQUENCE_LENGTH + 1:
        raise RuntimeError("Validation stream too short for the given SEQUENCE_LENGTH.")

    val_gen = TextSequenceGenerator(val_sequence, SEQUENCE_LENGTH, BATCH_SIZE)
    print(f"[info] Validation batches: {len(val_gen)} (batch_size={BATCH_SIZE}, seq_len={SEQUENCE_LENGTH})")

    # ---------- Evaluate ----------
    print("[info] Evaluating on validation set...")
    results = model.evaluate(val_gen, verbose=1, return_dict=True)
    val_loss = float(results.get("loss"))
    val_acc = results.get("accuracy")
    ppl = math.exp(val_loss)

    print("\n=== Evaluation Results ===")
    print(f"Validation loss       : {val_loss:.4f}")
    print(f"Validation perplexity : {ppl:.2f}")
    if val_acc is not None:
        print(f"Validation accuracy   : {float(val_acc):.4f}")
    print("==========================\n")


if __name__ == "__main__":
    # Fix numpy seed for deterministic shuffling of indices (order only)
    np.random.seed(1337)
    main()
