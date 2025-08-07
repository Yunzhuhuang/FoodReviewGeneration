# Import necessary libraries
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import os
import random
from sklearn.model_selection import train_test_split

# --- 1. Data Loading and Preprocessing ---
print("--- Starting Data Loading and Preprocessing ---")

# Placeholder for the dataset file path.
DATASET_PATH = 'Reviews.csv'

# --- Memory Optimization: Limit the number of reviews processed ---
MAX_REVIEWS_TO_PROCESS = 20000 
VALIDATION_SPLIT = 0.1 # 10% of the data for validation

# Check if the dataset file exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset file '{DATASET_PATH}' not found.")
    print("Please download the 'Reviews.csv' file from the Kaggle Amazon Fine Food Reviews dataset and place it in the same directory.")
    exit()

# Load the dataset using pandas
try:
    df = pd.read_csv(DATASET_PATH, nrows=MAX_REVIEWS_TO_PROCESS)
except Exception as e:
    print(f"Error loading the CSV file: {e}")
    exit()

text_data = df['Text'].tolist()

def clean_text(text):
    """
    Cleans the input text by removing HTML tags, special characters, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s\.,?!]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the cleaning function to all review texts and add the special End of Review token
EOR_TOKEN = '<EOR>'
cleaned_texts = [clean_text(text) + ' ' + EOR_TOKEN for text in text_data if clean_text(text)]

# --- 2. Tokenization and Sequence Generation ---
# Define model hyperparameters (moved to an earlier section for proper scope)
EMBEDDING_DIM = 128
LSTM_UNITS = 256
DROPOUT_RATE = 0.2
BATCH_SIZE = 128
EPOCHS = 20

TOKENIZER_TYPE = 'word'
SEQUENCE_LENGTH = 50

# Initialize tokenizer
# The EOR_TOKEN will be automatically added to the vocabulary by the tokenizer
tokenizer = Tokenizer(num_words=None, oov_token="<UNK>")
tokenizer.fit_on_texts(cleaned_texts)

total_words = len(tokenizer.word_index) + 1
word_to_int = tokenizer.word_index
int_to_word = {i: word for word, i in word_to_int.items()}

# Convert all cleaned texts into a single long sequence of integers
int_sequence = []
for text in cleaned_texts:
    int_sequence.extend(tokenizer.texts_to_sequences([text])[0])

# Split the integer sequence into training and validation sets
# We use a simple slice for reproducibility and to keep it straightforward.
val_split_index = int(len(int_sequence) * VALIDATION_SPLIT)
train_sequence = int_sequence[val_split_index:]
val_sequence = int_sequence[:val_split_index]

# --- Memory Optimization: Use a Keras Sequence for batch generation ---
class TextSequenceGenerator(Sequence):
    def __init__(self, data_sequence, sequence_length, batch_size):
        self.data_sequence = data_sequence
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data_sequence) - self.sequence_length)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.data_sequence) - self.sequence_length) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X_batch = []
        y_batch = []

        for i in batch_indices:
            if i + self.sequence_length + 1 <= len(self.data_sequence):
                X_batch.append(self.data_sequence[i : i + self.sequence_length])
                y_batch.append(self.data_sequence[i + self.sequence_length])
        
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        return X_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Initialize the data generators for training and validation
train_generator = TextSequenceGenerator(train_sequence, SEQUENCE_LENGTH, BATCH_SIZE)
val_generator = TextSequenceGenerator(val_sequence, SEQUENCE_LENGTH, BATCH_SIZE)

print(f"Total vocabulary size: {total_words}")
print(f"Training sequences: {len(train_sequence) - SEQUENCE_LENGTH}")
print(f"Validation sequences: {len(val_sequence) - SEQUENCE_LENGTH}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Batches per epoch: {len(train_generator)}")
print("--- Data Preprocessing Complete ---")

# --- 3. LSTM Model Building and Training ---
print("--- Starting Model Building and Training ---")

model = Sequential([
    Embedding(total_words, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    LSTM(LSTM_UNITS, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS),
    Dropout(DROPOUT_RATE),
    Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Define callbacks for monitoring and early stopping
# Now we monitor 'val_loss' to prevent overfitting
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    TensorBoard(log_dir='./logs')
]

# Train the model using both generators
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=val_generator
)

print("--- Model Training Complete ---")

# --- 4. Text Generation ---
print("--- Starting Text Generation ---")

try:
    from tensorflow.keras.models import load_model
    best_model = load_model('best_model.keras')
except Exception as e:
    print(f"Could not load saved model. Using the last trained model instead. Error: {e}")
    best_model = model

def generate_text(seed_text, n_words, diversity=1.0):
    """
    Generates new text using the trained model.
    Generation stops if the EOR token is predicted.
    """
    generated_text = seed_text
    
    for _ in range(n_words):
        cleaned_seed = clean_text(seed_text)
        token_list = tokenizer.texts_to_sequences([cleaned_seed])[0]
        
        if len(token_list) < SEQUENCE_LENGTH:
            padded_token_list = [0] * (SEQUENCE_LENGTH - len(token_list)) + token_list
        elif len(token_list) > SEQUENCE_LENGTH:
            padded_token_list = token_list[-SEQUENCE_LENGTH:]
        else:
            padded_token_list = token_list
        
        token_list_for_prediction = np.array(padded_token_list).reshape(1, SEQUENCE_LENGTH)

        predicted_probs = best_model.predict(token_list_for_prediction, verbose=0)[0]
        predicted_word_idx = sample_with_diversity(predicted_probs, diversity)
        
        output_word = int_to_word.get(predicted_word_idx, '<UNK>')
        
        # Stop generation if EOR token is predicted
        if output_word == EOR_TOKEN:
            break
        
        generated_text += " " + output_word
        seed_text += " " + output_word
        
    return generated_text

def sample_with_diversity(preds, diversity=1.0):
    """
    Helper function to sample a word from the probability distribution with a diversity factor.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate a starting sequence of words to seed the generation
if len(train_sequence) > SEQUENCE_LENGTH:
    start_idx = random.randint(0, len(train_sequence) - SEQUENCE_LENGTH - 1)
    starting_int_tokens = train_sequence[start_idx : start_idx + SEQUENCE_LENGTH]
    starting_text = " ".join([int_to_word.get(i, '<UNK>') for i in starting_int_tokens])
else:
    starting_text = " ".join([int_to_word.get(i, '<UNK>') for i in train_sequence])
    while len(starting_text.split()) < SEQUENCE_LENGTH:
        starting_text += " <UNK>"

print(f"Generating new text based on seed: '{starting_text}'\n")
generated = generate_text(starting_text, n_words=100, diversity=0.8)
print("--- Generated Text ---")
print(generated)
print("\n--- Text Generation Complete ---")
