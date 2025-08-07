import tensorflow as tf
from tensorflow.keras.models import load_model

# The path to your saved model file
model_path = 'model.keras'

# Load the model
try:
    loaded_model = load_model(model_path)
    print("Model loaded successfully!")
    loaded_model.summary()
except Exception as e:
    print(f"Error loading the model: {e}")

# You can now use the loaded_model object for predictions
# For example, to generate text:
# ... your text generation code here ...