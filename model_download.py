import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor # Required for preprocessing

# 1. Choose your model variant. 
# We use the Distilled English-to-Indic model for faster inference.
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

# 2. Set the device (uses GPU if available, falls back to CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on device: {DEVICE}")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load Model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)

# Initialize the Preprocessor (Handles script conversion and formatting)
ip = IndicProcessor(inference=True)

print("Model and Tokenizer loaded successfully!")