import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import os
import sys
import numpy as np
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# --- CONFIGURATION ---
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SETTING ENGLISH -> TAMIL AS DIRECTION ---
SRC_LANG = "eng_Latn"
TGT_LANG = "tam_Taml" 
# --- END ---

# Define the script codes for the transliterator
SOURCE_SCRIPT_CODE = sanscript.DEVANAGARI
TARGET_SCRIPT_CODE = sanscript.TAMIL


input_sentences = [
    "Please check the QR Code and drink coffee"
]

def load_translation_components():
    """Loads the model, tokenizer, and processor components."""
    print(f"--- Loading Model: {MODEL_NAME} on {DEVICE.upper()} ---")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        ip = IndicProcessor(inference=True) 

        print("✅ Components loaded successfully.")
        return model, tokenizer, ip

    except Exception as e:
        print(f"❌ CRITICAL ERROR during setup: {e}")
        sys.exit(1)


def post_process_and_transliterate(text):
    """
    Converts the Devanagari output script (which is the model's preferred output) 
    to clean native Tamil script.
    """
    try:
        # Use Devanagari (hi) -> Tamil (ta) script conversion
        clean_text = transliterate(
            text, 
            SOURCE_SCRIPT_CODE, 
            TARGET_SCRIPT_CODE
        )
        return clean_text
    except Exception as e:
        # Fallback to the raw, corrupted text if transliteration fails
        return text


def translate_batch(model, tokenizer, ip, sentences, src_lang, tgt_lang):
    """Performs the full translation inference pipeline."""
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, 
        truncation=True, 
        padding="longest", 
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, 
            max_length=256, 
            num_beams=5, 
            num_return_sequences=1,
            use_cache=False, 
        )

    translations = tokenizer.batch_decode(
        generated_tokens.detach().cpu().tolist(), 
        skip_special_tokens=True
    )
    return translations

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Load all components
    model, tokenizer, ip = load_translation_components()

    print("\n--- Starting Translation (English -> Tamil) ---")
    
    translated_texts = translate_batch(model, tokenizer, ip, input_sentences, SRC_LANG, TGT_LANG)

    print("\n--- Final Translation Results ---")
    for src, tgt in zip(input_sentences, translated_texts):
        # 1. Apply the final, robust script cleanup
        clean_tamil_text = post_process_and_transliterate(tgt)
        
        print(f"EN: {src}")
        print(f"TA: {clean_tamil_text}")
