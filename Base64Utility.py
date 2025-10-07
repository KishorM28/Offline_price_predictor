# File: Base64Utility.py
# Purpose: Utility script to convert a local image file into a Base64 string 
#          and write it directly to a file to avoid terminal display/truncation errors.

import base64
import os
import sys

# CRITICAL: This line forces standard output encoding, fixing the Unicode crash in PowerShell/Windows terminal.
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Configuration ---
IMAGE_FILE = 'tomato2.webp' 
OUTPUT_FILE = 'encoded_image_input.txt'

def encode_image_to_base64(file_path):
    """Encodes a file into a Base64 string."""
    try:
        if not os.path.exists(file_path):
             # Try a fallback common name if the default file name is missing
            if file_path == 'produce.jpg' and os.path.exists('tomato2.webp'):
                 file_path = 'tomato2.webp'
            else:
                raise FileNotFoundError(f"Image file not found at '{file_path}'.")

        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return encoded_string

    except Exception as e:
        print(f"ERROR during encoding: {e}")
        sys.exit(1)

def write_base64_to_file(encoded_string, output_file):
    """Writes the encoded string directly to a file using UTF-8 encoding for safety."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(encoded_string)
        print(f"\n✅ Base64 string successfully written to {output_file}")
        print("-" * 50)
        print("INSTRUCTION: This file is now ready for use in the integrated test command.")
        
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("--- Base64 Encoder Utility ---")
    
    if len(sys.argv) > 1:
        IMAGE_FILE = sys.argv[1]
        
    encoded_image = encode_image_to_base64(IMAGE_FILE)

    if encoded_image:
        write_base64_to_file(encoded_image, OUTPUT_FILE)
