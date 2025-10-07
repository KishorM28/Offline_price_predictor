# File: setup_mobilenet_offline.py
# Purpose: Prepare offline MobileNetV3-small model folder for Hugging Face usage

import os
import json
import torch
from torchvision.models import mobilenet_v3_small

MODEL_DIR = "models/mobilenet_v3-small"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1️⃣ Save config.json
config = {
    "architectures": ["MobileNetV3ForImageClassification"],
    "model_type": "mobilenet_v3",
    "num_labels": 3
}

with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# 2️⃣ Save preprocessor_config.json
preprocessor_config = {
    "feature_extractor_type": "ImageFeatureExtractor",
    "do_resize": True,
    "size": 224,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225]
}

with open(os.path.join(MODEL_DIR, "preprocessor_config.json"), "w") as f:
    json.dump(preprocessor_config, f, indent=2)

# 3️⃣ Download and save PyTorch weights
print("Downloading MobileNetV3-small pretrained weights...")
model = mobilenet_v3_small(pretrained=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pytorch_model.bin"))
print(f"Offline MobileNetV3 model ready at '{MODEL_DIR}'")
