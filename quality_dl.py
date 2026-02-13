import os
import random

MODEL_PATH = "food_quality_model.h5"

def analyze_quality_dl(image):
    if os.path.exists(MODEL_PATH):
        # Real model will be loaded later
        return "Model Loaded (Ready)", 1.0
    else:
        # Temporary simulated prediction
        classes = ["Good", "Average", "Poor"]
        quality = random.choice(classes)
        confidence = round(random.uniform(0.80, 0.99), 2)
        return quality + " (Simulated)", confidence
