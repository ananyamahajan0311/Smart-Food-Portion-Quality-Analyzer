import numpy as np
import cv2
import tensorflow as tf

# Load trained model (make sure file exists)
model = tf.keras.models.load_model("food_quality_model.h5")

classes = ["Average", "Good", "Poor"]

def analyze_quality_dl(image):

    # Resize to model input size
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]
    confidence = float(predictions[predicted_index])

    # Create probability dictionary
    class_probabilities = {
        "Average": float(predictions[0]),
        "Good": float(predictions[1]),
        "Poor": float(predictions[2])
    }

    return predicted_class, confidence, class_probabilities