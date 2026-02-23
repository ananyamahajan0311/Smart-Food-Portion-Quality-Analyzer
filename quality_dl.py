import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("food_quality_model.h5")

IMG_SIZE = 224
class_names = ["Average", "Good", "Poor"]

def analyze_quality_dl(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[class_index], confidence