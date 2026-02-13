import cv2
from preprocess import preprocess_image
from segment import segment_food
from portion import estimate_portion
from quality import analyze_quality
from evaluation import calculate_accuracy
import matplotlib.pyplot as plt
from metrics import confusion_matrix
from metrics import precision_recall_f1

# Input image
image_path = "dataset/image1.png"

# Step 1: Preprocessing
original, hsv = preprocess_image(image_path)

# Step 2: Segmentation
segmented = segment_food(original)

# Step 3: Portion Estimation
portion, portion_status = estimate_portion(segmented)

# Step 4: Quality Analysis
quality = analyze_quality(original, segmented)

# Display results
print("Portion Percentage:", round(portion, 2), "%")
print("Portion Status:", portion_status)
print("Food Quality:", quality)

# Show images
cv2.imshow("Original Image", original)
cv2.imshow("Segmented Food", segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()
from evaluation import calculate_accuracy

# Example test values
actual = ["Normal Portion", "Low Portion", "Excess Portion"]
predicted = ["Normal Portion", portion_status, "Excess Portion"]

accuracy = calculate_accuracy(actual, predicted)

print("System Accuracy:", accuracy, "%")
labels = ["Portion %"]
values = [portion]

plt.bar(labels, values)
plt.title("Food Portion Analysis")
plt.ylabel("Percentage")
plt.show()
cv2.imwrite("segmented_output.jpg", segmented)
labels = ["Low Portion", "Normal Portion", "Excess Portion"]

actual = ["Normal Portion", "Low Portion", "Excess Portion"]
predicted = ["Normal Portion", portion_status, "Excess Portion"]

cm = confusion_matrix(actual, predicted, labels)

print("\nConfusion Matrix:")
print(cm)
precision, recall, f1 = precision_recall_f1(cm)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
