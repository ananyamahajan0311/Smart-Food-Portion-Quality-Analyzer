import cv2
from preprocess import preprocess_image
from segment import segment_food
from portion import estimate_portion
from quality import analyze_quality

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = cv2.resize(frame, (500, 500))
    segmented = segment_food(original)
    portion, status = estimate_portion(segmented)
    quality = analyze_quality(original, segmented)

    cv2.putText(original, f"{status}, {quality}", 
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2)

    cv2.imshow("Smart Food Analyzer", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
