import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

from preprocess import preprocess_image
from segment import segment_food
from portion import estimate_portion
from quality_dl import analyze_quality_dl   # Deep Learning module


# ---------------- GLOBAL VARIABLES ----------------
cap = None
running = False


# ---------------- IMAGE ANALYSIS FUNCTION ----------------
def analyze_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    original, hsv = preprocess_image(file_path)
    segmented = segment_food(original)
    portion, status = estimate_portion(segmented)

    # Deep Learning Quality Prediction
    quality, confidence = analyze_quality_dl(original)

    show_frame(original)

    result_label.config(
        text=f"Portion: {round(portion,2)}%\n"
             f"Quality: {quality}\n"
             f"Confidence: {round(confidence*100,2)}%",
        fg="green" if quality == "Good" else "orange"
    )


# ---------------- SHOW FRAME IN GUI ----------------
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((400, 300))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


# ---------------- START CAMERA ----------------
def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    update_frame()


# ---------------- STOP CAMERA ----------------
def stop_camera():
    global running
    running = False
    if cap:
        cap.release()


# ---------------- UPDATE FRAME LOOP ----------------
def update_frame():
    global running

    if running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (500, 500))

            segmented = segment_food(frame)
            portion, status = estimate_portion(segmented)

            # Deep Learning Quality Prediction
            quality, confidence = analyze_quality_dl(frame)

            cv2.putText(frame,
                        f"{quality} ({round(confidence*100,1)}%)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            show_frame(frame)

            result_label.config(
                text=f"Portion: {round(portion,2)}%\n"
                     f"Quality: {quality}\n"
                     f"Confidence: {round(confidence*100,2)}%"
            )

        root.after(10, update_frame)


# ---------------- GUI DESIGN ----------------
root = tk.Tk()
root.title("Smart Food Portion & Quality Analyzer")
root.geometry("750x650")
root.configure(bg="#f4f6f9")

title = tk.Label(root,
                 text="Smart Food Portion & Quality Analyzer",
                 font=("Helvetica", 18, "bold"),
                 bg="#f4f6f9",
                 fg="#2c3e50")
title.pack(pady=15)

# Video Display
video_label = tk.Label(root, bg="black")
video_label.pack(pady=10)

# Buttons Frame
btn_frame = tk.Frame(root, bg="#f4f6f9")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame,
                       text="Upload Image",
                       command=analyze_image,
                       bg="#3498db",
                       fg="white",
                       font=("Helvetica", 12))
upload_btn.grid(row=0, column=0, padx=10)

start_btn = tk.Button(btn_frame,
                      text="Start Camera",
                      command=start_camera,
                      bg="#2ecc71",
                      fg="white",
                      font=("Helvetica", 12))
start_btn.grid(row=0, column=1, padx=10)

stop_btn = tk.Button(btn_frame,
                     text="Stop Camera",
                     command=stop_camera,
                     bg="#e74c3c",
                     fg="white",
                     font=("Helvetica", 12))
stop_btn.grid(row=0, column=2, padx=10)

# Result Label
result_label = tk.Label(root,
                        text="Results will appear here",
                        font=("Helvetica", 14, "bold"),
                        bg="#f4f6f9")
result_label.pack(pady=20)

# Footer
footer = tk.Label(root,
                  text="Hybrid System: DIP (Portion) + CNN (Quality)",
                  bg="#f4f6f9",
                  fg="gray")
footer.pack(side="bottom", pady=10)

root.mainloop()
footer = tk.Label(root,
                  text="Hybrid AI System | DIP + CNN (Model Pending Training)",
                  bg="#f4f6f9",
                  fg="gray")
