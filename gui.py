import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import time
import threading
import json

from preprocess import preprocess_image
from segment import segment_food
from portion import estimate_portion
from quality_dl import analyze_quality_dl
from report_generator import generate_report

BG_DARK    = "#0a0e14"
BG_PANEL   = "#0f1520"
BG_CARD    = "#141c2b"
BG_CARD2   = "#111827"
ACCENT     = "#00d4ff"
ACCENT2    = "#00ff9d"
ACCENT3    = "#ff6b35"
TEXT_PRI   = "#e8f4fd"
TEXT_SEC   = "#6b8cae"
TEXT_DIM   = "#2d4a6b"
BORDER     = "#1e2d42"
GOOD_CLR   = "#00ff9d"
AVG_CLR    = "#ffd700"
POOR_CLR   = "#ff4757"

FONT_TITLE  = ("Courier New", 22, "bold")
FONT_HEAD   = ("Courier New", 11, "bold")
FONT_MONO   = ("Courier New", 10)
FONT_MONO_S = ("Courier New", 9)
FONT_VAL    = ("Courier New", 20, "bold")
FONT_LABEL  = ("Courier New", 8)

root = tk.Tk()
root.title("NUTRISCAN  //  Food Intelligence System")
root.geometry("1080x720")
root.configure(bg=BG_DARK)
root.resizable(False, False)

bg_canvas = tk.Canvas(root, width=1080, height=720, bg=BG_DARK, highlightthickness=0)
bg_canvas.place(x=0, y=0)

def draw_grid():
    bg_canvas.delete("grid")
    for x in range(0, 1081, 40):
        bg_canvas.create_line(x, 0, x, 720, fill="#0d1825", tags="grid")
    for y in range(0, 721, 40):
        bg_canvas.create_line(0, y, 1080, y, fill="#0d1825", tags="grid")
    for pts in [(0,0,120,0,0,120), (960,0,1080,0,1080,120),
                (0,600,0,720,120,720), (960,720,1080,720,1080,600)]:
        bg_canvas.create_line(pts[0],pts[1],pts[2],pts[3], fill=ACCENT, width=1, tags="grid")
        bg_canvas.create_line(pts[0],pts[1],pts[4],pts[5], fill=ACCENT, width=1, tags="grid")

draw_grid()

header = tk.Frame(root, bg=BG_PANEL, height=64)
header.place(x=0, y=0, width=1080)

tk.Label(header, text="Smart Food Portion & Quality Analyzer", font=FONT_TITLE,
         bg=BG_PANEL, fg=ACCENT).place(relx=0.5, y=14, anchor="n")

status_dot = tk.Canvas(header, width=10, height=10, bg=BG_PANEL, highlightthickness=0)
status_dot.place(x=990, y=26)
status_dot.create_oval(1, 1, 9, 9, fill=ACCENT2, outline="")

tk.Label(header, text="SYSTEM READY", font=FONT_LABEL, bg=BG_PANEL, fg=ACCENT2).place(x=1004, y=26)

tk.Frame(root, bg=ACCENT, height=1).place(x=0, y=63, width=1080)
tk.Frame(root, bg=TEXT_DIM, height=1).place(x=0, y=64, width=1080)

def make_card(parent, x, y, w, h, label=""):
    frame = tk.Frame(parent, bg=BG_CARD, bd=0)
    frame.place(x=x, y=y, width=w, height=h)
    tk.Frame(frame, bg=ACCENT, height=2).pack(fill=tk.X)
    if label:
        hdr = tk.Frame(frame, bg=BG_CARD)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"  {label}", font=FONT_MONO_S,
                 bg=BG_CARD, fg=TEXT_SEC).pack(side=tk.LEFT, pady=4)
    return frame

orig_card = make_card(root, 16, 80, 310, 310, "▸ INPUT  //  ORIGINAL")
orig_inner = tk.Frame(orig_card, bg=BG_CARD)
orig_inner.pack(expand=True, fill=tk.BOTH, padx=8, pady=4)
orig_panel = tk.Label(orig_inner, bg="#080d14", text="NO IMAGE LOADED",
                       font=FONT_MONO_S, fg=TEXT_DIM, relief="flat")
orig_panel.pack(expand=True, fill=tk.BOTH)

seg_card = make_card(root, 340, 80, 310, 310, "▸ OUTPUT  //  SEGMENTED")
seg_inner = tk.Frame(seg_card, bg=BG_CARD)
seg_inner.pack(expand=True, fill=tk.BOTH, padx=8, pady=4)
seg_panel = tk.Label(seg_inner, bg="#080d14", text="AWAITING ANALYSIS",
                      font=FONT_MONO_S, fg=TEXT_DIM, relief="flat")
seg_panel.pack(expand=True, fill=tk.BOTH)

def metric_card(x, y, label, var_ref, unit="", color=ACCENT):
    f = tk.Frame(root, bg=BG_CARD2, bd=0)
    f.place(x=x, y=y, width=208, height=98)
    tk.Frame(f, bg=color, height=2).pack(fill=tk.X)
    tk.Label(f, text=label, font=FONT_LABEL, bg=BG_CARD2, fg=TEXT_SEC).pack(anchor="w", padx=10, pady=(6,0))
    val_lbl = tk.Label(f, textvariable=var_ref, font=FONT_VAL, bg=BG_CARD2, fg=color)
    val_lbl.pack(anchor="w", padx=10)
    tk.Label(f, text=unit, font=FONT_LABEL, bg=BG_CARD2, fg=TEXT_DIM).pack(anchor="w", padx=10)
    return val_lbl

var_portion    = tk.StringVar(value="—")
var_quality    = tk.StringVar(value="—")
var_confidence = tk.StringVar(value="—")

metric_card(16,  400, "PORTION FILL",  var_portion,    "% of plate",    ACCENT)
metric_card(228, 400, "QUALITY GRADE", var_quality,    "classification", ACCENT2)
metric_card(440, 400, "CONFIDENCE",    var_confidence, "% certainty",   ACCENT3)

prob_card = make_card(root, 16, 508, 632, 160, "▸ CLASS PROBABILITY DISTRIBUTION")
prob_inner = tk.Frame(prob_card, bg=BG_CARD)
prob_inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

bar_rows = {}
for cls, color in [("Good", GOOD_CLR), ("Average", AVG_CLR), ("Poor", POOR_CLR)]:
    row = tk.Frame(prob_inner, bg=BG_CARD)
    row.pack(fill=tk.X, pady=5)
    tk.Label(row, text=f"{cls.upper():<8}", font=FONT_MONO_S,
             bg=BG_CARD, fg=color, width=8, anchor="w").pack(side=tk.LEFT)
    track = tk.Frame(row, bg=TEXT_DIM, height=14)
    track.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6,6))
    fill = tk.Frame(track, bg=color, height=14)
    fill.place(x=0, y=0, relheight=1.0, width=0)
    pct_lbl = tk.Label(row, text="  0.00%", font=FONT_MONO_S, bg=BG_CARD, fg=color, width=8)
    pct_lbl.pack(side=tk.LEFT)
    bar_rows[cls] = (track, fill, pct_lbl)

log_card = make_card(root, 668, 80, 396, 590, "▸ ANALYSIS LOG  //  SYSTEM OUTPUT")
log_inner = tk.Frame(log_card, bg=BG_CARD)
log_inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
log_text = tk.Text(log_inner, bg="#080d14", fg=ACCENT2, font=FONT_MONO_S, relief="flat",
                   insertbackground=ACCENT2, wrap=tk.WORD, state=tk.DISABLED, bd=0)
log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

log_text.tag_configure("accent",  foreground=ACCENT)
log_text.tag_configure("good",    foreground=GOOD_CLR)
log_text.tag_configure("avg",     foreground=AVG_CLR)
log_text.tag_configure("poor",    foreground=POOR_CLR)
log_text.tag_configure("dim",     foreground=TEXT_DIM)
log_text.tag_configure("heading", foreground=ACCENT, font=FONT_HEAD)
log_text.tag_configure("val",     foreground=TEXT_PRI)
log_text.tag_configure("err",     foreground=POOR_CLR)

def log(msg, tag=""):
    log_text.configure(state=tk.NORMAL)
    log_text.insert(tk.END, msg + "\n", tag)
    log_text.see(tk.END)
    log_text.configure(state=tk.DISABLED)

def log_clear():
    log_text.configure(state=tk.NORMAL)
    log_text.delete("1.0", tk.END)
    log_text.configure(state=tk.DISABLED)

log("╔══════════════════════════════════════╗", "dim")
log("║    NUTRISCAN FOOD ANALYZER  v2.4.1   ║", "accent")
log("╚══════════════════════════════════════╝", "dim")
log("")
log("[ SYSTEM ]  All modules loaded", "dim")
log("[ SYSTEM ]  Deep learning model: READY", "dim")
log("[ SYSTEM ]  Segmentation engine: READY", "dim")
log("[ SYSTEM ]  Report generator:    READY", "dim")
log("")
log("► Upload an image to begin analysis.", "accent")

def show_evaluation():
    try:
        eval_window = tk.Toplevel(root)
        eval_window.title("Model Evaluation")
        eval_window.geometry("620x680")
        eval_window.configure(bg=BG_DARK)
        eval_window.resizable(False, False)

        hdr = tk.Frame(eval_window, bg=BG_PANEL, height=52)
        hdr.pack(fill=tk.X)
        tk.Frame(eval_window, bg=ACCENT2, height=2).pack(fill=tk.X)
        tk.Label(hdr, text="◈  MODEL EVALUATION REPORT",
                 font=FONT_HEAD, bg=BG_PANEL, fg=ACCENT2).place(x=16, y=16)

        with open("model_metrics.json", "r") as f:
            report = json.load(f)

        accuracy = report["accuracy"]

        acc_card = tk.Frame(eval_window, bg=BG_CARD2)
        acc_card.pack(fill=tk.X, padx=16, pady=(14, 0))
        tk.Frame(acc_card, bg=ACCENT2, height=2).pack(fill=tk.X)
        tk.Label(acc_card, text="  OVERALL ACCURACY",
                 font=FONT_MONO_S, bg=BG_CARD2, fg=TEXT_SEC).pack(anchor="w", pady=(6,0))
        tk.Label(acc_card, text=f"  {round(accuracy*100, 2)}%",
                 font=("Courier New", 26, "bold"), bg=BG_CARD2, fg=ACCENT2).pack(anchor="w")
        tk.Frame(acc_card, bg=BG_DARK, height=8).pack()

        cls_colors = {"Good": GOOD_CLR, "Average": AVG_CLR, "Poor": POOR_CLR}
        for cls in ["Good", "Average", "Poor"]:
            if cls not in report:
                continue
            color = cls_colors[cls]
            card = tk.Frame(eval_window, bg=BG_CARD)
            card.pack(fill=tk.X, padx=16, pady=6)
            tk.Frame(card, bg=color, height=2).pack(fill=tk.X)
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill=tk.X, padx=12, pady=8)
            tk.Label(row, text=cls.upper(), font=("Courier New", 13, "bold"),
                     bg=BG_CARD, fg=color, width=10, anchor="w").pack(side=tk.LEFT)
            for metric, key in [("PRECISION", "precision"), ("RECALL", "recall"), ("F1-SCORE", "f1-score")]:
                val = round(report[cls][key], 3)
                col = tk.Frame(row, bg=BG_CARD)
                col.pack(side=tk.LEFT, expand=True)
                tk.Label(col, text=metric, font=FONT_LABEL, bg=BG_CARD, fg=TEXT_SEC).pack()
                tk.Label(col, text=str(val), font=("Courier New", 14, "bold"),
                         bg=BG_CARD, fg=TEXT_PRI).pack()

        try:
            img = Image.open("confusion_matrix.png").resize((380, 180))
            img_tk_ref = ImageTk.PhotoImage(img)
            img_lbl = tk.Label(eval_window, image=img_tk_ref, bg=BG_DARK)
            img_lbl.image = img_tk_ref
            img_lbl.pack(pady=10)
        except Exception:
            tk.Label(eval_window, text="[ confusion_matrix.png not found ]",
                     font=FONT_MONO_S, bg=BG_DARK, fg=TEXT_DIM).pack(pady=4)

    except Exception as e:
        err_win = tk.Toplevel(root)
        err_win.configure(bg=BG_DARK)
        tk.Label(err_win, text=f"Evaluation Error:\n{e}",
                 font=FONT_MONO_S, bg=BG_DARK, fg=POOR_CLR, padx=20, pady=20).pack()

btn_frame = tk.Frame(root, bg=BG_DARK)
btn_frame.place(x=16, y=674, width=1048)

def btn_hover(e):   upload_btn.configure(bg=ACCENT,  fg=BG_DARK)
def btn_leave(e):   upload_btn.configure(bg=BG_DARK, fg=ACCENT)
def eval_hover(e):  eval_btn.configure(bg=ACCENT2,   fg=BG_DARK)
def eval_leave(e):  eval_btn.configure(bg=BG_DARK,   fg=ACCENT2)

upload_btn = tk.Button(btn_frame, text="⬆  UPLOAD & ANALYZE",
                        font=("Courier New", 12, "bold"),
                        bg=BG_DARK, fg=ACCENT,
                        activebackground=ACCENT, activeforeground=BG_DARK,
                        relief="flat", bd=0, padx=28, pady=8,
                        highlightthickness=1, highlightbackground=ACCENT,
                        cursor="hand2",
                        command=lambda: threading.Thread(target=analyze_image, daemon=True).start())
upload_btn.pack(side=tk.LEFT, expand=True)
upload_btn.bind("<Enter>", btn_hover)
upload_btn.bind("<Leave>", btn_leave)

eval_btn = tk.Button(btn_frame, text="◈  VIEW MODEL EVALUATION",
                      font=("Courier New", 12, "bold"),
                      bg=BG_DARK, fg=ACCENT2,
                      activebackground=ACCENT2, activeforeground=BG_DARK,
                      relief="flat", bd=0, padx=28, pady=8,
                      highlightthickness=1, highlightbackground=ACCENT2,
                      cursor="hand2", command=show_evaluation)
eval_btn.pack(side=tk.LEFT, expand=True)
eval_btn.bind("<Enter>", eval_hover)
eval_btn.bind("<Leave>", eval_leave)

prog_var = tk.DoubleVar(value=0)
prog_lbl = tk.StringVar(value="")

prog_frame = tk.Frame(root, bg=BG_DARK)
prog_frame.place(x=16, y=650, width=1048, height=20)

prog_label = tk.Label(prog_frame, textvariable=prog_lbl, font=FONT_MONO_S, bg=BG_DARK, fg=TEXT_SEC)
prog_label.pack(side=tk.LEFT)

prog_bar = ttk.Progressbar(prog_frame, variable=prog_var, maximum=100, length=800)
style = ttk.Style()
style.theme_use("clam")
style.configure("TProgressbar", troughcolor=BG_CARD, background=ACCENT, thickness=6)
prog_bar.pack(side=tk.RIGHT)

def set_progress(pct, msg=""):
    prog_var.set(pct)
    prog_lbl.set(msg)
    root.update_idletasks()

def animate_bar(cls, target_pct, color):
    track, fill, lbl = bar_rows[cls]
    track.update_idletasks()
    total_w = track.winfo_width()
    target_w = int(total_w * target_pct / 100)
    steps = 30
    for i in range(steps + 1):
        w = int(target_w * i / steps)
        fill.place(x=0, y=0, height=14, width=w)
        lbl.configure(text=f"  {target_pct:5.2f}%")
        time.sleep(0.012)
    root.update_idletasks()

def analyze_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    if not file_path:
        return

    upload_btn.configure(state=tk.DISABLED)
    log_clear()
    log("╔══════════════════════════════════════╗", "dim")
    log("║        INITIATING ANALYSIS           ║", "accent")
    log("╚══════════════════════════════════════╝", "dim")
    log("")
    log(f"[ FILE ]  {file_path.split('/')[-1]}", "dim")
    log("")

    try:
        var_portion.set("…")
        var_quality.set("…")
        var_confidence.set("…")

        set_progress(10, "Preprocessing…")
        log("[ 1/4 ]  Preprocessing image …", "dim")
        original, hsv = preprocess_image(file_path)
        log("         ✓ Histogram equalization complete", "good")
        log("         ✓ HSV color space converted", "good")

        img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((290, 270))
        img_tk  = ImageTk.PhotoImage(img_pil)
        orig_panel.configure(image=img_tk, text="")
        orig_panel.image = img_tk
        root.update_idletasks()

        set_progress(35, "Segmenting food region…")
        log("")
        log("[ 2/4 ]  Running GrabCut segmentation …", "dim")
        segmented = segment_food(original)
        log("         ✓ Foreground mask extracted", "good")

        set_progress(55, "Estimating portion…")
        log("")
        log("[ 3/4 ]  Calculating portion fill …", "dim")
        portion, status = estimate_portion(segmented)
        log(f"         ✓ Portion: {round(portion,2)}%", "good")
        log(f"         ✓ Status : {status}", "good")

        set_progress(80, "Running quality model…")
        log("")
        log("[ 4/4 ]  Executing deep learning inference …", "dim")
        quality, confidence, all_probs = analyze_quality_dl(original)
        log(f"         ✓ Predicted class : {quality}", "good")
        log(f"         ✓ Confidence      : {round(confidence*100,2)}%", "good")

        if len(segmented.shape) == 2:
            mask_binary = segmented
        else:
            mask_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
            _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        red_bg = original.copy()
        red_bg[:] = (0, 0, 255)
        mask_color = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
        overlay = np.where(mask_color == 255, original, red_bg)
        overlay = cv2.addWeighted(overlay, 0.7, original, 0.3, 0)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

        ov_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ov_pil = Image.fromarray(ov_rgb).resize((290, 270))
        ov_tk  = ImageTk.PhotoImage(ov_pil)
        seg_panel.configure(image=ov_tk, text="")
        seg_panel.image = ov_tk
        root.update_idletasks()

        set_progress(92, "Generating PDF report…")
        cv2.imwrite("temp_original.jpg", original)
        cv2.imwrite("temp_segmented.jpg", overlay)
        generate_report("temp_original.jpg", "temp_segmented.jpg",
                         portion, quality, confidence, all_probs)
        log("")
        log("[ RPT ]  PDF report saved.", "dim")

        var_portion.set(f"{round(portion,1)}%")
        var_quality.set(quality)
        var_confidence.set(f"{round(confidence*100,1)}%")

        log("")
        log("─────────────  PROBABILITY BREAKDOWN  ─────────────", "heading")
        for cls, color in [("Good", GOOD_CLR), ("Average", AVG_CLR), ("Poor", POOR_CLR)]:
            pct = round(all_probs.get(cls, 0) * 100, 2)
            log(f"  {cls:<9} {pct:6.2f}%", "val")
            threading.Thread(target=animate_bar, args=(cls, pct, color), daemon=True).start()

        set_progress(100, "Analysis complete.")
        log("")
        log("═══════════════  ANALYSIS COMPLETE  ═══════════════", "heading")
        log("  Report saved · All modules nominal", "dim")

    except Exception as exc:
        log(f"\n[ ERROR ]  {exc}", "err")
        set_progress(0, "Error — see log.")

    finally:
        upload_btn.configure(state=tk.NORMAL)

root.mainloop()