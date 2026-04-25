import base64
import os
import cv2
import eel
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIGURATION
# =========================
WEB_DIR = 'web'
MODEL_PATH = 'fauxsight.keras'

eel.init(WEB_DIR)

# Load model globally (fast access)
model = keras.models.load_model(MODEL_PATH, compile=False)
print("[Model] Loaded successfully.")


# =========================
# IMAGE DECODER (Base64 → OpenCV)
# =========================
def _load_image_from_base64(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image data")

    return image


# =========================
# IMAGE PREDICTION CORE
# =========================
@eel.expose
def predict_image(img_source):
    if isinstance(img_source, str) and img_source.startswith("data:"):
        img = _load_image_from_base64(img_source)
    else:
        img_path = os.path.abspath(img_source)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = cv2.imread(img_path)

    img = cv2.resize(img, (32, 32)).astype('float32') / 255.0
    prob = float(model.predict(np.expand_dims(img, 0), verbose=0)[0][0])

    label = "AI-Generated" if prob >= 0.5 else "Real"
    confidence = f"{max(prob, 1 - prob) * 100:.1f}%"

    return {"label": label, "confidence": confidence}


# =========================
# VIDEO PREDICTION CORE
# =========================
def predict_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"label": "Error", "confidence": "0%"}

    predictions = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % frame_skip == 0:
            try:
                frame_resized = cv2.resize(frame, (32, 32)).astype('float32') / 255.0
                prob = float(model.predict(np.expand_dims(frame_resized, 0), verbose=0)[0][0])
                predictions.append(prob)
            except:
                pass

        frame_idx += 1

    cap.release()

    if len(predictions) == 0:
        return {"label": "Error", "confidence": "0%"}

    avg_prob = np.mean(predictions)

    label = "AI-Generated" if avg_prob >= 0.5 else "Real"
    confidence = f"{max(avg_prob, 1 - avg_prob) * 100:.1f}%"

    return {"label": label, "confidence": confidence}


# =========================
# 1. INITIALIZATION FUNCTION
# =========================
@eel.expose
def load_system():
    print("Initializing system...")
    dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("System ready!")
    return {"status": "ready"}


# =========================
# 2. UNIFIED DETECTION FUNCTION
# =========================
@eel.expose
def process_file(file_path):
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        return {"error": "File not found"}

    ext = os.path.splitext(file_path)[1].lower()

    # IMAGE
    if ext in [".jpg", ".jpeg", ".png"]:
        result = predict_image(file_path)
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "type": "image"
        }

    # VIDEO
    elif ext in [".mp4", ".avi", ".mov"]:
        result = predict_video(file_path)
        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "type": "video"
        }

    return {"error": "Unsupported file type"}


# =========================
# 3. FILE PICKER FUNCTION
# =========================
@eel.expose
def select_file():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)  # <--- Add this line
    file_path = filedialog.askopenfilename(
        parent=root,                   # <--- Add this too
        title="Select Image or Video",
        filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
    )

    root.destroy()                     # Cleanup the root window
    return file_path


# =========================
# START APPLICATION
# =========================
if __name__ == "__main__":
    eel.start('index.html', size=(1200, 800), host='localhost', port=8000)