import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import joblib
import threading
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix
)

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================
RGB_TEST_DIR = r"C:\projects\cnn rf journal paddy\Dataset\rgb_dieases\test_images"
MODEL_PATH_DISEASE = "disease_classifier.joblib"
LE_PATH = "label_encoder.joblib"

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading model...")
disease_classifier = joblib.load(MODEL_PATH_DISEASE)
label_encoder = joblib.load(LE_PATH)

# =====================================================
# LOAD CNN FEATURE EXTRACTOR
# =====================================================
print("Loading ResNet50...")
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_feat_from_path(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = preprocess_input(np.expand_dims(img, axis=0))
    return feature_extractor.predict(x, verbose=0).flatten()

# =====================================================
# PRECOMPUTE TEST FEATURES (FAST CM)
# =====================================================
print("Precomputing test features...")
X_cached = []
y_cached = []

for cls in label_encoder.classes_:
    cls_dir = os.path.join(RGB_TEST_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue

    for f in os.listdir(cls_dir)[:20]:  # limit for speed
        img_path = os.path.join(cls_dir, f)
        feat = extract_feat_from_path(img_path)
        if feat is not None:
            X_cached.append(feat)
            y_cached.append(cls)

X_cached = np.array(X_cached)
y_cached = label_encoder.transform(y_cached)

print("Feature caching done!")

# =====================================================
# STRESS FUNCTIONS
# =====================================================
def rgb_to_thermal(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def predict_stress(image_path):
    thermal = rgb_to_thermal(image_path)
    if thermal is None:
        return 0.0
    gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)

def stress_category(val):
    if val < 0.25:
        return "Non-Stress"
    elif val < 0.5:
        return "Mild Stress"
    elif val < 0.75:
        return "Moderate Stress"
    else:
        return "Severe Stress"

# =====================================================
# GUI SETUP
# =====================================================
root = tk.Tk()
root.title("Paddy Disease + Stress Detection (CNN–RF)")
root.geometry("950x700")

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, font=("Arial", 14), justify="left")
result_label.pack(pady=10)

# =====================================================
# IMAGE UPLOAD
# =====================================================
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path).resize((500, 400))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    feat = extract_feat_from_path(file_path)
    if feat is None:
        result_label.config(text="Invalid image")
        return

    pred = disease_classifier.predict(feat.reshape(1, -1))[0]
    disease = label_encoder.inverse_transform([pred])[0]

    stress_val = predict_stress(file_path)
    stress_cat = stress_category(stress_val)

    result_label.config(
        text=f"Disease: {disease}\n"
             f"Stress Value: {stress_val:.2f}\n"
             f"Stress Level: {stress_cat}"
    )

# =====================================================
# CONFUSION MATRIX (SAFE VERSION)
# =====================================================
def plot_confusion_matrix_gui():

    def worker():

        if len(X_cached) == 0:
            print("No test data found.")
            return

        y_pred = disease_classifier.predict(X_cached)

        acc = accuracy_score(y_cached, y_pred)
        prec = precision_score(y_cached, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_cached, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_cached, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(
            y_cached,
            y_pred,
            labels=range(len(label_encoder.classes_))
        )

        cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-8)

        print("\nCLASSIFICATION RESULTS (TP / TN / FP / FN)")
        mcm = multilabel_confusion_matrix(
            y_cached,
            y_pred,
            labels=range(len(label_encoder.classes_))
        )

        for i, cls_name in enumerate(label_encoder.classes_):
            tn, fp, fn, tp = mcm[i].ravel()
            print(f"{cls_name:25s} TP={tp} TN={tn} FP={fp} FN={fn}")

        win = tk.Toplevel(root)
        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(cm_norm,
                    annot=True,
                    fmt=".2f",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    cmap="Blues",
                    ax=ax)

        ax.set_title(f"Confusion Matrix\nAcc={acc:.2f}, "
                     f"Prec={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    threading.Thread(target=worker, daemon=True).start()

# =====================================================
# BUTTONS
# =====================================================
tk.Button(root, text="Upload Image", command=upload_image).pack(pady=5)
tk.Button(root, text="Show Confusion Matrix", command=plot_confusion_matrix_gui).pack(pady=5)

root.mainloop()