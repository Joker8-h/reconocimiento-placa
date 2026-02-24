from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print("Classes in best.pt:", model.names)
else:
    print("best.pt not found")

MODEL_PATH_ROOT = os.path.join(BASE_DIR, "best.pt")
if os.path.exists(MODEL_PATH_ROOT):
    model_root = YOLO(MODEL_PATH_ROOT)
    print("Classes in root best.pt:", model_root.names)
else:
    print("root best.pt not found")
