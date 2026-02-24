from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

# Se puede usar la API Key desde el archivo .env o directamente
api_key = os.getenv("ROBOFLOW_API_KEY", "rQqDJdwUscBWYsDmmzcm")

rf = Roboflow(api_key=api_key)
project = rf.workspace("tarjetadeconducir").project("placa-4hsbt")
version = project.version(1)

# Descargar en formato yolov11
dataset = version.download("yolov11")

print(f"Dataset descargado en: {dataset.location}")
