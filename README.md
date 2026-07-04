<div align="center">

# Reconocimiento de Placa

### License Plate Recognition System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=ai&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EEF?style=for-the-badge&logo=opencv&logoColor=white)
![WebRTC](https://img.shields.io/badge/WebRTC-FF5722?style=for-the-badge&logo=webrtc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## About

An AI-powered system for real-time license plate detection and recognition. Uses YOLO object detection combined with OpenCV for image processing and WebRTC for live video streaming.

## Features

- **Real-time Detection** — Detect and read license plates in real-time
- **YOLO Integration** — State-of-the-art object detection
- **WebRTC Streaming** — Live video feed support
- **Dockerized** — Easy deployment and scaling
- **Multiple Input Sources** — Support for camera feeds and images

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv11 |
| Image Processing | OpenCV |
| Video Streaming | WebRTC |
| Backend | Flask |
| Deployment | Docker |

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/Joker8-h/reconocimiento-placa.git
cd reconocimiento-placa

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Docker

```bash
docker build -t plate-recognition .
docker run -p 5000:5000 plate-recognition
```

## Project Structure

```
reconocimiento-placa/
├── app.py              # Main Flask application
├── app2.py             # Alternative implementation
├── webrtc_app.py       # WebRTC streaming
├── train_licencia.py   # Model training script
├── download_dataset.py # Dataset downloader
├── yolo11s.pt          # YOLO model weights
├── requirements.txt
├── Dockerfile
└── yolo-client/        # Client-side YOLO integration
```

---

<div align="center">

[![View Repository](https://img.shields.io/badge/View-Repository-0d1117?style=for-the-badge&logo=github)](https://github.com/Joker8-h/reconocimiento-placa)

</div>
