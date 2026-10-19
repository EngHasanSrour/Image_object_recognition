# Image Object Recognition 🔍

An automated image object recognition and detection application built with Python and OpenCV, supporting deep learning models like **YOLOX** and **MobileNet-SSD** in ONNX format.

---

## 📌 Features
- **Object Detection:** Detects and classifies various objects within images.
- **Pre-trained Model Support:** Utilizes YOLOX and MobileNet-SSD ONNX models for fast and accurate inference.
- **COCO Class Labels:** Supports 80+ object categories using standard COCO datasets.

---

## 🛠️ Project Structure
```text
Image_object_recognition/
├── Images/                         # Sample input images for testing
├── coco.txt                        # COCO labels/classes file
├── image_object_detection.py       # Main Python script for image detection
├── video_object_detection.py       # Python script for video detection
├── object_detection_yolox_2022nov.onnx  # Trained model weights
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed, along with OpenCV and NumPy:
```bash
pip install opencv-python numpy

### 2. Running the Detection Script
To run object recognition on a sample image:
```bash
python image_object_detection.py

## 👤 Author
- **Hasan Srour** - [GitHub Profile](https://github.com/EngHasanSrour)
