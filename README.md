# 🚀 VisionDetect AI

A modern AI-powered real-time object detection system built using **YOLOv8**, **FastAPI**, **React**, and **Tailwind CSS**. VisionDetect AI enables users to detect objects through a live webcam, uploaded images, and videos with high accuracy and low latency.

---

## 📌 Overview

VisionDetect AI combines the power of **Ultralytics YOLOv8** with a modern web interface to provide an interactive object detection experience. The application is designed with performance, scalability, and usability in mind, making it suitable for learning, research, and real-world AI applications.

---

## ✨ Key Features

- 🎥 Real-Time Webcam Object Detection
- 🖼️ Image Object Detection
- 🎬 Video Object Detection
- 🎯 High-Accuracy YOLOv8 Predictions
- 📦 Support for Custom YOLO Models (.pt)
- 📊 Live Detection Dashboard
- ⚡ FPS & Inference Time Monitoring
- 📈 Detection Statistics
- 📥 Export Detection Results to CSV
- 📸 Screenshot Capture
- 🌙 Modern Cyberpunk User Interface
- 📱 Fully Responsive Design

---

## 🛠️ Tech Stack

### Frontend
- React.js
- Vite
- Tailwind CSS
- JavaScript

### Backend
- FastAPI
- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- NumPy

---

## 📂 Project Structure

```
VisionDetect-AI
│
├── backend
│   ├── main.py
│   ├── requirements.txt
│   ├── routes
│   ├── models
│   ├── utils
│   └── uploads
│
├── frontend
│   ├── src
│   ├── components
│   ├── assets
│   └── public
│
├── README.md
└── LICENSE
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/VisionDetect-AI.git
cd VisionDetect-AI
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The backend will run on **http://localhost:8000**

The frontend will run on **http://localhost:5173**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Check API status |
| POST | `/detect-image` | Detect objects in an image |
| POST | `/detect-video` | Detect objects in a video |
| POST | `/detect-frame` | Real-time webcam detection |
| POST | `/upload-model` | Upload a custom YOLOv8 model |
| POST | `/export-csv` | Export detection history |

---

## ⚙️ How It Works

1. Start the FastAPI backend.
2. Launch the React frontend.
3. Open the **Live Detection** page.
4. Allow webcam access or upload an image/video.
5. Frames are processed by YOLOv8.
6. Detected objects are displayed with bounding boxes, labels, and confidence scores.
7. View live statistics and export detection results.

---

## 🎯 Future Enhancements

- Object Tracking (ByteTrack / DeepSORT)
- Face Detection & Recognition
- Vehicle Counting
- Person Counting
- OCR Integration
- Pose Estimation
- WebSocket-Based Streaming
- Cloud Deployment
- Docker Support
- Mobile Responsive PWA

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 💡 Acknowledgements

Special thanks to the open-source community and the teams behind:

- Ultralytics YOLOv8
- FastAPI
- React
- OpenCV
- PyTorch
- Tailwind CSS

---

⭐ If you found this project useful, consider starring the repository and sharing it with others.
