👁️ VisionDetect AI - Real-Time YOLOv8 Object Detection Web Application
PythonFastAPIYOLOv8ReactTailwindCSSVite

VisionDetect AI is a premium, dark futuristic cyberpunk web application for real-time object detection using a webcam feed, image upload, video stream analysis, and custom YOLOv8 model management. Built with a high-performance FastAPI + PyTorch/YOLOv8 backend and a responsive React + Tailwind CSS frontend.

✨ Features
📹 Real-Time Webcam Detection: Low-latency video frame streaming with continuous object detection.
🗣️ Text-to-Speech (TTS) Voice Announcements: Native browser Web Speech API integration that speaks detected object labels aloud (e.g., "Bottle", "Person", "Cell phone").
🎨 Cyberpunk Animated Canvas Overlay: Neon cyan (#00F5FF) and electric purple (#7C3AED) bounding boxes with corner target markers and confidence badges.
⚡ Live Statistics Telemetry: Live FPS meter, inference latency timer (ms), total object counter, and CPU/GPU hardware indicator.
📸 Snapshot & Fullscreen: Capture instant image snapshots with bounding boxes overlaid.
🖼️ Image & Video Detection: Upload static images or video files for frame-by-frame deep learning analysis.
📊 Detection History & CSV Export: Detailed log of all detected objects during a session with downloadable CSV reporting.
⚙️ Custom YOLO Model Hot-Swapping: Upload custom PyTorch .pt model files dynamically via the UI.
📁 Project Structure

vision-detect-ai/
├── backend/
│   ├── main.py              # FastAPI server with YOLOv8 inference API
│   ├── requirements.txt     # Python package dependencies
│   └── models/              # Directory for uploaded custom .pt models
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx        # Glassmorphic top navigation & voice toggle
│   │   │   ├── HeroSection.jsx   # Futuristic landing section
│   │   │   ├── LiveDetection.jsx # Live webcam feed & canvas bounding box renderer
│   │   │   ├── UploadSection.jsx # Drag-and-drop image/video detection
│   │   │   ├── Dashboard.jsx     # Analytics & CSV exporter
│   │   │   └── AboutSection.jsx   # Custom model uploader & system diagnostics
│   │   ├── utils/
│   │   │   └── audio.js          # Web Speech API & Web Audio synth engine
│   │   ├── App.jsx               # Application routing and state management
│   │   ├── main.jsx              # React entry point
│   │   └── index.css             # Cyberpunk theme CSS & glassmorphic styles
│   ├── package.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── vite.config.js
└── README.md
🚀 Quick Start Guide
Prerequisites
Ensure you have the following installed on your machine:

Node.js (v18 or higher) & npm
Python (v3.9 or higher)
1. Backend Setup (FastAPI + YOLOv8)
Navigate to the backend folder and install Python dependencies:

bash

cd backend
pip install -r requirements.txt
Start the FastAPI server:

bash

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
The backend server will run at:

API Base URL: http://localhost:8000
Swagger Documentation: http://localhost:8000/docs
2. Frontend Setup (React + Vite)
In a new terminal window, navigate to the frontend folder and install dependencies:

bash

cd frontend
npm install
Start the Vite development server:

bash

npm run dev
Open your browser and visit:

Web App: http://localhost:5173
📡 API Reference
Endpoint	Method	Description
/health	GET	Hardware status, active model info, supported COCO classes
/detect-frame	POST	Base64 frame processing for live webcam feed
/detect-image	POST	Image file upload detection with annotated base64 response
/detect-video	POST	Video file upload frame-by-frame analysis
/upload-model	POST	Upload custom PyTorch .pt model file and hot-swap active model
/export-csv	POST	Generate downloadable CSV report from session detection logs
🛠️ Built With
Frontend: React, Vite, Tailwind CSS, Lucide React Icons, Web Speech API
Backend: FastAPI, Ultralytics YOLOv8, OpenCV, PyTorch, NumPy, Uvicorn
📝 License
Distributed under the MIT License. See LICENSE for more information
