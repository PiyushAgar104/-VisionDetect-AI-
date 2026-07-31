import os
import io
import time
import base64
import tempfile
import numpy as np
import cv2
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import torch

app = FastAPI(title="VisionDetect AI API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Default model initialization
DEFAULT_MODEL_NAME = "yolov8n.pt"
active_model_name = DEFAULT_MODEL_NAME

try:
    model = YOLO(DEFAULT_MODEL_NAME)
    print(f"Loaded default YOLOv8 model: {DEFAULT_MODEL_NAME}")
except Exception as e:
    print(f"Error loading model {DEFAULT_MODEL_NAME}: {e}")
    model = None

class FrameDetectionRequest(BaseModel):
    image: str  # Base64 encoded image string
    confidence: float = 0.25

class DetectionExportItem(BaseModel):
    id: str
    label: str
    confidence: float
    timestamp: str
    box: List[float]

@app.get("/health")
def health_check():
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (Standard Host)"
    
    classes = list(model.names.values()) if model else []
    
    return {
        "status": "online",
        "active_model": active_model_name,
        "device": device_name,
        "gpu_available": gpu_available,
        "class_count": len(classes),
        "classes_sample": classes[:10]
    }

@app.post("/detect-frame")
def detect_frame(payload: FrameDetectionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLOv8 Model not initialized")

    try:
        b64_data = payload.image
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
            
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        h, w, _ = img.shape
        t0 = time.perf_counter()
        
        results = model.predict(source=img, conf=payload.confidence, verbose=False)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i].xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(boxes[i].conf[0])
                cls_id = int(boxes[i].cls[0])
                label = model.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "id": f"det-{i}-{int(time.time()*1000)}",
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(conf * 100, 1),
                    "box": [round(v, 2) for v in box],
                    "box_norm": [
                        round(box[0]/w, 4),
                        round(box[1]/h, 4),
                        round(box[2]/w, 4),
                        round(box[3]/h, 4)
                    ]
                })

        return {
            "success": True,
            "inference_time": inference_time_ms,
            "width": w,
            "height": h,
            "count": len(detections),
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLOv8 Model not initialized")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")

        h, w, _ = img.shape
        t0 = time.perf_counter()
        results = model.predict(source=img, conf=confidence, verbose=False)
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        annotated_img = img.copy()
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i].xyxy[0].tolist()
                conf = float(boxes[i].conf[0])
                cls_id = int(boxes[i].cls[0])
                label = model.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "id": f"img-det-{i}",
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(conf * 100, 1),
                    "box": [round(v, 2) for v in box],
                    "box_norm": [
                        round(box[0]/w, 4),
                        round(box[1]/h, 4),
                        round(box[2]/w, 4),
                        round(box[3]/h, 4)
                    ]
                })

                # Draw bounding box on annotated image
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 245, 0), 2)
                text = f"{label} {round(conf*100,1)}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - 22), (x1 + tw + 8, y1), (255, 245, 0), -1)
                cv2.putText(annotated_img, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 15, 25), 2)

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated_img)
        b64_annotated = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "inference_time": inference_time_ms,
            "width": w,
            "height": h,
            "count": len(detections),
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{b64_annotated}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLOv8 Model not initialized")

    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        frame_samples = []
        sample_step = max(1, total_frames // 25)
        
        frame_idx = 0
        all_detections = []
        total_objects = 0
        total_time_ms = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                t0 = time.perf_counter()
                results = model.predict(source=frame, conf=confidence, verbose=False)
                t_ms = (time.perf_counter() - t0) * 1000
                total_time_ms += t_ms

                frame_dets = []
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        conf = float(boxes[i].conf[0])
                        cls_id = int(boxes[i].cls[0])
                        label = model.names.get(cls_id, f"class_{cls_id}")
                        frame_dets.append({"label": label, "confidence": round(conf * 100, 1)})
                        total_objects += 1

                frame_samples.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "count": len(frame_dets),
                    "detections": frame_dets
                })

            frame_idx += 1
            if len(frame_samples) >= 25:
                break

        cap.release()

        avg_inference_time = round(total_time_ms / max(1, len(frame_samples)), 2)

        return {
            "success": True,
            "filename": file.filename,
            "total_frames": total_frames,
            "fps": fps,
            "sampled_frames_count": len(frame_samples),
            "total_objects_detected": total_objects,
            "avg_inference_time": avg_inference_time,
            "frame_samples": frame_samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global model, active_model_name
    if not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only PyTorch YOLO model files (.pt) are supported.")

    save_path = os.path.join(MODELS_DIR, file.filename)
    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        new_model = YOLO(save_path)
        model = new_model
        active_model_name = file.filename

        return {
            "success": True,
            "message": f"Successfully loaded custom model: {file.filename}",
            "active_model": active_model_name,
            "classes_count": len(model.names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load custom model: {str(e)}")

@app.post("/export-csv")
def export_csv(items: List[DetectionExportItem]):
    csv_lines = ["ID,Object,Confidence(%),Timestamp,BoundingBox"]
    for item in items:
        box_str = f"[{';'.join(map(str, item.box))}]"
        csv_lines.append(f'"{item.id}","{item.label}",{item.confidence},"{item.timestamp}","{box_str}"')
    
    csv_content = "\n".join(csv_lines)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visiondetect_export.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
