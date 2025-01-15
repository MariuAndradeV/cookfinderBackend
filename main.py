from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Carga el modelo YOLO
model = YOLO("prueba.pt")  # Reemplaza con la ruta real a tu modelo

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Leer la imagen del archivo cargado
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detectar objetos usando el modelo
    results = model(image)
    detections = [{"class": result.names[box.cls[0].item()], "confidence": round(box.conf[0].item(), 2)}
                  for result in results for box in result.boxes]
    
    return JSONResponse(content={"detections": detections})
