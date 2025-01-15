from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from pydantic import BaseModel
import google.generativeai as genai

# Crear la instancia de FastAPI
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo YOLO
model = YOLO("prueba.pt")  # Reemplaza con la ruta real a tu modelo

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # Leer la imagen del archivo cargado
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar objetos usando el modelo
        results = model(image)
        detections = [{"class": result.names[box.cls[0].item()], "confidence": round(box.conf[0].item(), 2)}
                      for result in results for box in result.boxes]
        
        return JSONResponse(content={"detections": detections})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Configuración de la API Gemini
try:
    genai.configure(api_key="AIzaSyAl3uj5uNGujqy_W7vCHQC42X-rRLvzb-M")
except Exception as e:
    raise RuntimeError(f"Error al configurar Gemini: {e}")

# Modelo para recibir los ingredientes
class RecipeRequest(BaseModel):
    ingredients: list[str]

@app.post("/generate-recipe/")
async def generate_recipe(request: RecipeRequest):
    try:
        ingredients = ", ".join(request.ingredients)
        prompt = (
            f"Eres un chef profesional. Dame una receta que incluya estos ingredientes: {ingredients}. "
            "El formato debe incluir el nombre, los ingredientes y los pasos."
        )

        # Enviar el prompt al modelo de Gemini
        response = genai.chat(
            messages=[
                {"role": "system", "content": "Eres un chef profesional."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Verificar si la respuesta contiene texto
        if response and "content" in response["messages"][-1]:
            recipe = response["messages"][-1]["content"]
            return {"recipe": recipe}
        else:
            return JSONResponse(content={"error": "No se obtuvo respuesta del modelo Gemini."}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
