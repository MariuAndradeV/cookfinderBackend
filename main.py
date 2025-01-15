from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel


from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np


from ultralytics import YOLO
import google.generativeai as genai

# Cargar el modelo YOLO
model = YOLO("prueba.pt")  # Reemplaza con la ruta real a tu modelo
genai.configure(api_key="AIzaSyAl3uj5uNGujqy_W7vCHQC42X-rRLvzb-M")

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


# Modelo para la solicitud de ingredientes
class RecipeRequest(BaseModel):
    ingredients: list[str]

@app.post("/generate-recipe/")
async def generate_recipe(request: RecipeRequest):
    try:
        # Convertir la lista de ingredientes a un texto separado por comas
        ingredients = ", ".join(request.ingredients)
        prompt = (
            f"Eres un chef profesional dame una receta que incluya estos ingredientes: {ingredients}, estos serán los principales y más importantes. "
            "Puedes incluir otros ingredientes también. Si algún ingrediente está en inglés, tradúcelo al español. "
            "El formato que quiero es el siguiente:\n\n"
            "Primero, el nombre de la receta.\n"
            "Segundo, lista de los ingredientes (incluyendo los que te indiqué).\n"
            "Tercero, pasos para hacer la receta.\n\n"
            "Solo pon lo que te indiqué, no des recomendaciones ni nada más. De preferencia, dame una receta ecuatoriana, es decir, de Ecuador."
        )

        # Crear la configuración del modelo
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "text/plain",
        }

        # Crear el modelo
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
        )

        # Crear la sesión de chat
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [prompt],
                }
            ]
        )

        # Enviar el mensaje y obtener la respuesta
        response = chat_session.send_message(prompt)

        # Retornar la receta generada
        return {"recipe": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la receta: {str(e)}")
