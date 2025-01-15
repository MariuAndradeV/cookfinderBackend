from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes especificar los orígenes permitidos como una lista, ej.: ["http://localhost:8100"]
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Carga el modelo YOLO
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

# Configuración de la API de Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key="AIzaSyAl3uj5uNGujqy_W7vCHQC42X-rRLvzb-M")

@app.post("/generate-recipe/")
async def generate_recipe(ingredients: list[str]):
    try:
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "max_output_tokens": 2048,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"Eres un chef profesional dame una receta que incluya estos ingredientes: {', '.join(ingredients)}. "
                        "Estos serán los principales y más importantes. Puedes incluir otros ingredientes también. "
                        "Si algún ingrediente está en inglés, tradúcelo al español. "
                        "El formato que quiero es el siguiente:\n\n"
                        "Primero, el nombre de la receta.\n"
                        "Segundo, lista de los ingredientes (incluyendo los que te indiqué).\n"
                        "Tercero, pasos para hacer la receta.\n\n"
                        "Solo pon lo que te indiqué, no des recomendaciones ni nada más. "
                        "Y de preferencia, dame una receta ecuatoriana."
                    ],
                }
            ]
        )

        response = chat_session.send_message("")
        return {"recipe": response.text}

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
