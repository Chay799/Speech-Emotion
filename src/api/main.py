import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.inference.predict import predict_emotion
import webbrowser
from src.language.language_detector import detect_language   # ⭐ ADD THIS IMPORT

app = FastAPI(title="Indic SER API")

app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

# Allow frontend to connect to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    temp_path = f"temp_uploads/{file.filename}"

    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    # ⭐ NEW STEP 1 — LANGUAGE DETECTION
    language = detect_language(temp_path)

    # ⭐ EXISTING STEP 2 — EMOTION
    result = predict_emotion(temp_path)

    # ⭐ Return BOTH language + emotion
    clean_result = {
        "language": str(language),   # ⭐ NEW FIELD
        "emotion": str(result["emotion"]),
        "confidence": float(result["confidence"]),
        "probabilities": {
            "neutral": float(result["probabilities"]["neutral"]),
            "angry": float(result["probabilities"]["angry"]),
            "sad": float(result["probabilities"]["sad"]),
        }
    }

    return clean_result


@app.on_event("startup")
def open_browser():
    webbrowser.open("http://127.0.0.1:8000/app/")
