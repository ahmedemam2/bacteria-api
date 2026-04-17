from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import cv2
import numpy as np
import os

load_dotenv()

app = FastAPI(title="MUST Lab - Bacteria Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def classify_gram(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Bild konnte nicht gelesen werden oder Format wird nicht unterstützt")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # GRAM POSITIVE (purple / violet)
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    gram_positive_mask = purple_mask + blue_mask

    # GRAM NEGATIVE (pink / red)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    lower_pink = np.array([140, 30, 50])
    upper_pink = np.array([170, 200, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    gram_negative_mask = red_mask + pink_mask

    total_pixels = hsv.shape[0] * hsv.shape[1]
    pos_ratio = np.sum(gram_positive_mask > 0) / total_pixels
    neg_ratio = np.sum(gram_negative_mask > 0) / total_pixels

    if pos_ratio > neg_ratio:
        gram = "positive"
        confidence = pos_ratio / (pos_ratio + neg_ratio + 1e-8)
    else:
        gram = "negative"
        confidence = neg_ratio / (pos_ratio + neg_ratio + 1e-8)

    return {
        "gram": gram,
        "confidence": float(confidence),
        "positive_ratio": float(pos_ratio),
        "negative_ratio": float(neg_ratio)
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Nur Bild-Dateien erlaubt")

    image_data = await file.read()

    try:
        result = classify_gram(image_data)
        
        return {
            "class": f"Gram-{result['gram'].capitalize()}",
            "gram": result["gram"],
            "positive_ratio": result["positive_ratio"],
            "negative_ratio": result["negative_ratio"],
            "confidence": result["confidence"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildverarbeitung: {str(e)}")


@app.get("/")
def root():
    return {"status": "ok", "message": "MUST Lab API läuft (Color-based Gram Detection)"}