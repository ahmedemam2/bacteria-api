from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests, base64, re, os

load_dotenv()

app = FastAPI(title="MUST Lab - Bacteria Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("OPENROUTER_API_KEY")

PROMPT = """You are an expert microbiologist specializing in Gram stain analysis.
Look carefully at the COLOR and SHAPE of the bacteria in this microscopy image.
STEP 1 - COLOR (Gram stain result):
- Purple or violet = Gram-POSITIVE
- Pink or red = Gram-NEGATIVE
STEP 2 - SHAPE:
- Round, spherical, oval cells = COCCI
- Rod-shaped, elongated cells = BACILLI
STEP 3 - Choose exactly ONE of these 4 classes:
1. Gram-positive Cocci
2. Gram-positive Bacilli
3. Gram-negative Cocci
4. Gram-negative Bacilli
Respond ONLY in this exact format, nothing else:
Class: [one of the 4 classes above]
Gram: [positive or negative]
Shape: [cocci or bacilli]
Confidence: [high / medium / low]
Reason: [one sentence describing what you see] and the answer cannot be None"""


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key fehlt")

    image_data = await file.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": PROMPT}
                    ]
                }]
            },
            timeout=30
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="OpenRouter Timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))

    text = response.json()["choices"][0]["message"]["content"]

    def extract(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else "Unknown"

    return {
        "class":      extract(r'class:\s*(.+)'),
        "gram":       extract(r'gram:\s*(.+)'),
        "shape":      extract(r'shape:\s*(.+)'),
        "confidence": extract(r'confidence:\s*(.+)'),
        "reason":     extract(r'reason:\s*(.+)'),
        "raw":        text
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "MUST Lab API läuft"}