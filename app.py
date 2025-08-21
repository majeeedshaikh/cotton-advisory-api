import os, io, json
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from ultralytics import YOLO
import google.generativeai as genai
from advisory_prompt import SYSTEM_PROMPT

# ---------- Config via env ----------
MODEL_PATH    = os.getenv("MODEL_PATH", "weights/best.pt")
USE_ONNX      = os.getenv("USE_ONNX", "false").lower() == "true"
GEMINI_MODEL  = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")
SERVICE_KEY   = os.getenv("SERVICE_API_KEY", "")          # optional lightweight auth
CONF_DEFAULT  = float(os.getenv("CONF_DEFAULT", "0.20"))  # lower for higher recall
IOU_DEFAULT   = float(os.getenv("IOU_DEFAULT", "0.50"))
IMGSZ_DEFAULT = int(os.getenv("IMGSZ_DEFAULT", "896"))
TTA_DEFAULT   = os.getenv("TTA_DEFAULT", "true").lower() == "true"
INCLUDE_HEALTHY_DEFAULT = os.getenv("INCLUDE_HEALTHY_DEFAULT", "false").lower() == "true"

# ---------- Load model ----------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)  # works for .pt or .onnx
CLASS_NAMES = model.names                  # {id: name}

# ---------- Gemini ----------
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# map raw YOLO class to advisory disease name
NAME_NORMALIZATION = {
    "healthy": "healthy",
    "leaf_curl": "leaf_curl",
    "curl_stage1": "leaf_curl",
    "curl_stage2": "leaf_curl",
    "leaf_enation": "leaf_enation",
    "sooty": "sooty_mold",
}

app = FastAPI(title="Cotton Disease Advisory API", version="1.0.0")

class PredictResponse(BaseModel):
    detections: dict
    summary: dict
    advisory: Optional[dict] = None

def pil_to_numpy_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))

def results_to_summary(res) -> dict:
    """One-image Ultralytics Results -> compact JSON + severity."""
    h, w = res.orig_shape
    img_area = float(h*w)
    rows = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy().astype(float)
        confs = res.boxes.conf.cpu().numpy().astype(float)
        clsids= res.boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2), c, k in zip(xyxy, confs, clsids):
            raw = CLASS_NAMES[int(k)]
            disease = NAME_NORMALIZATION.get(raw, raw)
            a = max(0.0, (x2-x1)*(y2-y1))
            rows.append({
                "class_id": int(k),
                "class_name_raw": raw,
                "disease": disease,
                "conf": float(c),
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "box_area_frac": float(a / img_area)
            })
    per = {}
    for r in rows:
        d = r["disease"]
        per.setdefault(d, {"count":0, "conf_sum":0.0, "area_sum":0.0})
        per[d]["count"] += 1
        per[d]["conf_sum"] += r["conf"]
        per[d]["area_sum"] += r["box_area_frac"]
    for d,v in per.items():
        v["mean_conf"] = v["conf_sum"]/max(1, v["count"])
        # simple severity score (tune if needed)
        score = 0.65*v["area_sum"] + 0.35*min(1.0, v["count"]/6.0)
        v["severity_score"] = round(score, 3)
        v["severity"] = "low" if score < 0.05 else ("moderate" if score < 0.15 else "high")
        del v["conf_sum"]
    return {
        "image_size": {"width": int(w), "height": int(h)},
        "per_disease": per,
        "all_detections": rows
    }

def make_gemini_advisory(image_bytes: bytes, det_summary: dict, user_prompt: Optional[str]) -> dict:
    if not GEMINI_KEY:
        return {"error": "GEMINI_API_KEY not set", "advisory": None}
    model_g = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    user_msg = "Generate advisory JSON for Pakistani cotton farmers using the detections below."
    if user_prompt:
        user_msg += " User prompt: " + user_prompt
    resp = model_g.generate_content(
        [image_part, user_msg, json.dumps(det_summary, ensure_ascii=False)],
        generation_config=genai.types.GenerationConfig(
            temperature=0.6, max_output_tokens=700, response_mime_type="application/json"
        )
    )
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        # fallback: return raw text so client can see what happened
        return {"_raw": text}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "classes": CLASS_NAMES}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    user_prompt: Optional[str] = Form(None),
    conf: float = Form(CONF_DEFAULT),
    iou: float = Form(IOU_DEFAULT),
    imgsz: int = Form(IMGSZ_DEFAULT),
    tta: bool = Form(TTA_DEFAULT),
    include_healthy: bool = Form(INCLUDE_HEALTHY_DEFAULT),
    api_key: Optional[str] = Form(None),             # optional service key
):
    if SERVICE_KEY and api_key != SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # read image
    img_bytes = await file.read()
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    np_img = pil_to_numpy_rgb(pil)

    # choose classes (optionally drop 'healthy' for recall)
    if include_healthy:
        classes = None
    else:
        classes = [cid for cid, nm in CLASS_NAMES.items() if nm != "healthy"]

    # YOLO inference (with optional TTA)
    results = model.predict(
        source=np_img,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        augment=bool(tta),     # TTA for higher recall
        classes=classes,
        verbose=False,
        device="cpu"           # Railway has no GPU
    )
    res = results[0]
    det_summary = results_to_summary(res)

    # Minimal detection dict for clients that want raw boxes back too
    det_export = {
        "names": CLASS_NAMES,
        "speed_ms": {"pre": res.speed.get("preprocess", None),
                     "inf": res.speed.get("inference", None),
                     "post": res.speed.get("postprocess", None)},
        "summary": det_summary
    }

    advisory = make_gemini_advisory(img_bytes, det_summary, user_prompt)

    return JSONResponse(
        content={
            "detections": det_export,
            "summary": det_summary,
            "advisory": advisory
        }
    )
