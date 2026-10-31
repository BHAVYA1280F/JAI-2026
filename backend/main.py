"""FastAPI pipeline.  Run from the project folder:  python -m uvicorn backend.main:app --reload

POST /analyze          - full pipeline: Image Quality Agent -> Diagnosis Agent -> Triage Agent
POST /quality/analyze  - Image Quality Agent only (full report)
POST /quality/frame    - Image Quality Agent for webcam frames
POST /predict          - strict gate: diagnosis runs only if the image is approved
GET  /metrics          - calibration test results from the last evaluate run
GET  /dataset-summary  - results of the last run_dataset run (approval rate, accuracy)
"""
import json
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.triage import run_triage
from model_1_quality import QualityConfig, QualityEngine

app = FastAPI(title="DermaGuard")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

CALIB = os.environ.get("QUALITY_CONFIG", "calibrated_config.json")
engine = QualityEngine(QualityConfig.from_json(CALIB) if os.path.exists(CALIB) else QualityConfig())
MAX_BYTES = 15 * 1024 * 1024


def run_diagnosis(image_bytes: bytes, quality: dict):
    """CONNECT YOUR DIAGNOSIS AGENT HERE.

    Return a dict with: label, confidence, probabilities{}, p_malignant, entropy,
    tta_agreement. `quality["quality_factor"]` (0-1) is available if you want to use it.
    Raise NotImplementedError while it is not connected.
    """
    raise NotImplementedError("Diagnosis Agent not connected")


def _diagnose(data, quality):
    if not quality["quality_analysis"]:        # unreadable image: nothing to diagnose
        return None
    try:
        return run_diagnosis(data, quality)
    except NotImplementedError:
        return None


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    return data


@app.post("/analyze")
async def analyze_full(file: UploadFile = File(...)):
    data = await _read(file)
    quality = engine.analyze_bytes(data, image_id=file.filename or "upload")
    diagnosis = _diagnose(data, quality)
    return {"image_quality": quality, "diagnosis": diagnosis, "triage": run_triage(quality, diagnosis)}


@app.post("/quality/analyze")
async def quality_analyze(file: UploadFile = File(...)):
    return engine.analyze_bytes(await _read(file), image_id=file.filename or "upload")


@app.get("/metrics")
def metrics():
    path = os.path.join("evaluation", "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Run: python -m model_1_quality.evaluate first")
    with open(path) as f:
        return json.load(f)


@app.get("/dataset-summary")
def dataset_summary():
    """Results of the last `python -m model_1_quality.run_dataset` run."""
    path = os.path.join("evaluation_run", "summary.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Run: python -m model_1_quality.run_dataset first")
    with open(path) as f:
        return json.load(f)


@app.post("/quality/frame")
async def quality_frame(file: UploadFile = File(...)):
    return engine.analyze_bytes(await _read(file), image_id="webcam_frame")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await _read(file)
    report = engine.analyze_bytes(data, image_id=file.filename or "upload")
    if not report["decision"]["approved"]:
        return {"status": "REJECTED", "quality_report": report, "diagnosis": None}
    return {"status": "APPROVED", "quality_report": report, "diagnosis": _diagnose(data, report)}


# Must stay last so it does not shadow the API routes above.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")