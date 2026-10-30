"""FastAPI pipeline for DermaGuard.  Run:  uvicorn backend.main:app --reload

POST /analyze          - full pipeline: Image Quality Agent -> Diagnosis Agent -> Triage,
                         returns every agent payload plus a decision and agent_trace
POST /quality/analyze  - upload an image, get the full quality report
POST /quality/frame    - same engine, for webcam frames (send JPEG blobs ~5-10 fps)
POST /predict          - quality gate enforced on the BACKEND: Model 1 re-checks the
                         image itself; Model 2 only runs if it is approved.
GET  /metrics          - dataset-level quality-agent performance from the last evaluate run
GET  /health
"""
import json
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_1_quality import QualityConfig, QualityEngine

app = FastAPI(title="DermaGuard - Agentic Skin Lesion Screening")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

CALIB = os.environ.get("QUALITY_CONFIG", "calibrated_config.json")
engine = QualityEngine(QualityConfig.from_json(CALIB) if os.path.exists(CALIB) else QualityConfig())
MAX_BYTES = 15 * 1024 * 1024

# Draft triage thresholds (provisional; tune on validation predictions).
MALIGNANT = {"mel", "bcc", "akiec"}
P_MALIGNANT_REFER = 0.20
MIN_CONFIDENCE = 0.70
MAX_ENTROPY = 1.0
MIN_TTA_AGREEMENT = 0.75


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    return data


@app.get("/health")
def health():
    return {"status": "ok", "agents": ["image_quality", "diagnosis", "triage"]}


@app.get("/metrics")
def metrics():
    """Dataset-level performance (sensitivity, specificity...) from the last evaluate run."""
    path = os.path.join("evaluation", "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(404, "Run: python -m model_1_quality.evaluate first")
    with open(path) as f:
        return json.load(f)


@app.post("/quality/analyze")
async def analyze_quality(file: UploadFile = File(...)):
    return engine.analyze_bytes(await _read(file), image_id=file.filename or "upload")


@app.post("/quality/frame")
async def frame(file: UploadFile = File(...)):
    return engine.analyze_bytes(await _read(file), image_id="webcam_frame")


def model_2_predict(image_bytes: bytes, quality_report: dict):
    """Placeholder: replace with the HAM10000 Diagnosis Agent.

    Expected return: {"label", "confidence", "probabilities": {...}, "p_malignant",
                      "entropy", "tta_agreement"}
    """
    raise NotImplementedError("Plug in Model 2 here")


def triage(quality: dict, diagnosis: dict | None):
    """Draft risk-asymmetric triage: quality -> malignancy -> uncertainty -> benign."""
    if quality["status"] == "FAIL":
        why = "; ".join(quality["reasons"]) or "image quality below threshold"
        return "RE_IMAGE", f"Image quality check failed: {why}."
    if diagnosis is None:
        return None, "Diagnosis Agent is not connected yet; no triage decision issued."

    label = diagnosis["label"]
    p_mal = diagnosis.get("p_malignant", 0.0)
    conf = diagnosis["confidence"] * quality["confidence_modifier"]
    entropy = diagnosis.get("entropy", 0.0)
    tta = diagnosis.get("tta_agreement", 1.0)

    if p_mal > P_MALIGNANT_REFER or label in MALIGNANT:
        return "REFER", (f"Malignant risk: predicted '{label}', p_malignant={p_mal:.2f} "
                         f"(threshold {P_MALIGNANT_REFER}).")
    if conf < MIN_CONFIDENCE or entropy > MAX_ENTROPY or tta < MIN_TTA_AGREEMENT:
        return "REFER", (f"Uncertain prediction: quality-adjusted confidence={conf:.2f}, "
                         f"entropy={entropy:.2f}, TTA agreement={tta:.2f}.")
    return "AUTO_TRIAGE", (f"Benign class '{label}' with quality-adjusted confidence {conf:.2f}, "
                           f"low entropy and consistent TTA; quality {quality['status']}.")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await _read(file)
    image_id = file.filename or "upload"
    quality = engine.analyze_bytes(data, image_id=image_id)

    diagnosis, diag_note = None, "skipped: image quality failed"
    if quality["status"] != "FAIL":
        try:
            diagnosis = model_2_predict(data, quality)
            diag_note = f"predicted {diagnosis['label']} (confidence {diagnosis['confidence']:.2f})"
        except NotImplementedError:
            diag_note = "not connected yet"

    decision, rationale = triage(quality, diagnosis)
    trace = [
        {"agent": "image_quality", "status": quality["status"], "score": quality["score"],
         "contribution": "; ".join(quality["reasons"]) or "image quality is good",
         "confidence_modifier": quality["confidence_modifier"]},
        {"agent": "diagnosis", "contribution": diag_note},
        {"agent": "triage", "contribution": rationale},
    ]
    return {"image_id": image_id, "decision": decision, "rationale": rationale,
            "quality": quality, "diagnosis": diagnosis, "agent_trace": trace}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await _read(file)
    report = engine.analyze_bytes(data, image_id=file.filename or "upload")
    if not report["decision"]["approved"]:
        return {"status": "REJECTED", "quality_report": report, "diagnosis": None}
    try:
        diagnosis = model_2_predict(data, report)
    except NotImplementedError:
        diagnosis = None  # Model 2 not connected yet
    return {"status": "APPROVED", "quality_report": report, "diagnosis": diagnosis}


if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")