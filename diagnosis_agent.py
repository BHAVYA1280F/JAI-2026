"""Diagnosis Agent: inference wrapper + FastAPI router for the multi-agent pipeline."""
import io
import os
from typing import Dict, List

import numpy as np
import timm
import torch
from PIL import Image
from pydantic import BaseModel

from common import CLASSES, NUM, agent_outputs, predict_logits, prep_image


class DiagnosisOutput(BaseModel):
    label: str
    confidence: float                 # calibrated probability of the top class
    probabilities: Dict[str, float]   # calibrated, all 7 classes
    top2: List[str]
    p_malignant: float                # P(akiec)+P(bcc)+P(mel): drives asymmetric triage risk
    entropy: float                    # normalised predictive entropy, 0 (certain) .. 1 (uniform)
    tta_agreement: float              # fraction of TTA views agreeing with the final label
    tta_std: float                    # instability of the top-class probability across views


class DiagnosisAgent:
    def __init__(self, ckpt="outputs/diagnosis_agent.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt, map_location=self.device)
        self.model = timm.create_model(ck["model_name"], pretrained=False, num_classes=NUM)
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.to(self.device).to(memory_format=torch.channels_last).eval()
        self.T = ck["temperature"]

    def predict_batch(self, images: List[Image.Image]) -> List[dict]:
        X = np.stack([prep_image(im) for im in images])
        o = agent_outputs(predict_logits(self.model, X, self.device, tta=True), self.T)
        res = []
        for i in range(len(images)):
            p = o["probs"][i]; order = p.argsort()[::-1]
            res.append(dict(
                label=CLASSES[order[0]], confidence=float(p[order[0]]),
                probabilities={c: float(p[k]) for k, c in enumerate(CLASSES)},
                top2=[CLASSES[order[0]], CLASSES[order[1]]],
                p_malignant=float(o["p_malignant"][i]), entropy=float(o["entropy"][i]),
                tta_agreement=float(o["tta_agreement"][i]), tta_std=float(o["tta_std"][i])))
        return res

    def predict(self, image: Image.Image) -> dict:
        return self.predict_batch([image])[0]


def create_router(agent: DiagnosisAgent):
    from fastapi import APIRouter, File, UploadFile
    router = APIRouter()

    @router.post("/diagnose", response_model=DiagnosisOutput)
    async def diagnose(file: UploadFile = File(...)):
        return agent.predict(Image.open(io.BytesIO(await file.read())))

    return router


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    app = FastAPI(title="DermaGuard")
    app.include_router(create_router(DiagnosisAgent(os.environ.get("DERMA_CKPT", "outputs/diagnosis_agent.pt"))))
    uvicorn.run(app, host="0.0.0.0", port=8000)
