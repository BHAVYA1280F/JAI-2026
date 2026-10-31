"""
Multi-Agent Lesion Pipeline
============================
Inputs:
  1. Image: file path, PIL Image, or numpy array.
  2. CSV metadata: dict, pandas Series, or path to a CSV file (e.g. age, sex, localization).

Architecture:
  - Image is passed to:
      * Model A1 (Image Quality Agent)
      * Model A2 (Diagnosis / Classification Agent)
      * Model A3 (Lesion ABCD Analysis Agent)
  - CSV metadata is passed to:
      * Model A4 (Patient Context & Metadata Reasoning Agent)
  - Outputs of A1, A2, A3, and A4 are passed to:
      * Model A5 (Safety Auditor Agent)
  - Output of A5 along with A1, A2, A3, and A4 are passed to:
      * Model A6 (Coordinator Agent)
  - Output of Model A6 is returned as the final decision.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Ensure agent modules are accessible
CURRENT_DIR = Path(__file__).resolve().parent
for sub in ["", "a1", "a3", "a4", "a5", "a6"]:
    p = str(CURRENT_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from model_1_quality.quality_engine import QualityEngine
from diagnosis_agent import DiagnosisAgent
from agent3_lesion_analysis import lesion_analysis_agent
from agent4_context import context_agent
from agent5 import safety_audit
from coordinator_agent import CoordinatorAgent


class LesionPipeline:
    """End-to-end multi-agent diagnostic & triage pipeline."""

    def __init__(
        self,
        diagnosis_ckpt: Optional[str] = None,
        device: Optional[str] = None,
    ):
        ckpt = diagnosis_ckpt or str(CURRENT_DIR / "outputs" / "diagnosis_agent.pt")
        self.a1_quality = QualityEngine()
        self.a2_diagnosis = DiagnosisAgent(ckpt=ckpt, device=device)
        self.a6_coordinator = CoordinatorAgent()

    @staticmethod
    def _load_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> tuple[Image.Image, np.ndarray]:
        """Convert input image into both PIL.Image (for A2) and BGR numpy array (for A1, A3)."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            pil_img = Image.open(path_str).convert("RGB")
            bgr_img = cv2.imread(path_str)
            if bgr_img is None:
                # Fallback if cv2 fails to read
                bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            # Assume RGB if 3 channels and standard format
            if len(image_input.shape) == 2:
                bgr_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 3:
                # Default assume BGR if numpy, but provide fallback
                bgr_img = image_input.copy()
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return pil_img, bgr_img

    @staticmethod
    def _parse_csv_input(
        csv_input: Union[str, Path, dict, pd.Series, pd.DataFrame],
        row_index: int = 0,
    ) -> Dict[str, Any]:
        """Extract metadata (age, sex, localization) from CSV file path, dict, or DataFrame."""
        if isinstance(csv_input, (str, Path)):
            df = pd.read_csv(csv_input)
            data = df.iloc[row_index].to_dict()
        elif isinstance(csv_input, pd.DataFrame):
            data = csv_input.iloc[row_index].to_dict()
        elif isinstance(csv_input, pd.Series):
            data = csv_input.to_dict()
        elif isinstance(csv_input, dict):
            data = csv_input
        else:
            raise TypeError(f"Unsupported CSV metadata type: {type(csv_input)}")

        # Normalize key names
        cleaned = {}
        for k, v in data.items():
            k_low = str(k).strip().lower()
            if k_low in ("age", "patient_age"):
                cleaned["age"] = v
            elif k_low in ("sex", "gender"):
                cleaned["sex"] = v
            elif k_low in ("localization", "location", "site", "body_site"):
                cleaned["location"] = v
            else:
                cleaned[k] = v

        return cleaned

    def run(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        csv_metadata: Union[str, Path, dict, pd.Series, pd.DataFrame],
        row_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Executes the 6-agent workflow:
          - Image -> A1, A2, A3
          - CSV Metadata -> A4
          - A1, A2, A3, A4 -> A5
          - A1, A2, A3, A4, A5 -> A6 (final decision)
        """
        # 1. Prepare inputs
        pil_img, bgr_img = self._load_image(image)
        metadata = self._parse_csv_input(csv_metadata, row_index=row_index)

        # 2. Pass Image to A1 (Quality)
        a1_output = self.a1_quality.analyze(bgr_img)

        # 3. Pass Image to A2 (Diagnosis)
        a2_output = self.a2_diagnosis.predict(pil_img)

        # 4. Pass Image to A3 (Lesion Analysis)
        a3_output = lesion_analysis_agent(bgr_img)

        # 5. Pass CSV Metadata to A4 (Context Agent, incorporating A2's predicted class if available)
        a4_output = context_agent(
            age=metadata.get("age"),
            sex=metadata.get("sex"),
            location=metadata.get("location"),
            predicted_class=a2_output.get("label"),
        )

        # 6. Map outputs of A1, A2, A3, A4 to A5 (Safety Auditor)
        a5_result = safety_audit(a1_output, a2_output, a3_output, a4_output)
        a5_output = a5_result.model_dump()

        # 7. Map A5 output along with A1, A2, A3, A4 to A6 (Coordinator)
        a6_result = self.a6_coordinator.decide(
            a1=a1_output,
            a2=a2_output,
            a3=a3_output,
            a4=a4_output,
            a5=a5_output,
        )
        final_output = a6_result.to_dict()

        return {
            "final_output": final_output,
            "final_label": a6_result.final_label,
            "triage_decision": a6_result.triage,
            "rationale": a6_result.rationale,
            "agent_outputs": {
                "a1": a1_output,
                "a2": a2_output,
                "a3": a3_output,
                "a4": a4_output,
                "a5": a5_output,
                "a6": final_output,
            },
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 6-agent Lesion analysis pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Path to input lesion image")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV metadata file")
    parser.add_argument("--row", type=int, default=0, help="Row index in CSV to use (default: 0)")

    args = parser.parse_args()

    pipeline = LesionPipeline()
    result = pipeline.run(image=args.image, csv_metadata=args.csv, row_index=args.row)

    print("\n" + "=" * 50)
    print("FINAL DECISION (MODEL A6):")
    print(f"  Final Label:     {result['final_label']}")
    print(f"  Triage Decision: {result['triage_decision']}")
    print("  Rationale:")
    for r in result["rationale"]:
        print(f"    - {r}")
    print("=" * 50)
