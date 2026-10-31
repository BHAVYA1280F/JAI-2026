"""
Batch inference script for the 6-agent Lesion pipeline on health-hidden dataset.
Produces health_submission.csv containing:
image_id,label,triage
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add Lesion directories to sys.path
LESION_DIR = Path("/home/user1/Lesion")
for sub in ["", "a1", "a3", "a4", "a5", "a6"]:
    p = str(LESION_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import LesionPipeline


def run_batch():
    records_path = LESION_DIR / "health-hidden" / "Health" / "hidden" / "records.csv"
    images_dir = LESION_DIR / "health-hidden" / "Health" / "hidden" / "images"
    output_path = LESION_DIR / "health_submission.csv"

    print(f"Loading records from: {records_path}")
    df_records = pd.read_csv(records_path)
    print(f"Total records: {len(df_records)}")

    # Initialize pipeline
    print("Initializing LesionPipeline (GPU enabled if available)...")
    pipeline = LesionPipeline(device="cuda")

    results = []
    start_time = time.time()

    for idx, row in df_records.iterrows():
        image_id = str(row["image_id"]).strip()
        img_filename = f"{image_id}.jpg"
        img_path = images_dir / img_filename

        if not img_path.exists():
            print(f"Warning: image {img_path} not found!")
            results.append({
                "image_id": image_id,
                "label": "unknown",
                "triage": "RE_IMAGE"
            })
            continue

        res = pipeline.run(
            image=str(img_path),
            csv_metadata=row.to_dict()
        )

        final_label = res["final_label"]
        triage = res["triage_decision"]

        results.append({
            "image_id": image_id,
            "predicted_label": final_label,
            "decision": triage
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(df_records):
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{len(df_records)} images ({elapsed:.1f}s)...")

    # Create submission dataframe
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)
    print(f"\nSubmission CSV saved successfully to: {output_path}")
    print("\nSummary of results:")
    print("Label distribution:")
    print(df_out["predicted_label"].value_counts())
    print("\nTriage distribution:")
    print(df_out["decision"].value_counts())


if __name__ == "__main__":
    run_batch()
