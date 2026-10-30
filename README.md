# Model 1: Image Quality Assessment (DermaGuard)

Rule-based OpenCV quality gate. No Model 2 image is analysed unless Model 1 approves it.

## Run
    pip install -r requirements.txt
    python -m tests.test_quality          # synthetic sanity tests
    uvicorn backend.main:app --reload     # API at http://localhost:8000/docs

## Checks (model_1_quality/checks.py)
clarity (Laplacian variance) | illumination | glare | obstruction (hair heuristic) |
resolution | framing (colour-segmentation heuristic)

## Decision
approved = overall >= 80 AND every critical check passes. Weights and thresholds are in
config.py and are provisional: calibrate them on your own images and capture device.

## Webcam
Send a frame (JPEG blob, 5-10 fps) to /quality/frame, show `decision.feedback`, and unlock the
capture button only after ~5 consecutive approved frames (see StabilityTracker). Then POST the
captured photo to /predict, which re-checks it server-side.

## Known limits
- Obstruction only detects hair-like thin dark lines, not fingers or objects.
- Framing assumes the lesion differs in colour from surrounding skin; unreliable on
  dermoscopy images with a dark circular vignette.
- A passing quality score does not mean a diagnosis is medically reliable.
