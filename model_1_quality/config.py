"""All tunable numbers for Model 1 (Image Quality Agent) live here.

Every value is a PROVISIONAL starting point, not a medically validated
threshold. Calibrate against your own images and capture device.
"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class QualityConfig:
    # Images are shrunk to this longest side before analysis so scores do not
    # depend on file size (resolution is judged on the ORIGINAL size).
    analysis_size: int = 640

    # --- Blur: variance of the Laplacian ---
    blur_min_var: float = 100.0      # below this -> blurry (FAIL)
    blur_good_var: float = 300.0     # at/above this -> score 100

    # --- Illumination ---
    mean_ok_range: tuple = (80, 180)     # ideal mean gray level
    mean_hard_range: tuple = (45, 215)   # outside this -> FAIL
    max_clipped_fraction: float = 0.15   # too many pure-black/white pixels -> FAIL

    # --- Glare ---
    glare_v_min: int = 240
    glare_s_max: int = 50
    glare_max_fraction: float = 0.02
    glare_max_blob_fraction: float = 0.01

    # --- Obstruction (hair heuristic) ---
    hair_max_fraction: float = 0.06

    # --- Resolution (original image) ---
    min_side_px: int = 224
    good_side_px: int = 450

    # --- Framing (advisory only: unreliable on vignetted dermoscopy) ---
    lesion_area_range: tuple = (0.05, 0.85)
    border_margin: float = 0.01
    centre_tolerance: float = 0.30

    # --- Scoring / decision ---
    weights: dict = field(default_factory=lambda: {
        "clarity": 0.30, "illumination": 0.20, "glare": 0.15,
        "obstruction": 0.15, "resolution": 0.10, "framing": 0.10,
    })
    warn_score: float = 75.0          # a passing check below this -> WARN
    approve_threshold: float = 80.0   # overall score needed for approval
    full_confidence_score: float = 90.0  # overall >= this -> confidence_modifier 1.0
    # Checks whose failure forces rejection. Framing is advisory (WARN only).
    critical: tuple = ("clarity", "illumination", "glare", "obstruction", "resolution")

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        cfg = cls()
        with open(path) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
        return cfg