"""All tunable numbers for Model 1 live here.

Every value is a PROVISIONAL starting point, not a medically validated
threshold. Calibrate against your own images and capture device.

SCORING SCALE (the same for every feature)
  worst  -> 0      the feature is as bad as it realistically gets
  boundary -> 50   exactly at its pass/fail threshold
  best   -> 100    ideal
Raw measurements have different units (Laplacian variance, 0-255 brightness,
area fractions, pixels). Mapping each one onto this shared scale around its OWN
threshold means no feature dominates just because of its units; how much each
feature matters is then decided only by `weights`.
"""
import json
from dataclasses import asdict, dataclass, field


@dataclass
class QualityConfig:
    # Images are shrunk to this longest side before analysis so scores do not
    # depend on file size (resolution is judged on the ORIGINAL size).
    analysis_size: int = 640

    # --- Clarity: variance of the Laplacian (scaled in log space: heavy-tailed) ---
    blur_worst_var: float = 10.0     # -> score 0
    blur_min_var: float = 100.0      # boundary: below this -> FAIL (score 50)
    blur_good_var: float = 300.0     # -> score 100

    # --- Illumination ---
    mean_ok_range: tuple = (80, 180)     # ideal mean gray level -> score 100
    mean_hard_range: tuple = (45, 215)   # boundary: outside this -> FAIL (score 50)
    mean_worst: tuple = (20, 245)        # -> score 0
    max_clipped_fraction: float = 0.15   # boundary for pure-black / pure-white pixels
    clipped_worst: float = 0.50

    # --- Glare / Obstruction: scaled by value / threshold (1.0 = boundary) ---
    glare_v_min: int = 240
    glare_s_max: int = 50
    glare_max_fraction: float = 0.02
    glare_max_blob_fraction: float = 0.01
    hair_max_fraction: float = 0.06
    ratio_worst: float = 5.0         # 5x over the threshold -> score 0

    # --- Resolution (original image, shorter side in px) ---
    min_side_px: int = 224           # boundary (score 50)
    good_side_px: int = 450          # -> score 100
    worst_side_px: int = 112         # -> score 0

    # --- Framing ---
    lesion_area_range: tuple = (0.05, 0.85)
    border_margin: float = 0.01
    centre_tolerance: float = 0.30

    # --- Scoring / decision ---
    # Importance of each feature. Deliberately unequal; they should sum to 1.
    weights: dict = field(default_factory=lambda: {
        "clarity": 0.30, "illumination": 0.20, "glare": 0.15,
        "obstruction": 0.15, "resolution": 0.10, "framing": 0.10,
    })
    approve_threshold: float = 80.0
    warn_score: float = 75.0   # passed-but-below-this -> warning (halfway between boundary and ideal)
    # Checks that can reject an image on their own. Framing is advisory: its colour
    # heuristic is unreliable on dermoscopy (dark circular border), so it can only warn.
    critical: tuple = ("clarity", "illumination", "glare", "obstruction", "resolution")

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    # Decision policy always comes from code, never from a saved calibration file.
    POLICY_KEYS = ("critical", "weights", "approve_threshold", "warn_score")

    @classmethod
    def from_json(cls, path):
        cfg = cls()
        with open(path) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k) and k not in cls.POLICY_KEYS:
                    setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
        return cfg