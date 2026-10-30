"""Sanity tests on synthetic images. Run: python -m tests.test_quality
These only prove the checks react in the right direction; real thresholds
must be calibrated on your own images."""
import cv2
import numpy as np

from model_1_quality import QualityEngine, StabilityTracker

CONTRACT_KEYS = {"agent", "score", "status", "reasons", "defects",
                 "confidence_modifier", "recommended_action", "decision"}


def make_lesion(size=600, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (150, 170, 220), np.uint8)             # skin (BGR)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))     # skin grain
    c = (size // 2, size // 2)
    cv2.ellipse(img, c, (size // 5, size // 6), 20, 0, 360, (40, 60, 110), -1)
    tex = rng.integers(0, 60, (size, size, 1), dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, c, (size // 5, size // 6), 20, 0, 360, 255, -1)
    img[mask > 0] = cv2.subtract(img[mask > 0], tex[mask > 0].repeat(3, 1) // 2)
    return img


def run():
    eng = QualityEngine()
    good = make_lesion()
    cases = {
        "good": good,
        "blurry": cv2.GaussianBlur(good, (0, 0), 12),
        "dark": (good * 0.2).astype(np.uint8),
        "bright": cv2.convertScaleAbs(good, alpha=1.0, beta=110),
        "glare": cv2.circle(good.copy(), (300, 300), 70, (255, 255, 255), -1),
        "tiny": cv2.resize(good, (100, 100)),
        "hairy": good.copy(),
    }
    rng = np.random.default_rng(1)
    for _ in range(60):
        p1 = (int(rng.integers(0, 600)), int(rng.integers(0, 600)))
        p2 = (p1[0] + int(rng.integers(-250, 250)), p1[1] + int(rng.integers(-250, 250)))
        cv2.line(cases["hairy"], p1, p2, (20, 25, 30), 2, cv2.LINE_AA)

    for name, img in cases.items():
        r = eng.analyze(img, name)
        assert CONTRACT_KEYS <= set(r), f"{name}: missing agent-contract keys"
        scores = {k: v["score"] for k, v in r["quality_analysis"].items()}
        print(f"{name:7s} overall={r['score']:5.1f} {r['status']:5s} mod={r['confidence_modifier']:.2f}",
              scores, r["reasons"])

    g = eng.analyze(cases["good"])
    assert g["decision"]["approved"], "good image should pass"
    assert g["status"] in ("PASS", "WARN") and g["recommended_action"] != "RE_IMAGE"
    assert g["confidence_modifier"] > 0

    for bad in ("blurry", "dark", "bright", "glare", "tiny", "hairy"):
        r = eng.analyze(cases[bad])
        assert not r["decision"]["approved"], f"{bad} should be rejected"
        assert r["status"] == "FAIL" and r["recommended_action"] == "RE_IMAGE"
        assert r["confidence_modifier"] == 0.0 and r["reasons"], f"{bad}: needs reasons"

    # defect flags point at the right cause
    assert eng.analyze(cases["blurry"])["defects"]["blur"] == "FAIL"
    assert eng.analyze(cases["dark"])["defects"]["low_illumination"] == "FAIL"
    assert eng.analyze(cases["bright"])["defects"]["overexposure"] == "FAIL"
    assert eng.analyze(cases["glare"])["defects"]["glare"] == "FAIL"
    assert eng.analyze(cases["hairy"])["defects"]["hair_obstruction"] == "FAIL"

    # hard-rejection rule: high overall score must not rescue a failed critical check
    r = eng.analyze(cases["blurry"])
    print("blurry overall:", r["score"], "-> approved:", r["decision"]["approved"])

    # framing is advisory: a framing problem alone must never reject
    off = np.roll(good, 250, axis=1)
    r = eng.analyze(off)
    assert r["quality_analysis"]["framing"]["severity"] in ("PASS", "WARN")

    t = StabilityTracker(3)
    ok = [t.update(eng.analyze(cases["good"])) for _ in range(3)]
    assert ok == [False, False, True]

    inv = eng.analyze_bytes(b"not an image")
    assert inv["decision"]["status"] == "REJECTED" and inv["status"] == "FAIL"
    assert CONTRACT_KEYS <= set(inv)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run()