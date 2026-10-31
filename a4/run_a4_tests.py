import pandas as pd
from agent4_context import context_agent

df = pd.read_csv("a4_test_dataset.csv", dtype=str, keep_default_na=False)
res = []
for _, r in df.iterrows():
    o = context_agent(age=r["age"], sex=r["sex"], location=r["localization"],
                      predicted_class=r["predicted_class"])
    exp_risk = None if r["expected_risk_score"] == "" else float(r["expected_risk_score"])
    checks = {
        "risk": (o["risk_score"] is None and exp_risk is None) or
                (o["risk_score"] is not None and exp_risk is not None and abs(o["risk_score"] - exp_risk) <= 0.01),
        "level": o["risk_level"] == r["expected_risk_level"],
        "reliability": abs(o["context_reliability"] - float(r["expected_context_reliability"])) <= 0.001,
        "missing": sorted(o["missing_fields"]) == sorted(filter(None, r["expected_missing_fields"].split(";"))),
        "flags": sorted(o["context_flags"]) == sorted(filter(None, r["expected_context_flags"].split(";"))),
    }
    res.append({"test_id": r["test_id"], "result": "PASS" if all(checks.values()) else "FAIL",
                "failed_checks": ",".join(k for k, v in checks.items() if not v),
                "actual_risk": o["risk_score"], "actual_level": o["risk_level"],
                "actual_reliability": o["context_reliability"],
                "actual_missing": ";".join(o["missing_fields"]), "actual_flags": ";".join(o["context_flags"])})
out = pd.DataFrame(res)
out.to_csv("a4_test_results.csv", index=False)
print(out.to_string(index=False))
print(f"\n{(out.result=='PASS').sum()}/{len(out)} passed")
