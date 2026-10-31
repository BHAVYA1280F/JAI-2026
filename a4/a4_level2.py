"""
Agent 4 (Reasoning / Context) - Level 2 runner.

IN : records.csv  (image_id, age, sex, localization)   [dx, dx_type, lesion_id are withheld]
OUT: per image_id -> age, sex, site, missing_fields (+ risk score / level / reliability / evidence)

A4 is an independent read: it uses ONLY records.csv. It never sees the image and never
uses A2's label, so it passes no predicted_class (plausibility conflicts are left to A5 Audit).

Usage:
  python a4_level2.py                                   (asks for the records.csv path)
  python a4_level2.py --records records.csv
  python a4_level2.py --records records.csv --images images/    (also checks the CSV covers the image set exactly)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from agent4_context import context_agent

ap = argparse.ArgumentParser()
ap.add_argument("--records", default=None, help="path to records.csv")
ap.add_argument("--images", default=None, help="folder of image files, to verify image_id coverage")
ap.add_argument("--out", default=None, help="output folder (default: a4_outputs next to your CSV)")
args = ap.parse_args()

path = args.records or input("Enter the path of your records CSV: ").strip().strip('"').strip("'")
if not os.path.isfile(path):
    sys.exit(f"File not found: {path}")
if args.out is None:
    args.out = os.path.join(os.path.dirname(os.path.abspath(path)), "a4_outputs")

df = pd.read_csv(path, dtype=str, keep_default_na=False)
low = {c.lower().strip(): c for c in df.columns}
if "image_id" not in low:
    sys.exit(f"records.csv must have an image_id column. Found: {list(df.columns)}")
idc = low["image_id"]
agec = low.get("age"); sexc = low.get("sex")
locc = next((low[n] for n in ["localization", "localisation", "location", "site"] if n in low), None)
for name, c in [("age", agec), ("sex", sexc), ("localization", locc)]:
    if c is None:
        print(f"WARNING: no '{name}' column found, so it will be treated as missing for every image")

# Withheld columns must not be used
leaked = [c for c in df.columns if c.lower() in ("dx", "dx_type", "lesion_id")]
if leaked:
    print(f"NOTE: ignoring withheld columns present in the file: {leaked}")

df["_id"] = df[idc].str.strip().str.replace(r"\.(jpe?g|png)$", "", regex=True, case=False)

# ---- coverage checks: the CSV must cover the image set exactly ----
problems = []
dups = df["_id"][df["_id"].duplicated()].unique().tolist()
blank = int((df["_id"] == "").sum())
if dups:  problems.append(f"{len(dups)} duplicated image_id(s): {dups[:10]}")
if blank: problems.append(f"{blank} blank image_id row(s)")
if args.images:
    if not os.path.isdir(args.images):
        sys.exit(f"Image folder not found: {args.images}")
    files = {os.path.splitext(f)[0] for f in os.listdir(args.images)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    ids = set(df["_id"])
    no_record = sorted(files - ids); no_image = sorted(ids - files)
    print(f"Images in folder: {len(files)} | rows in records.csv: {len(df)}")
    if no_record: problems.append(f"{len(no_record)} image(s) have NO row in records.csv: {no_record[:10]}")
    if no_image:  problems.append(f"{len(no_image)} records.csv row(s) have NO image: {no_image[:10]}")
if problems:
    print("\nCOVERAGE PROBLEMS (records.csv does not match the image set exactly):")
    for p in problems: print("  -", p)
else:
    print("Coverage check: OK" if args.images else "Coverage check skipped (no --images given)")

# ---- run A4 per image_id ----
get = lambda r, c: r[c] if c else None
out_rows = []
for _, r in df.iterrows():
    o = context_agent(age=get(r, agec), sex=get(r, sexc), location=get(r, locc))   # no predicted_class
    out_rows.append({
        "image_id": r["_id"], "agent": "A4",
        "age": o["used"]["age"], "sex": o["used"]["sex"], "site": o["used"]["location"],
        "missing_fields": o["missing_fields"],
        "risk_score": o["risk_score"], "risk_level": o["risk_level"],
        "context_reliability": o["context_reliability"], "evidence": o["evidence"],
    })

os.makedirs(args.out, exist_ok=True)
json.dump(out_rows, open(f"{args.out}/a4_output.json", "w"), indent=2)
flat = pd.DataFrame(out_rows)
flat["missing_fields"] = flat["missing_fields"].apply(";".join)
flat["evidence"] = flat["evidence"].apply(" | ".join)
flat.to_csv(f"{args.out}/a4_output.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", None)
print("\n" + "=" * 20 + " A4 OUTPUT (one row per image) " + "=" * 20)
print(flat.drop(columns=["agent", "evidence"]).to_string(index=False))
print(f"\nProcessed {len(flat)} images")
print("Missing-field counts:"); print(flat["missing_fields"].replace("", "none").value_counts().to_string())
print("Risk levels:"); print(flat["risk_level"].value_counts().to_string())
print(f"\nSaved: {args.out}/a4_output.json  (for the pipeline / A5)")
print(f"       {args.out}/a4_output.csv   (for reading, includes evidence text)")
