import json
from pathlib import Path
base = Path(r"C:\dev\ecopin_ml_models\image_validation\artifacts\exp03_2_clip_refined")
for model in ["linear", "small_mlp", "reg_mlp"]:
    for split in ["val", "test"]:
        with open(base / model / f"{split}_results.json") as f:
            d = json.load(f)
        n_probs = len(d.get("probabilities", []))
        n_preds = len(d.get("predictions", []))
        print(f"{model}/{split}: n_probs={n_probs}  n_preds={n_preds}  keys={list(d.keys())[:6]}")
