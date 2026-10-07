"""
download_data.py - download X-ray data from the internet (Normal vs Pneumonia only).

    python download_data.py          # open GitHub dataset subset (~40 MB, no account needed)
    python download_data.py --nih    # also the NIH ChestX-ray14 sample from Kaggle (needs a Kaggle token)

For a LARGER test, download Kaggle's "Chest X-Ray Images (Pneumonia)" (Paul Mooney) and run:
    python evaluate.py --kaggle_dir "path/to/chest_xray"
The pretrained model weights are already bundled in ./weights.
"""
import argparse, os, subprocess
import pandas as pd

RAW = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master"   # open X-ray collection on GitHub
EXCLUDE = "COVID|SARS|MERS|CoV|Tuberculosis|Pneumocystis"                           # keep ONLY ordinary pneumonia vs normal
os.makedirs("data/samples", exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--nih", action="store_true"); a = ap.parse_args()

subprocess.run(["curl", "-sL", "-o", "data/metadata.csv", f"{RAW}/metadata.csv"], check=True)
d = pd.read_csv("data/metadata.csv")
d = d[(d.modality == "X-ray") & (d.finding != "todo") & d.view.isin(["PA", "AP", "AP Erect"])]
d = d[~d.finding.str.contains(EXCLUDE, case=False, regex=True)]
normal = d[d.finding == "No Finding"].drop_duplicates("patientid").assign(group="Normal")
pneu = d[d.finding.str.startswith("Pneumonia")].drop_duplicates("patientid").head(60).assign(group="Pneumonia")
s = pd.concat([normal, pneu]); s.to_csv("data/samples_meta.csv", index=False)

ok = 0
for _, r in s.iterrows():
    out = f"data/samples/{r.filename}"
    if not os.path.exists(out):
        subprocess.run(["curl", "-sL", "-f", "-o", out, f"{RAW}/images/{r.filename}"])
    ok += os.path.exists(out) and os.path.getsize(out) > 5000
print(f"Downloaded {ok}/{len(s)} images ({len(normal)} normal, {len(pneu)} pneumonia) -> data/samples")

if a.nih:
    import kagglehub
    path = kagglehub.dataset_download("nih-chest-xrays/sample")
    print("NIH sample downloaded to:", path, '\nEvaluate with: python evaluate.py --nih_dir "%s"' % path)
