"""
evaluate.py - measure how well the model separates NORMAL from PNEUMONIA.

  python evaluate.py                                     bundled GitHub images (75 images)       -> results/
  python evaluate.py --kaggle_dir "path/to/chest_xray"   Kaggle "Chest X-Ray Images (Pneumonia)"  -> results_kaggle/
  python evaluate.py --nih_dir "path/to/nih"             NIH ChestX-ray14, 14-disease metrics     -> results/

Outputs: metrics.csv, predictions.csv, confusion_matrix.png, roc_curve.png, verdict_counts.csv, thresholds.json
The app reads thresholds.json (results_kaggle/ if it exists, otherwise results/).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, multilabel_confusion_matrix)
from engine import XrayAI, PNEU_SIGNS

BAND = 0.10   # half-width of the "Uncertain" band around the chosen threshold
ai = XrayAI()


def score_images(paths):
    rows, times = [], []
    for p in paths:
        r = ai.predict(Image.open(p)); times.append(r.seconds)
        rows.append({"p_pneumonia": r.probs["Pneumonia"], "p_signs": max(r.probs[k] for k in PNEU_SIGNS)})
    return pd.DataFrame(rows), float(np.mean(times)) * 1000


def binary_report(paths, y, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    s, ms = score_images(paths); s["path"] = [os.path.basename(p) for p in paths]; s["y"] = y
    s.to_csv(f"{out_dir}/predictions.csv", index=False)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    figr, axr = plt.subplots(figsize=(5, 4.5)); best_signs = 0.5
    for k, (col, name) in enumerate([("p_pneumonia", "Pneumonia output only"), ("p_signs", "Combined pneumonia signs")]):
        auc = roc_auc_score(s.y, s[col]); fpr, tpr, thr = roc_curve(s.y, s[col]); best = float(thr[np.argmax(tpr - fpr)])
        if col == "p_signs": best_signs = best
        for tname, t in [("0.50", 0.5), (f"{best:.2f} (best)", best)]:
            pred = (s[col] >= t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(s.y, pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(s.y, pred, labels=[0, 1]).ravel()
            rows.append({"score": name, "threshold": tname, "AUROC": auc, "accuracy": accuracy_score(s.y, pred), "precision": pr,
                         "recall": rc, "specificity": tn / max(tn + fp, 1), "F1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn})
        cmx = confusion_matrix(s.y, (s[col] >= best).astype(int), labels=[0, 1]); ax = axes[k]
        ax.imshow(cmx, cmap="Blues")
        for (a, b), v in np.ndenumerate(cmx): ax.text(b, a, v, ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Normal", "Pneumonia"]); ax.set_yticklabels(["Normal", "Pneumonia"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name}\nthreshold {best:.2f}", fontsize=9)
        axr.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
    fig.tight_layout(); fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=140)
    axr.plot([0, 1], [0, 1], "k--"); axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate"); axr.legend(fontsize=8)
    axr.set_title("ROC: Normal vs Pneumonia"); figr.tight_layout(); figr.savefig(f"{out_dir}/roc_curve.png", dpi=140)
    res = pd.DataFrame(rows).round(3); res.to_csv(f"{out_dir}/metrics.csv", index=False)

    lo, hi = best_signs - BAND, best_signs + BAND
    v = s.p_signs.map(lambda x: "Pneumonia signs" if x >= hi else ("Looks normal" if x <= lo else "Uncertain"))
    vc = pd.crosstab(s.y.map({0: "Actual Normal", 1: "Actual Pneumonia"}), v); vc.to_csv(f"{out_dir}/verdict_counts.csv")
    json.dump({"best": best_signs, "lo": lo, "hi": hi, "n_normal": int((s.y == 0).sum()), "n_pneumonia": int((s.y == 1).sum()),
               "source": title, "ms_per_image": ms}, open(f"{out_dir}/thresholds.json", "w"), indent=1)
    print(f"{title}: {(s.y==0).sum()} normal / {(s.y==1).sum()} pneumonia images")
    print(res.to_string(index=False)); print("\nSimple-mode verdicts:\n", vc); print(f"\n{ms:.0f} ms/image")


def nih_mode(nih_dir, limit):
    csv = glob.glob(f"{nih_dir}/**/*Data_Entry*.csv", recursive=True)[0]
    paths = {os.path.basename(p): p for p in glob.glob(f"{nih_dir}/**/*.png", recursive=True)}
    df = pd.read_csv(csv); df = df[df["Image Index"].isin(paths)].sample(frac=1, random_state=0).head(limit).reset_index(drop=True)
    classes = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
               "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    Y = np.array([[float(c.replace("_", " ") in lab.replace("_", " ")) for c in classes] for lab in df["Finding Labels"]])
    P = np.array([[ai.predict(Image.open(paths[f])).probs[c] for c in classes] for f in df["Image Index"]]); pred = (P >= 0.5).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(Y, pred, zero_division=0)
    auc = [roc_auc_score(Y[:, i], P[:, i]) if 0 < Y[:, i].sum() < len(Y) else np.nan for i in range(len(classes))]
    res = pd.DataFrame({"class": classes, "AUROC": auc, "precision": pr, "recall": rc, "F1": f1, "support": sup.astype(int)}).round(3)
    res.loc[len(res)] = ["MACRO AVG", np.nanmean(auc), pr.mean(), rc.mean(), f1.mean(), int(sup.sum())]
    os.makedirs("results", exist_ok=True); res.round(3).to_csv("results/metrics_nih.csv", index=False); print(res.to_string(index=False))
    cms = multilabel_confusion_matrix(Y, pred); fig, axes = plt.subplots(3, 5, figsize=(15, 8.5))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(classes): ax.axis("off"); continue
        ax.imshow(cms[i], cmap="Blues"); ax.set_title(classes[i], fontsize=9); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for (a, b), v in np.ndenumerate(cms[i]): ax.text(b, a, v, ha="center", va="center")
    plt.tight_layout(); plt.savefig("results/confusion_matrices_nih.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/samples_meta.csv"); ap.add_argument("--img_dir", default="data/samples")
    ap.add_argument("--kaggle_dir"); ap.add_argument("--split", default="test"); ap.add_argument("--nih_dir")
    ap.add_argument("--limit", type=int, default=2000, help="max images per class (Kaggle) or in total (NIH)")
    a = ap.parse_args()
    if a.nih_dir:
        nih_mode(a.nih_dir, a.limit)
    elif a.kaggle_dir:
        paths, y = [], []
        for lab, folder in [(0, "NORMAL"), (1, "PNEUMONIA")]:
            fs = sorted(p for ext in ("jpeg", "jpg", "png") for p in glob.glob(f"{a.kaggle_dir}/**/{a.split}/{folder}/*.{ext}", recursive=True))[:a.limit]
            paths += fs; y += [lab] * len(fs)
        if not paths: raise SystemExit(f"No images found under {a.kaggle_dir}/{a.split}/NORMAL|PNEUMONIA")
        binary_report(paths, np.array(y), "results_kaggle", f"Kaggle Chest X-Ray Images (Pneumonia), {a.split} split")
    else:
        m = pd.read_csv(a.meta); m = m[m.group.isin(["Normal", "Pneumonia"])]
        binary_report([os.path.join(a.img_dir, f) for f in m.filename], (m.group == "Pneumonia").astype(int).values, "results",
                      "GitHub open X-ray collection (bundled)")
