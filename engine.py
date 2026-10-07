"""
engine.py - core AI logic (no UI code here, so it can be tested and reused).

Pipeline:  image -> grayscale + normalise -> centre-crop + resize 224
           -> pretrained DenseNet121 (torchxrayvision, 18 findings)
           -> probabilities (+ Grad-CAM heatmap for any chosen finding)
"""
import os, shutil, time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
from PIL import Image
from matplotlib import cm

def _install_bundled_weights():
    """Copy weights shipped in ./weights into the torchxrayvision cache, so no download is needed."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    dst = os.path.join(os.path.expanduser("~"), ".torchxrayvision", "models_data")
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith(".pt") and not os.path.exists(os.path.join(dst, f)):
                shutil.copy(os.path.join(src, f), dst)


_install_bundled_weights()
PNEU_SIGNS = ["Pneumonia", "Lung Opacity", "Consolidation", "Infiltration"]   # scores combined for the simple verdict
WEIGHTS = "densenet121-res224-all"   # trained on NIH, CheXpert, MIMIC, PadChest, etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Plain-language descriptions shown in the app
DESCRIPTIONS = {
    "Atelectasis": "Partial or complete collapse of lung tissue.",
    "Consolidation": "Lung air spaces filled with fluid/pus (often infection).",
    "Infiltration": "Substance denser than air within lung tissue.",
    "Pneumothorax": "Air in the pleural space (collapsed lung).",
    "Edema": "Fluid in the lungs, often heart-related.",
    "Emphysema": "Over-inflated, damaged air sacs (COPD).",
    "Fibrosis": "Scarring of lung tissue.",
    "Effusion": "Fluid around the lung (pleural effusion).",
    "Pneumonia": "Lung infection causing inflammation.",
    "Pleural_Thickening": "Thickening of the lung lining.",
    "Cardiomegaly": "Enlarged heart.",
    "Nodule": "Small round spot (<3 cm) in the lung.",
    "Mass": "Larger lesion (>3 cm) in the lung.",
    "Hernia": "Organ pushing through the diaphragm.",
    "Lung Lesion": "Abnormal area of lung tissue.",
    "Fracture": "Broken rib or bone in the chest area.",
    "Lung Opacity": "Cloudy area where lung should look dark.",
    "Enlarged Cardiomediastinum": "Widened central chest structures.",
}


@dataclass
class Result:
    probs: dict            # finding -> probability (0..1), 0.5 = model operating point
    image224: np.ndarray   # preprocessed image shown to the model (0..1)
    tensor: torch.Tensor   # model input
    seconds: float


class XrayAI:
    def __init__(self, weights: str = WEIGHTS):
        self.model = xrv.models.DenseNet(weights=weights).to(DEVICE).eval()
        self.pathologies = list(self.model.pathologies)
        self.tf = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
        )

    # ---------- preprocessing ----------
    def preprocess(self, pil_img: Image.Image):
        a = np.array(pil_img.convert("L")).astype(np.float32)
        a = xrv.datasets.normalize(a, 255)          # -> approx [-1024, 1024]
        a = self.tf(a[None, ...])                   # (1, 224, 224)
        x = torch.from_numpy(a).unsqueeze(0).float()
        shown = (a[0] + 1024) / 2048
        return x.to(DEVICE), np.clip(shown, 0, 1)

    # ---------- prediction ----------
    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Result:
        t0 = time.time()
        x, shown = self.preprocess(pil_img)
        out = self.model(x)[0].cpu().numpy()
        probs = {p: float(v) for p, v in zip(self.pathologies, out) if p}
        return Result(probs, shown, x, time.time() - t0)

    # ---------- Grad-CAM ----------
    def gradcam(self, x: torch.Tensor, finding: str) -> np.ndarray:
        """Heatmap (224x224, 0..1) of the image regions driving `finding`."""
        idx = self.pathologies.index(finding)
        self.model.zero_grad()
        feats = self.model.features(x)
        feats.retain_grad()
        pooled = F.adaptive_avg_pool2d(F.relu(feats), (1, 1)).flatten(1)
        logit = self.model.classifier(pooled)[0, idx]
        logit.backward()
        w = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1))[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)) / 255.0
        return cam

    @staticmethod
    def overlay(gray224: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        base = np.stack([gray224] * 3, -1)
        heat = cm.jet(cam)[..., :3]
        return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


# ---------- helpers used by UI + evaluation ----------
def risk_level(p: float) -> str:
    """Probabilities are calibrated so 0.5 is the model's operating threshold."""
    if p >= 0.75: return "High"
    if p >= 0.5:  return "Moderate"
    if p >= 0.35: return "Low"
    return "Minimal"


def image_quality_check(pil_img: Image.Image) -> list:
    """Cheap guardrails: returns (message_key, argument) pairs for inputs unlikely to be a usable chest X-ray."""
    warns = []
    rgb = np.array(pil_img.convert("RGB")).astype(float)
    if np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean() > 12:
        warns.append(("warn_color", None))
    w, h = pil_img.size
    if min(w, h) < 224:
        warns.append(("warn_res", f"{w}x{h}"))
    if np.array(pil_img.convert("L")).astype(float).std() < 20:
        warns.append(("warn_contrast", None))
    return warns
