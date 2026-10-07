# Chest X-ray AI Assistant: Pneumonia or Normal

A deep-learning app that looks at a chest X-ray and says **Pneumonia signs detected**, **Looks normal**, or **Uncertain: needs a human reader**. It shows a heatmap of where the AI looked and gives simple tips for the patient. The interface works in **English and Kinyarwanda**, in **light or dark** theme.

## 1. Problem and approach
- **Domain:** image recognition (medical imaging), helping a reader prioritise and double-check chest X-rays.
- **Approach:** Deep Learning with a CNN, using transfer learning from a **pretrained DenseNet121** (torchxrayvision, `densenet121-res224-all`).
- **Why this model:** DenseNet121 is the CheXNet architecture, already trained on 8 public chest X-ray datasets, so it starts with chest-specific knowledge. About 7M parameters, so it runs on a laptop CPU (about 0.3 s per image). It scores 18 findings; the app turns the four pneumonia-related scores (Pneumonia, Lung Opacity, Consolidation, Infiltration) into one simple result.

## 2. Architecture
```
Chest X-ray (PNG/JPG)
  -> input checks (colour, resolution, contrast)
  -> greyscale, normalise, centre-crop, resize 224 x 224
  -> DenseNet121 CNN (pretrained)
  -> 18 scores; the highest of 4 pneumonia-related scores = pneumonia-sign score
  -> Pneumonia signs / Looks normal / Uncertain  + Grad-CAM heatmap + patient tips   (Streamlit UI)
```
| File | Role |
|---|---|
| `engine.py` | Preprocessing, model, Grad-CAM, input checks |
| `app.py` | Streamlit interface (themes, language switch, results, tips) |
| `i18n.py` | All text in English and Kinyarwanda, including patient tips |
| `evaluate.py` | Metrics, confusion matrix, ROC; bundled data, Kaggle pneumonia dataset, NIH multi-label |
| `download_data.py` | Downloads the data from the internet |
| `weights/`, `data/`, `sample_images/`, `results/` | Bundled model weights, 75 test images, demo images, measured results |

## 3. Run it
Everything is bundled. Only Python packages need installing (Python 3.10-3.12).

**Windows:** extract the zip, open `chest_xray_ai`, double-click `setup_and_run.bat`. Double-click `run_evaluation.bat` to recompute the metrics.

**Manual:**
```bash
cd chest_xray_ai
pip install -r requirements.txt
streamlit run app.py
python evaluate.py            # optional
```

## 4. Using the app
- Sidebar: choose **Language** (English / Kinyarwanda), **Theme** (Light / Dark), then upload an X-ray or pick a sample.
- **Simple mode:** one result card, heatmap, how-the-AI-decided, patient tips and downloadable report.
- **Detailed mode:** all 18 findings with an adjustable threshold and a heatmap per finding.
- The result uses two cut-offs around the best threshold found by `evaluate.py`: at or below the lower cut-off = *Looks normal*, at or above the upper = *Pneumonia signs*, in between = *Uncertain*.

## 5. Evaluation (measured, Normal vs Pneumonia)
Test set: 75 X-rays from an open collection on GitHub, not used to train the model: 15 normal, 60 pneumonia (bacterial, viral, lipoid and other causes).

| Score | Threshold | AUROC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Pneumonia output only | 0.50 | 0.741 | 0.560 | 0.966 | 0.467 | 0.933 | 0.629 |
| Combined pneumonia signs | 0.50 | 0.906 | 0.840 | 0.875 | 0.933 | 0.467 | 0.903 |
| Combined pneumonia signs | 0.56 (best) | 0.906 | 0.827 | 0.980 | 0.800 | 0.933 | 0.881 |

Simple-mode result on these images: of 60 pneumonia, 37 *Pneumonia signs*, 3 *Looks normal*, 20 *Uncertain*; of 15 normal, 6 *Looks normal*, 0 *Pneumonia signs*, 9 *Uncertain*. Inference time about 0.3-0.4 s per image on CPU. Confusion matrix and ROC curve are in `results/`.

**Read honestly:** 75 images (only 15 normal) give wide uncertainty; the best threshold was chosen on the same images so that row is optimistic; many images are portable bedside X-rays from other hospitals, a hard shift from the training data; the pneumonia output alone misses over half of cases, which is why the combined score is used.

**Bigger test (recommended):** download Kaggle's *Chest X-Ray Images (Pneumonia)* (Paul Mooney), then
```bash
python evaluate.py --kaggle_dir "path/to/chest_xray"
```
This writes `results_kaggle/`; the app then uses its thresholds and shows its results automatically. (Those images are children's X-rays, so expect different numbers.) `--nih_dir` gives per-disease metrics for 14 findings on the NIH ChestX-ray14 sample (those images were in the model's training data).

## 6. Patient tips
Shown with every result in Simple mode, in the chosen language: when to see a doctor, urgent warning signs (with the Rwanda ambulance number 912, SAMU), finishing prescribed medicine, rest and fluids, avoiding smoke, protecting others, vaccines. They are general health information, not medical advice, and say so. **Please have a health worker review the Kinyarwanda wording in `i18n.py` and the emergency number before real-world use.**

## 7. Responsible use
| Risk | How it is reduced |
|---|---|
| Used as a diagnosis | Permanent disclaimer; three-way result with "Uncertain"; "normal" never claimed as a guarantee; tips send the patient to a health worker |
| Different hospitals and scanners lower accuracy | Tested on independent images, weaker results published, local validation recommended |
| Noisy public labels | Disclosed; clinician-verified labels needed for clinical work |
| Over-trust in confident output | Heatmap lets a person check where the AI looked |
| Wrong input (not an X-ray) | Input checks warn about colour, low resolution, low contrast |
| Privacy | Runs locally; no image is uploaded; use only de-identified images |

## 8. Credits
- Model: [torchxrayvision](https://github.com/mlmed/torchxrayvision) (Cohen et al.).
- Test images: open chest X-ray image collection by J. P. Cohen et al. (GitHub: ieee8023). Images carry individual licences; use for research and education.
- Optional: Kaggle *Chest X-Ray Images (Pneumonia)* (Kermany et al.) and NIH ChestX-ray14 (Wang et al.).
