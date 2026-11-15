# Real-Time Face Mask Detection Using VGG16 Transfer Learning

## 1. Project overview

This project implements a real-time binary face-mask classifier using **VGG16 transfer learning**, TensorFlow/Keras, and OpenCV. It detects a face in a webcam frame using a Haar cascade, crops the detected region, and predicts **MASK** or **WITHOUT MASK**.

**Project type:** M.Tech AI computer vision project and experimental model comparison.  
**Implementation:** Python, TensorFlow/Keras, OpenCV.  
**Hardware used:** Apple MacBook Air M1 (8 GB RAM).  
**Status:** Trained model and webcam inference tested; preliminary 200-sample live evaluation completed.

> **Scope:** This is a face-mask *classification* system with a separate face-detection stage. It does not independently locate faces using VGG16.

## 2. Objectives

- Build a binary mask/no-mask classifier using transfer learning.
- Connect the classifier to live webcam input.
- Evaluate live classification accuracy and inference latency.
- Compare results with a PCA + Linear SVM baseline on the same captured face samples.

## 3. Dataset

The training data was obtained from Kaggle through `kagglehub`:

```python
import kagglehub
root = kagglehub.dataset_download("belsonraja/face-mask-dataset-with-and-without-mask")
```

Dataset identifier: `belsonraja/face-mask-dataset-with-and-without-mask`.  
Class directory names: `without_mask`, `with_mask`.  
Class mapping in the final model: **0 = WITHOUT MASK; 1 = MASK**.

The exact image count, class distribution, and image-level training/validation/test split for the final VGG16 run should be exported from `CampusX-Mask.ipynb` before claiming them in a report. The final VGG16 notebook reported approximately **92.18% offline test accuracy**, but the exact split was not recorded in this README.

## 4. Architecture and methodology

### 4.1 Training flow

```text
Kaggle images
    ↓
Load images and binary labels
    ↓
Resize images to 224 × 224
    ↓
Normalize image pixels to [0, 1]
    ↓
VGG16 pretrained convolutional feature extractor
    ↓
Task-specific classification head
    ↓
Sigmoid probability / binary decision
    ↓
MASK or WITHOUT MASK
```

VGG16 is a convolutional neural network originally developed for image classification. **Transfer learning** reuses visual features learned by a pretrained model and adapts the final classification layers to the two mask classes. The exact custom classification head, trainable-layer settings, optimizer, epochs, batch size, and learning rate must be verified against `CampusX-Mask.ipynb` rather than assumed.

### 4.2 Live webcam flow

```text
Webcam frame (OpenCV)
    ↓
Convert full frame to grayscale for face detection
    ↓
Haar cascade face detection
    ↓
Select largest detected face and crop its BGR image
    ↓
Resize face to 224 × 224
    ↓
Convert to float32 and divide by 255
    ↓
VGG16 model prediction
    ↓
Threshold at 0.5
    ↓
Display MASK / WITHOUT MASK
```

**Important preprocessing detail:** The current live implementation passes OpenCV **BGR** face crops normalized by `/255.0`, matching the documented project pipeline. Do not silently switch to RGB or `tf.keras.applications.vgg16.preprocess_input` without checking the notebook's training preprocessing, as that could change predictions.

### 4.3 Prediction function

```python
import cv2
import numpy as np

def predict_vgg(face, model):
    image = cv2.resize(face, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = image.reshape(1, 224, 224, 3)
    probability = float(model.predict(image, verbose=0)[0][0])
    return int(probability >= 0.5)  # 0: WITHOUT MASK; 1: MASK
```

## 5. Project files

```text
FaceDetection/
├── CampusX-Mask.ipynb          # VGG16 development/training notebook
├── face_mask_model.keras       # Saved VGG16-based classifier
├── compare_models.py           # Side-by-side webcam inference
├── evaluate_live.py            # Automated 100+100 live sample evaluation
├── models/
│   └── pca_svm_mask.joblib     # Other model used for comparison
└── results/
    └── live_predictions.csv    # Per-sample predictions and timings
```

## 6. Environment and dependencies

The working environment is named `Face-x-mask`. Previously observed packages included TensorFlow 2.21, OpenCV 4.11.0.86, NumPy, scikit-learn, joblib, and pandas. Exact installed versions can be exported with `python -m pip freeze > requirements-lock.txt`.

```bash
conda activate Face-x-mask
cd ~/FaceDetection
python compare_models.py
```

For the automated evaluation:

```bash
python evaluate_live.py
```

The script prompts for two phases: **100 detected-face samples without a mask**, followed by **100 with a mask**. Press Enter at each prompt. The CSV is saved to `results/live_predictions.csv`. Use `x` or Esc in the webcam window to stop early.

## 7. Experimental results

### 7.1 Offline evaluation

- Reported final VGG16 notebook test accuracy: **approximately 92.18%**.
- Exact test split and test-set image identities are not established here; do not compare this directly with the SVM's offline result as though they used the same held-out images.

### 7.2 Preliminary live evaluation (14 November 2026)

Both models were evaluated on the **same 200 detected-face samples** from a single webcam session: 100 labeled WITHOUT MASK and 100 labeled MASK.

| Metric | VGG16 result |
|---|---:|
| Evaluated detected-face samples | 200 |
| Correct predictions | 188 |
| Classification accuracy | **94.00%** |
| Mean measured model prediction time | **136.53 ms/sample** |
| WITHOUT MASK recall | 99.00% |
| MASK recall | 89.00% |
| Macro F1 score | ~0.94 |

**Confusion matrix** (rows = actual class; columns = predicted class):

| Actual / predicted | WITHOUT MASK | MASK |
|---|---:|---:|
| WITHOUT MASK | 99 | 1 |
| MASK | 11 | 89 |

Interpretation: The model correctly identified 99/100 unmasked samples and 89/100 masked samples. Eleven masked samples were incorrectly classified as WITHOUT MASK.

The full evaluation script reported **271 webcam frames with no face detected across both phases**. Those frames were excluded from the 200-sample classification accuracy. This count is **not** a standardized face-detector failure rate, because the script did not log all captured frames and per-class detection totals as a controlled denominator.

### 7.3 Comparison with PCA + Linear SVM

| Metric | VGG16 | PCA + Linear SVM |
|---|---:|---:|
| Live classification accuracy | **94.00%** | 45.00% |
| Average measured inference | 136.53 ms | **1.10 ms** |
| Correct out of 200 | 188 | 90 |

The SVM prediction pipeline was approximately **124× faster** by this measured timing ratio. These timings exclude Haar face detection and camera capture; they should not be described as full end-to-end system latency.

## 8. Evaluation limitations

1. **Single-person test:** All 200 live samples came from one user's webcam session; results cannot establish generalization across people.
2. **Temporal correlation:** Consecutive webcam frames are similar, so 200 samples are not 200 independent subjects or scenes.
3. **Face detection selection:** The test counted only frames in which Haar detected a face.
4. **Unmeasured detector failure by class:** No-face frames were counted in aggregate only.
5. **Potential domain shift:** Training images and cropped webcam faces can differ in framing, lighting, mask types, and backgrounds.
6. **Different offline splits:** The VGG16 and SVM offline accuracy figures were not obtained from a verified shared image-level test partition.
7. **Timing conditions:** Measurements were sequential on one machine, and warmup/variance statistics were not documented.

## 9. Future work

- Use a shared, reproducible train/validation/test split for both models.
- Measure precision, recall, F1, and confusion matrices on a held-out dataset.
- Collect a balanced multi-person, multi-lighting live test set.
- Measure detector recall and full pipeline latency separately.
- Verify and document training preprocessing and transfer-learning hyperparameters.
- Evaluate lightweight deep learning models (for example MobileNetV2) for speed–accuracy tradeoffs.
- Consider deployment to a mobile device after evaluating memory, speed, and reliability.

## 10. Research integrity and attribution

This VGG16 project is **not** a direct reproduction of the IEEE 2023 PCA + SVM method. It is a deep-learning comparison model trained on a Kaggle dataset. The reference paper reported results for **VGG19**, which must not be mislabeled as this project's **VGG16** results.

**Reference paper:** *An Improved Machine Learning Approach to Detect Real Time Face Mask*, IEEE ICSSIT 2023, DOI: `10.1109/ICSSIT55814.2023.10060976`.

**Dataset:** Kaggle, `belsonraja/face-mask-dataset-with-and-without-mask`.

## 11. Reproducibility checklist

- [x] Saved model exists (`face_mask_model.keras`)
- [x] Live webcam inference works
- [x] 100 no-mask and 100 mask detected-face samples evaluated
- [x] Per-sample live predictions saved to CSV
- [ ] Export exact model summary and training hyperparameters
- [ ] Record dataset version, total count, and exact split IDs
- [ ] Perform shared offline test-set comparison
- [ ] Repeat live evaluation across multiple subjects and environments
