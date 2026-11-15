# Real-Time Face Mask Detection Using PCA and Linear SVM

## 1. Project overview

This project implements a classical machine-learning face-mask classification pipeline based on **Principal Component Analysis (PCA)** for dimensionality reduction and a **linear Support Vector Machine (SVM)** for binary classification. OpenCV's Haar cascade detects a face in webcam frames before the cropped face is passed to the classifier.

The work is an **adapted implementation inspired by an IEEE 2023 paper**, not a bit-for-bit reproduction: the project uses a Kaggle dataset and specific preprocessing/hyperparameters that were not fully specified by the paper.

**Project type:** M.Tech AI machine-learning / computer vision experiment.  
**Implementation:** Python, OpenCV, scikit-learn, joblib.  
**Hardware used:** Apple MacBook Air M1 (8 GB RAM).  
**Status:** Trained SVM saved; offline test and preliminary live evaluation completed.

## 2. Objectives

- Build a mask/no-mask classifier without a deep neural network.
- Reduce the dimensionality of flattened face images with PCA.
- Train a linear SVM on the PCA features.
- Perform real-time classification of Haar-detected faces.
- Compare accuracy and latency with the VGG16 model.

## 3. Reference research paper

**Title:** *An Improved Machine Learning Approach to Detect Real Time Face Mask*  
**Venue:** IEEE ICSSIT, 2023  
**DOI:** `10.1109/ICSSIT55814.2023.10060976`

### 3.1 Reported paper method

```text
Webcam / image input
    ↓
Haar cascade / Viola–Jones face detection
    ↓
Face image preparation
    ↓
PCA feature extraction / dimensionality reduction
    ↓
Linear SVM classification
    ↓
Mask / no-mask decision
```

The reference paper described collecting **2,000 webcam images** (1,000 with a mask and 1,000 without), using a **75% training / 25% testing** split, and reporting approximately **98% SVM accuracy**. The paper did not fully specify all implementation details such as image resize dimensions, PCA component count, SVM C value, and random seed. Its results are **paper-reported** results, not this project's measured performance.

### 3.2 How this implementation differs

| Item | IEEE paper | This implementation |
|---|---|---|
| Data source | Webcam-collected images | Kaggle `belsonraja` dataset |
| Images | 2,000 reported | 3,833 in recorded split (3,066 train + 767 test) |
| Train/test split | 75% / 25% | 80% / 20% |
| PCA components | Not clearly specified | 100 |
| Image preparation | Not fully specified | Grayscale, 64×64, flatten, /255, StandardScaler |
| Classifier | Linear SVM | `SVC(kernel="linear", C=1.0)` |
| Reported test accuracy | ~98% (paper) | **73.14%** (local experiment) |

These differences mean that a discrepancy between 98% and 73.14% cannot be attributed solely to the classifier without a controlled reproduction.

## 4. Dataset and labels

The Kaggle dataset was downloaded using:

```python
import kagglehub
root = kagglehub.dataset_download("belsonraja/face-mask-dataset-with-and-without-mask")
```

Directory names: `without_mask`, `with_mask`.  
Final class mapping: **0 = WITHOUT MASK; 1 = MASK**.

The recorded split contained **3,066 training** and **767 test** images, created using an 80/20 stratified split with `random_state=42`.

## 5. Detailed methodology

### 5.1 Face preprocessing

For each image/face crop:

1. Convert color image from BGR to grayscale.
2. Resize to **64 × 64** pixels.
3. Flatten to **4,096 features**.
4. Convert to `float32` and divide by **255**.
5. Apply the previously fitted **StandardScaler**.
6. Apply the previously fitted **PCA** transformation.
7. Predict with the trained **linear SVM**.

```text
Face image
    ↓
Grayscale
    ↓
64 × 64 pixels
    ↓
Flatten to 4096 values
    ↓
Normalize by 255
    ↓
StandardScaler
    ↓
PCA: 100 principal components
    ↓
Linear SVM
    ↓
WITHOUT MASK (0) / MASK (1)
```

### 5.2 StandardScaler

Standardization is applied feature-wise using statistics learned from the **training data**:

`z = (x - mean_training) / standard_deviation_training`

The saved scaler must be reused during testing and webcam inference; fitting it again on webcam samples would change the feature space and invalidate comparisons.

### 5.3 PCA

PCA transforms correlated pixel features into a lower-dimensional representation along directions of greatest variance. In this experiment:

- Original feature count: **4,096**.
- Number of principal components: **100**.
- Reported explained variance retained: **89.40%**.

PCA is unsupervised: it maximizes retained variance, **not** mask-class separation directly. Consequently, some information useful for classification may be lost.

### 5.4 Linear SVM

The classifier was configured as:

```python
from sklearn.svm import SVC
svm = SVC(kernel="linear", C=1.0)
```

A linear SVM learns a separating hyperplane in PCA feature space. The parameter `C=1.0` controls the tradeoff between margin size and training classification penalties. Predictions are mapped to the two stored class labels.

### 5.5 Live inference function

```python
import cv2
import numpy as np

def predict_svm(face, scaler, pca, svm):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    features = gray.flatten().astype(np.float32).reshape(1, -1) / 255.0
    features = scaler.transform(features)
    features = pca.transform(features)
    return int(svm.predict(features)[0])
```

The **same preprocessing order** must be preserved in both training and inference.

## 6. Project files

```text
FaceDetection/
├── PCA_SVM_Face_Mask.ipynb    # SVM training/evaluation notebook
├── models/
│   └── pca_svm_mask.joblib   # Saved scaler, PCA, SVM, image size, classes
├── compare_models.py         # Webcam comparison with VGG16
├── evaluate_live.py          # Automated live sample collection
├── face_mask_model.keras     # VGG16 comparison model
└── results/
    └── live_predictions.csv  # Per-sample results
```

The `.joblib` file stores a dictionary containing at least `scaler`, `pca`, `svm`, `img_size`, and `categories`.

## 7. Dependencies and execution

The SVM training was developed in a Conda environment named `face-mask-svm`; the combined live comparison runs in `Face-x-mask`, which also has TensorFlow installed.

Required Python packages include `opencv-python`, `numpy`, `scikit-learn`, `joblib`, `pandas`, and `kagglehub` for dataset retrieval. The combined comparison also requires TensorFlow/Keras.

```bash
conda activate Face-x-mask
cd ~/FaceDetection
python compare_models.py
```

Automated evaluation:

```bash
python evaluate_live.py
```

The script asks the user to record 100 detected-face samples WITHOUT MASK, then 100 MASK samples. Per-sample results are saved to `results/live_predictions.csv`.

## 8. Experimental results

### 8.1 Offline held-out evaluation

| Metric | PCA + Linear SVM |
|---|---:|
| Training images | 3,066 |
| Test images | 767 |
| Test accuracy | **73.14%** |
| Correct test predictions | 561 |
| PCA components | 100 |
| Explained variance retained | **89.40%** |

**Offline confusion matrix** (rows = actual; columns = predicted):

| Actual / predicted | WITHOUT MASK | MASK |
|---|---:|---:|
| WITHOUT MASK | 277 | 107 |
| MASK | 99 | 284 |

Correct = 277 + 284 = **561**. Incorrect = 107 + 99 = **206**. Accuracy = 561 / 767 = **73.14%**.

### 8.2 Preliminary live evaluation (14 November 2026)

The same 200 detected-face samples were passed to both VGG16 and PCA + SVM: 100 WITHOUT MASK and 100 MASK samples from one webcam session.

| Metric | PCA + Linear SVM |
|---|---:|
| Evaluated detected-face samples | 200 |
| Correct predictions | 90 |
| Live classification accuracy | **45.00%** |
| Average measured prediction time | **1.10 ms/sample** |
| WITHOUT MASK recall | 25.00% |
| MASK recall | 65.00% |
| Macro F1 score | ~0.43 |

**Live confusion matrix** (rows = actual; columns = predicted):

| Actual / predicted | WITHOUT MASK | MASK |
|---|---:|---:|
| WITHOUT MASK | 25 | 75 |
| MASK | 35 | 65 |

The SVM mislabeled **75 of 100 unmasked samples** as MASK. In this session it predicted MASK for 140/200 samples, indicating a tendency toward the MASK label. The live 45% accuracy is lower than the 50% majority-class baseline for this balanced sample.

### 8.3 Speed–accuracy comparison

| Metric | PCA + Linear SVM | VGG16 |
|---|---:|---:|
| Live classification accuracy | 45.00% | **94.00%** |
| Mean measured inference | **1.10 ms** | 136.53 ms |
| Correct out of 200 | 90 | 188 |

By the measured mean inference times, PCA + SVM was approximately **124× faster** than VGG16. This timing is for the **model prediction pipelines only**; camera capture and Haar detection are excluded.

The evaluation also recorded **271 webcam frames without a detected face**, aggregated across both conditions. These frames were not included in classification accuracy.

## 9. Why offline and live results differ

Potential contributing factors include:

- **Domain shift:** Kaggle images may differ from live webcam face crops.
- **Cropping differences:** The classifier was trained on images with potentially different face/background proportions than Haar crops.
- **Grayscale representation:** Color and high-level spatial features are discarded.
- **PCA objective:** Maximum pixel variance is not necessarily maximum class discrimination.
- **Correlated live samples:** Repeated frames of one face can amplify a systematic classification error.
- **Lighting and camera angle:** PCA + Linear SVM can be sensitive to changes in appearance.

These are hypotheses, not experimentally proven causes. The current data do not establish which factor dominates.

## 10. Evaluation limitations

1. The implementation is an adapted reproduction, not an exact implementation of the IEEE paper's data and parameters.
2. The 200 live samples are from one person and one recording session.
3. The script samples only when Haar detects a face; face-detection misses are excluded from accuracy.
4. The live sampling interval does not make successive frames statistically independent.
5. VGG16 and SVM offline results used different, unverified image-level test partitions.
6. The measured 1.10 ms excludes Haar detection and webcam acquisition.
7. The 271 no-face frames are an aggregate count without a standardized denominator.

## 11. Future improvements

- Inspect and save the actual face crops seen by the SVM.
- Verify that training and live preprocessing match exactly.
- Reproduce a shared, image-level offline train/validation/test split for both models.
- Evaluate other values of PCA components and SVM `C` using cross-validation.
- Try alternative representations such as HOG features while keeping the same test protocol.
- Collect multi-person webcam samples across mask types and lighting conditions.
- Measure detector misses and end-to-end system latency separately.
- Report confidence intervals across independent subjects or sessions.

## 12. Research integrity and attribution

**Reference paper:** *An Improved Machine Learning Approach to Detect Real Time Face Mask*, IEEE ICSSIT 2023, DOI `10.1109/ICSSIT55814.2023.10060976`.

**Dataset:** Kaggle, `belsonraja/face-mask-dataset-with-and-without-mask`.

The paper's **98%** is the paper authors' reported result, not this project's accuracy. This implementation obtained **73.14% offline** and **45.00% in its preliminary single-user live test**. Results must be labeled with their evaluation conditions rather than combined into one accuracy claim.

## 13. Reproducibility checklist

- [x] Saved scaler, PCA, and SVM (`models/pca_svm_mask.joblib`)
- [x] Offline test confusion matrix recorded
- [x] Webcam inference works
- [x] 100 no-mask + 100 mask detected-face samples evaluated
- [x] Per-sample CSV saved
- [ ] Record dataset revision/hash and exact train/test image IDs
- [ ] Export full training notebook hyperparameters and environment lockfile
- [ ] Repeat tests across independent people, scenes, and lighting
- [ ] Establish a fair shared offline evaluation protocol
