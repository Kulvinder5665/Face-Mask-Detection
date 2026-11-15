# Real-Time Face Mask Detection — VGG16 vs PCA + Linear SVM

**M.Tech Artificial Intelligence | Computer Vision / Machine Learning project**  
**Platform:** macOS, Apple Silicon (MacBook Air M1)  
**Status:** Both trained models and live webcam comparison working; preliminary evaluation completed on 14 November 2026.

## 1. Project overview

This project implements and compares two approaches to binary face-mask classification:

1. **VGG16 transfer learning:** a convolutional neural network pretrained on ImageNet, adapted for mask classification.
2. **PCA + Linear SVM:** a traditional machine-learning pipeline that converts grayscale face images to vectors, scales them, reduces their dimensionality with principal component analysis, and classifies them with a linear support vector machine.

Both approaches use **OpenCV Haar Cascade** to locate a face in webcam frames before classifying the cropped face. The task has two labels: `0 = WITHOUT MASK` and `1 = MASK`.

The PCA + SVM approach was inspired by *An Improved Machine Learning Approach to Detect Real Time Face Mask* (IEEE ICSSIT 2023, DOI: `10.1109/ICSSIT55814.2023.10060976`). This project is an **adapted implementation**, not an exact reproduction: it uses a Kaggle dataset and project-specific preprocessing choices. The deep-learning comparison here uses **VGG16**, not the paper's VGG19 result.

## 2. Project directory and what every file does

```text
FaceDetection/
├── README.md                       # This combined guide
├── CampusX-Mask.ipynb              # VGG16 training and offline evaluation
├── PCA_SVM_Face_Mask.ipynb         # PCA + Linear SVM training/evaluation
├── compare_models.py              # Live webcam predictions side by side
├── evaluate_live.py               # Automated 100+100 sample evaluation
├── face_mask_model.keras          # Saved VGG16 model (about 512 MB)
├── models/
│   └── pca_svm_mask.joblib         # Saved scaler, PCA, SVM and metadata
├── results/                        # Created by evaluate_live.py
│   └── live_predictions.csv        # One row per sampled detected face
└── Sample/                         # Project sample files (contents may vary)
```

**Note:** `results/` may not appear in a Jupyter file browser until refreshed. The saved models are required for the `.py` webcam scripts; the notebooks are used for training/retraining. The two Python scripts are **not training scripts**.

### File 1 — `CampusX-Mask.ipynb` (VGG16)

**Purpose:** Download/load dataset, prepare images, train a VGG16-based binary classifier, evaluate it, and save the model.

**Confirmed notebook configuration:**

- Kaggle dataset: `belsonraja/face-mask-dataset-with-and-without-mask`.
- Class directories: `without_mask` and `with_mask`.
- VGG16 pretrained ImageNet weights with `include_top=False`, input `(224, 224, 3)`; notebook also contains exploratory VGG16 cells.
- Final dataset split cell: `train_test_split(X, y, test_size=0.2)` **without a fixed random seed**.
- Model compilation: Adam optimizer, binary cross-entropy loss, accuracy metric.
- Full training call: `model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))`.
- Saved model: `model.save("face_mask_model.keras")`.
- Reported offline accuracy from the latest run: **92.18%**.

**Important evaluation limitation:** The same `X_test, y_test` split was used as validation during fitting, so do not present it as a fully untouched independent test set. Because the split has no fixed seed, rerunning may change the result.

**Webcam prediction path:** Haar face detection → crop BGR face → resize to `224×224` → convert to `float32` and divide by `255` → VGG16 prediction → threshold at `0.5` (`>=0.5` means MASK). This matches the working webcam script's inference convention; do not silently switch to another VGG preprocessing function without retraining or verifying consistency.

### File 2 — `PCA_SVM_Face_Mask.ipynb` (PCA + Linear SVM)

**Purpose:** Train a traditional ML baseline on grayscale face-mask images and save its preprocessing and classification stages together.

**Confirmed notebook configuration:**

- Same Kaggle dataset and two label folders.
- Images converted to grayscale, resized to `64×64`, flattened to **4,096** values and scaled by `1/255`.
- Train/test split: `test_size=0.2`, `random_state=42`, `stratify=y`.
- Recorded dataset split: **3,066 training** and **767 testing** images.
- `StandardScaler` fitted on training features; test features transformed with the same scaler.
- `PCA(n_components=100, random_state=42)` fitted on scaled training features; reported variance retained **89.40%**.
- `SVC(kernel="linear", C=1.0)` trained on the 100 PCA features.
- Saved artifact: `models/pca_svm_mask.joblib`, a dictionary containing `scaler`, `pca`, `svm`, `img_size`, `categories`.
- Recorded offline accuracy: **73.14%** (561 correct out of 767).

**Offline SVM confusion matrix** (rows: actual; columns: predicted; class order WITHOUT MASK, MASK):

| Actual / Predicted | WITHOUT MASK | MASK |
|---|---:|---:|
| WITHOUT MASK | 277 | 107 |
| MASK | 99 | 284 |

**Webcam prediction path:** Haar face detection → crop face → BGR to grayscale → resize `64×64` → flatten → divide by `255` → `scaler.transform` → `pca.transform` → `svm.predict`.

### File 3 — `compare_models.py` (interactive webcam comparison)

**Purpose:** Load **both saved models**, detect the largest face visible in the webcam, run both classifiers on the **same cropped face**, and display predictions and individual-frame inference times.

**Outputs on the video:**

- `VGG16: MASK/WITHOUT MASK | ... ms`
- `SVM: MASK/WITHOUT MASK | ... ms`
- A bounding rectangle around the detected face, or `NO FACE DETECTED`.

**Controls:** Press **`x`** or **Esc** while the OpenCV webcam window is active to stop. The Terminal then prints mean inference time for frames on which a face was detected. These times measure each model's preprocessing and classification call, **not** webcam capture or Haar detection.

### File 4 — `evaluate_live.py` (automatic live evaluation)

**Purpose:** Collect **100 detected-face samples without a mask**, followed by **100 detected-face samples with a mask**, run both models on each, and calculate preliminary live performance.

**Exact workflow:**

1. Script loads both saved models and opens the webcam.
2. Terminal displays `Prepare: WITHOUT MASK` and waits for Enter.
3. Remove mask, press Enter in **Terminal**, and keep your face visible. The webcam counter progresses to `100/100` automatically.
4. Terminal displays `Prepare: MASK` and waits for Enter. The webcam may appear frozen while waiting.
5. Put on mask, press Enter in Terminal, and wait for the second `100/100`.
6. Script prints accuracy, average inference time, confusion matrix, classification report, and total frames without a detected face.
7. It saves `results/live_predictions.csv`.

Sampling uses a minimum interval of `0.3` seconds between accepted detected-face samples. A face must be detected for a sample to enter the CSV. Press **`x`** or **Esc** in the OpenCV window to interrupt; partial runs may still save a CSV, so verify the printed sample count.

**CSV columns:** `true_label`, `vgg_pred`, `svm_pred`, `vgg_ms`, `svm_ms`.

## 3. Installation on Mac (Conda)

Open **Mac Terminal** (not a Jupyter notebook cell). The working Conda environment in this project is named `Face-x-mask`.

```bash
conda activate Face-x-mask
cd ~/FaceDetection
pwd
ls
```

`pwd` should print `/Users/<your-username>/FaceDetection`. `ls` should show the four code files and the saved model. The project is in `~/FaceDetection`, **not** `~/Documents/FaceDetection`.

Check installed dependencies:

```bash
python -c "import cv2, tensorflow, sklearn, joblib, numpy, pandas; print('Dependencies OK')"
```

If packages are missing, install them **inside the active environment**:

```bash
python -m pip install tensorflow opencv-python scikit-learn joblib numpy pandas matplotlib kagglehub jupyter
```

The existing working setup used TensorFlow 2.21 and OpenCV 4.11; avoid unnecessarily upgrading OpenCV to a version incompatible with the current script. Webcam permission may need to be granted to Terminal in **System Settings → Privacy & Security → Camera**.

### 3.1. Required libraries and their roles

| Library | Used for |
|---|---|
| `tensorflow` / `keras` | Load and run the trained VGG16 neural network; VGG16 notebook training |
| `opencv-python` (`cv2`) | Webcam capture, Haar Cascade face detection, image resizing and grayscale conversion |
| `numpy` | Image arrays, feature vectors and numerical calculations |
| `scikit-learn` (`sklearn`) | StandardScaler, PCA, linear SVM, train/test splitting and evaluation metrics |
| `joblib` | Load/save the SVM preprocessing pipeline and classifier |
| `pandas` | Save the live evaluation results to CSV and inspect them |
| `matplotlib` | Training/evaluation graphs in notebooks |
| `kagglehub` | Download the face-mask dataset for notebook training |
| `jupyter` / `notebook` | Open and execute the two `.ipynb` training notebooks |

The two webcam scripts use the **`Face-x-mask`** Conda environment. The SVM training notebook was originally developed in a separate **`face-mask-svm`** environment; you do not need to switch to it to run the already-saved models.

### 3.2. Record exact package versions (recommended for reproducibility)

After activating the **working** environment, run these commands in Mac Terminal:

```bash
conda activate Face-x-mask
cd ~/FaceDetection
python -m pip freeze > requirements.txt
conda env export --no-builds > environment.yml
```

- `requirements.txt` records installed Python packages and their versions from the active environment. It is **generated on your Mac**, not included with this README until you run the command.
- `environment.yml` records the Conda environment and its dependencies. Before sharing publicly, inspect it and remove any machine-specific `prefix:` line; note that a macOS/Apple Silicon export may not install unchanged on Windows or Linux.
- These commands **do not install or upgrade** packages, and do not modify your saved models.

Optional: record the separate SVM training environment too:

```bash
conda env export -n face-mask-svm --no-builds > environment-svm.yml
```

Only run that optional command if `face-mask-svm` exists on your computer.

### 3.3. Setup on another compatible Mac

With Conda already installed, prefer the exported environment file **if you have generated it**:

```bash
conda env create -f environment.yml
conda activate Face-x-mask
cd ~/FaceDetection
python -c "import cv2, tensorflow, sklearn, joblib, numpy, pandas; print('Dependencies OK')"
```

If you only have `requirements.txt`, create an environment first and install packages into it:

```bash
conda create -n Face-x-mask python=3.11 -y
conda activate Face-x-mask
python -m pip install -r requirements.txt
```

Python 3.11 here is a **setup example**, not a verified version of the original environment. To record the exact original version, run `python --version` in the existing environment. Platform-specific packages may require adjustments. Ensure `face_mask_model.keras` and `models/pca_svm_mask.joblib` are present; neither dependency file includes model weights or datasets.

## 4. How to run each of the four code files

### A. Run the VGG16 training notebook

```bash
conda activate Face-x-mask
cd ~/FaceDetection
jupyter notebook
```

In the Jupyter browser, open **`CampusX-Mask.ipynb`**. Choose the matching Python kernel and run cells in order. This notebook downloads the Kaggle dataset and trains/saves `face_mask_model.keras`. **Training can take time; do not rerun if you only want to use the existing model.** Rerunning can overwrite the saved model.

### B. Run the PCA + SVM training notebook

Open **`PCA_SVM_Face_Mask.ipynb`** from Jupyter and use a kernel with NumPy, OpenCV, scikit-learn, joblib and kagglehub. Run cells in order to train and save `models/pca_svm_mask.joblib`. The original SVM work used a separate `face-mask-svm` environment, but the existing `.joblib` model can be loaded by the working `Face-x-mask` environment for webcam inference. Retraining can overwrite the saved SVM artifact.

### C. Run the live side-by-side webcam script

In a **separate Mac Terminal**:

```bash
conda activate Face-x-mask
cd ~/FaceDetection
python compare_models.py
```

Face the camera, test with and without a mask, then press `x` or Esc in the webcam window. Review the average timings in Terminal.

### D. Run the automated 200-sample evaluation

```bash
conda activate Face-x-mask
cd ~/FaceDetection
python evaluate_live.py
```

Follow the two Enter prompts described in section 2. When finished, open `~/FaceDetection/results/live_predictions.csv` in a spreadsheet editor or Python.

To inspect the first few rows:

```bash
python -c "import pandas as pd; print(pd.read_csv('results/live_predictions.csv').head())"
```

**Do not run both webcam scripts simultaneously:** They may compete for the same camera.

## 5. Results — 14 November 2026

### Preliminary live experiment

**Test:** 100 accepted detected-face samples per class, captured from one person in one webcam session; both models classified the same 200 accepted face crops.

| Metric | VGG16 | PCA + Linear SVM |
|---|---:|---:|
| Correct predictions | 188 / 200 | 90 / 200 |
| Live accuracy | **94.00%** | **45.00%** |
| Average measured inference | **136.53 ms** | **1.10 ms** |
| Recall: WITHOUT MASK | 99% | 25% |
| Recall: MASK | 89% | 65% |
| Macro F1 (rounded) | 0.94 | 0.43 |

The PCA + SVM pipeline was about **124× faster** by the recorded mean per-sample inference times, but its classification accuracy was substantially lower in this live test. The 45% SVM accuracy is also below the 50% majority-class baseline for this balanced sample.

**VGG16 live confusion matrix:**

| Actual / Predicted | WITHOUT MASK | MASK |
|---|---:|---:|
| WITHOUT MASK | 99 | 1 |
| MASK | 11 | 89 |

**PCA + SVM live confusion matrix:**

| Actual / Predicted | WITHOUT MASK | MASK |
|---|---:|---:|
| WITHOUT MASK | 25 | 75 |
| MASK | 35 | 65 |

**Face detection:** The evaluation script also printed **271 frames with no detected face** during its loops. This is a count of loop iterations with failed face detection, **not a validated failure percentage**: the script did not record the total number of camera frames by class, and it excluded undetected frames from the 200 classification samples.

### Offline notebook results (not a directly fair head-to-head test)

| Metric | VGG16 notebook | PCA + SVM notebook |
|---|---:|---:|
| Reported accuracy | ~92.18% | ~73.14% |
| Train/test split | 80/20, no fixed seed in split cell | 80/20, seed 42, stratified |
| Test-set treatment | Also used as validation during training | Held-out test split |

**Do not claim these offline accuracy values are a controlled comparison** because the models did not use the same fixed image-level test split, and the VGG16 evaluation used the test split for training-time validation.

## 6. How to interpret the results

- **VGG16:** High accuracy on the accepted live face crops, with much higher inference latency.
- **PCA + SVM:** Very low inference latency, but the current live pipeline frequently predicts MASK for unmasked faces (75 out of 100), suggesting weak generalization in this setting.
- **Possible causes of SVM performance loss:** different training-versus-webcam framing, lighting, face crops, grayscale feature limitations, and distribution shift. These are hypotheses, not established causes.
- **Haar Cascade:** Can fail to detect partially occluded or masked faces; classification accuracy alone does not measure full end-to-end reliability.
- **Single-user correlated sampling:** Consecutive webcam samples are not independent. Results should be labeled *preliminary* rather than a general population performance estimate.

## 7. Reproducibility and research limitations

1. Use the **same fixed train/validation/test image IDs** for both approaches before making rigorous offline comparisons.
2. Keep the test set untouched during model selection and training.
3. Record dataset version, exact class counts, training hyperparameters, random seeds, and package versions.
4. Repeat live evaluation with **multiple people, mask types, lighting conditions, camera distances, and sessions**.
5. Track face detection success and failure separately from mask classification accuracy.
6. Measure inference latency after warm-up, and report distribution (median, percentiles), not only the mean.
7. Do not equate the project's VGG16 results with the reference paper's VGG19 results.
8. Record machine and runtime environment when reporting performance.

## 8. Troubleshooting

**`python: can't open file '/Users/.../evaluate_live.py'`:** You are in the wrong directory. Run `cd ~/FaceDetection` first; verify with `pwd` and `ls`.

**`cd ~/Documents/FaceDetection: No such file or directory`:** This project is located at `~/FaceDetection`.

**`No such file or directory` for `.keras` or `.joblib`:** Check that `face_mask_model.keras` and `models/pca_svm_mask.joblib` exist. Both scripts expect those paths relative to their own directory.

**Webcam doesn't open:** Close other camera apps, confirm Terminal camera permission, and check that only one webcam script is running.

**Evaluation appears stuck after 100 without-mask samples:** Look in Terminal for `Prepare: MASK`, put on your mask, and press **Enter** in Terminal.

**`ModuleNotFoundError`:** Activate `Face-x-mask` and install the missing package into that environment.

**OpenCV Haar cascade error:** Check that the installed `cv2` has `CascadeClassifier` and that the Haar XML file exists under `cv2.data.haarcascades`. The project previously worked with OpenCV 4.11.

**Cannot see `results/` in Jupyter:** Refresh the Jupyter file browser after running `evaluate_live.py`.

## 9. Future work

- Make offline evaluation fair using a common split and fully held-out test set.
- Improve robustness of face detection and preprocessing across mask types and lighting.
- Investigate PCA component count, SVM hyperparameters and feature choices with validation data.
- Benchmark smaller deep models (e.g. MobileNetV2) for the accuracy/latency tradeoff.
- Build a multi-person evaluation set and report uncertainty and per-condition metrics.
- Add automated experiment plots and reproducible model-version tracking.

## 10. Reference and acknowledgments

- *An Improved Machine Learning Approach to Detect Real Time Face Mask*, IEEE ICSSIT (2023). DOI: `10.1109/ICSSIT55814.2023.10060976`.
- Dataset: Kaggle, `belsonraja/face-mask-dataset-with-and-without-mask`.
- Software: TensorFlow/Keras, OpenCV, scikit-learn, NumPy, pandas, joblib, Jupyter.

**Academic note:** This is a learning and comparison project. The current results alone do not establish a novel research contribution or reproduce the IEEE paper's reported accuracy.
