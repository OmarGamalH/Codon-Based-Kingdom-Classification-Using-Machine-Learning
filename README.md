# 🧬 Codon Usage-Based Organism Kingdom Classification

A machine learning project that classifies organisms into biological kingdoms using codon usage frequencies from genomic data. Four classifiers — **Random Forest**, **Support Vector Machine (SVM)**, **XGBoost**, and **K-Nearest Neighbors (KNN)** — are compared both before and after **PCA dimensionality reduction**, then combined into a **Soft Voting Ensemble** for a final prediction.

---

## 📌 Problem Statement

Codon usage bias varies significantly across different biological kingdoms (bacteria, viruses, plants, mammals, etc.). This project leverages those differences as features to train machine learning models that can predict which kingdom an organism belongs to, purely from its codon usage statistics.

---

## 🗂️ Dataset

- **Source:** `codon_usage.csv`
- **Target variable:** `Kingdom` — one of 8 classes:
  - `arc` (Archaea), `bct` (Bacteria), `inv` (Invertebrates), `mam` (Mammals), `phg` (Phages), `pln` (Plants), `vrl` (Viruses), `vrt` (Vertebrates)
- **Preprocessing steps:**
  - Dropped non-informative columns: `SpeciesName`, `SpeciesID`, `DNAtype`, `Ncodons`
  - Filtered out organisms with fewer than 1,000 codons (`Ncodons < 1000`)
  - Merged sub-classes `pri` and `rod` into `mam` (Mammals)
  - Removed `plm` class due to very few data points
  - Checked for missing values, duplicates, and garbage values
  - Visualized feature distributions (histograms, box plots) and kingdom-level codon heatmaps

Two datasets were used for modeling:
| Dataset | Description |
|---|---|
| `Pre_PCA.csv` | Original feature set (64 codon frequencies) |
| `pca_results.csv` | Dimensionality-reduced to **top 25 principal components** |

### PCA Details

Features were first scaled with `StandardScaler`, then a scree plot was used to select the optimal number of components. The top 25 PCs were retained, capturing the majority of variance while reducing noise and dimensionality.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=25)
X_pca = pca.fit_transform(X_scaled)
# Top 25 PCs were selected based on scree plot analysis
```

---

## 🧪 Models

### 1. 🌲 Random Forest

**Hyperparameter tuning via GridSearchCV (5-fold CV, scored on macro F1):**

```python
params = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
```

| | Pre-PCA | After PCA |
|---|---|---|
| **Accuracy** | 0.91 | 0.90 |
| **Macro F1** | 0.83 | 0.83 |
| **Weighted F1** | 0.91 | 0.90 |

**Notable observations:**
- `arc` class struggles with recall (0.40 pre-PCA, 0.44 post-PCA) due to low support (25 samples)
- `phg` similarly shows low recall (~0.41–0.45)
- PCA marginally reduces accuracy but preserves macro F1

---

### 2. ⚙️ Support Vector Machine (SVM)

**Hyperparameter tuning via GridSearchCV (5-fold CV, scored on macro F1):**

```python
param_grid = {
    'C': np.linspace(0.1, 1, 5),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
```

| | Pre-PCA | After PCA |
|---|---|---|
| **Accuracy** | 0.92 | 0.95 |
| **Macro F1** | 0.84 | 0.91 |
| **Weighted F1** | 0.92 | 0.95 |

**Notable observations:**
- SVM **significantly benefits from PCA** — accuracy jumps from 0.92 → 0.95
- Post-PCA macro F1 improves from 0.84 → 0.91, indicating better performance on minority classes
- `arc` and `phg` are the hardest classes across both configurations

---

### 3. 🚀 XGBoost

**Two-stage hyperparameter tuning:**

```python
# Stage 1 – tree structure
param_grid_1 = {
    'max_depth': [1, 2, 3],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

# Stage 2 – number of estimators
param_grid_2 = {'n_estimators': [100, 150, 200, 250, 300]}

# Final model includes regularization
model = XGBClassifier(**best_params, reg_lambda=2, reg_alpha=1)
```

> **Note:** XGBoost requires integer-encoded labels. Kingdom labels are mapped to integers (0–7) before training.

| | Pre-PCA | After PCA |
|---|---|---|
| **Accuracy** | 0.93 | 0.90 |
| **Macro F1** | 0.89 | 0.85 |
| **Weighted F1** | 0.93 | 0.90 |

**Notable observations:**
- XGBoost performs best **without PCA** (0.93 accuracy)
- PCA reduces XGBoost performance, suggesting the original features are informative for tree-based methods

---

### 4. 🔵 K-Nearest Neighbors (KNN)

KNN was trained on PCA-reduced features with **SMOTE oversampling** inside an `imblearn` Pipeline to handle class imbalance. Hyperparameter tuning used GridSearchCV over `k` and distance metrics.

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [...],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}
```

> KNN was trained on both the **PCA dataset (top 25 PCs)** and the **full 64-feature dataset**. The PCA version was used in the final ensemble.

---

### 5. 🗳️ Soft Voting Ensemble

The three best PCA-trained models — **Random Forest**, **SVM**, and **KNN** — are combined using **soft voting** (probability averaging). Each model outputs class probabilities, which are averaged to produce the final prediction.

```python
import joblib

rf_model  = joblib.load('Models/RF_model.pkl')
svm_model = joblib.load('Models/SVM.pkl')
knn_model = joblib.load('Models/KNN.pkl')

# Average predicted probabilities
prob_rf  = rf_model.predict_proba(X_test)
prob_svm = svm_model.predict_proba(X_test)
prob_knn = knn_model.predict_proba(X_test.values)

avg_prob = (prob_rf + prob_svm + prob_knn) / 3
final_predictions = rf_model.classes_[np.argmax(avg_prob, axis=1)]
```

| Metric | Score |
|---|---|
| **Accuracy** | **0.94** |
| **Macro F1** | 0.89 |
| **Weighted F1** | 0.94 |

**Notable observations:**
- The ensemble **improves minority class performance** — `arc` recall jumps to 0.80 (vs 0.40–0.44 in individual models) and `phg` recall improves to 0.66
- Combining diverse models smooths out individual weaknesses
- Weighted F1 of 0.94 makes this the **second-best** overall configuration

---

## 📊 Results Summary

| Model | Pre-PCA Accuracy | Post-PCA Accuracy | Best Config |
|---|---|---|---|
| Random Forest | 0.91 | 0.90 | Pre-PCA |
| SVM | 0.92 | **0.95** | Post-PCA ✅ |
| XGBoost | **0.93** | 0.90 | Pre-PCA |
| KNN + SMOTE | — | ~0.92 | Post-PCA |
| **Soft Voting Ensemble** (RF + SVM + KNN) | — | **0.94** | Post-PCA |

**Overall best single model: SVM with PCA (95% accuracy, macro F1 = 0.91)**
**Best for minority classes: Soft Voting Ensemble — `arc` recall 0.80, `phg` recall 0.66**

---

## 📁 Project Structure

```
├── Preprocessing_And_KNN.ipynb              # EDA, preprocessing, PCA, and KNN training
├── Codon_usage.ipynb                        # Additional EDA notebook
├── Random_Forest.ipynb                      # Random Forest training & evaluation
├── SVM.ipynb                                # SVM training & evaluation
├── XGBoost.ipynb                            # XGBoost training & evaluation
├── Ensemble.ipynb                           # Soft voting ensemble of RF + SVM + KNN
│
└── Images/
    ├── Before SVM Classification report.png
    ├── Before SVM Confusion matrix.png
    ├── Ensemble classification report.png
    ├── Ensemble_confusion matrix.png
    ├── Pre XGBoost classification report.png
    ├── Pre_XGBoost Confusion matrix.png
    ├── Random Forest_confusion matrix.png
    ├── Random Forest_confusion matrix PCA.png
    ├── RF_classification report.png
    ├── RF_classification report_PCA.png
    ├── SVM classification report.png
    ├── SVM Confusion matrix.png
    ├── XGBoost classification report.png
    └── XGBoost Confusion Matrix.png
```

---

## ⚙️ Setup & Requirements

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

### Running on Google Colab

All notebooks are designed for **Google Colab** with Google Drive integration. Mount your Drive and update the `PATH`/`DATA_DIR` variable to point to your dataset location:

```python
from google.colab import drive
drive.mount('/content/drive')

PATH = '/content/drive/MyDrive/your_dataset_folder'
```

### Running Locally

Replace the Drive path with your local path and remove the Drive mount cell:

```python
data_path = 'path/to/Pre_PCA.csv'
pca_path  = 'path/to/pca_results.csv'
```

---

## 🔄 Workflow

```
codon_usage.csv
      │
      ▼
Preprocessing_And_KNN.ipynb  ──→  EDA, cleaning, SMOTE, PCA (top 25 PCs)
      │
      ├──→ Pre_PCA.csv         (full 64-feature set)
      └──→ pca_results.csv     (25 PCA components)
              │
      ┌───────┼────────┬───────┐
      ▼       ▼        ▼       ▼
     RF      SVM    XGBoost   KNN
  (.ipynb) (.ipynb) (.ipynb) (in Preprocessing_And_KNN.ipynb)
      │       │        │       │
      └───────┴────────┘       │
     saved as .pkl models ◄────┘
              │
              ▼
       Ensemble.ipynb
    (Soft Voting: RF + SVM + KNN)
              │
      Final Predictions
      Classification Report
      Confusion Matrix
```

---

## 🏷️ Kingdom Label Mapping (XGBoost)

XGBoost requires numeric labels. The following encoding is used:

| Label | Kingdom |
|---|---|
| 0 | arc (Archaea) |
| 1 | bct (Bacteria) |
| 2 | inv (Invertebrates) |
| 3 | mam (Mammals) |
| 4 | phg (Phages) |
| 5 | pln (Plants) |
| 6 | vrl (Viruses) |
| 7 | vrt (Vertebrates) |

---

## 📝 Notes

- The `arc` and `phg` classes consistently have the lowest recall across individual models, due to their small sample sizes (25 and 44 samples respectively). The soft voting ensemble best addresses this, raising `arc` recall to 0.80.
- **SMOTE** (Synthetic Minority Over-sampling Technique) is applied inside the KNN pipeline to handle class imbalance during training only — never on the test set.
- Model files are saved as `.pkl` using `joblib` for reuse in the ensemble notebook.
- All train/test splits use `random_state=42` and `stratify=y` for reproducibility.

---

## 🖼️ Results Gallery

### Random Forest

| Pre-PCA | After PCA |
|---|---|
| ![RF Classification Report](Images/RF_classification%20report.png) | ![RF Classification Report PCA](Images/RF_classification%20report_PCA.png) |
| ![RF Confusion Matrix](Images/Random%20Forest_confusion%20matrix.png) | ![RF Confusion Matrix PCA](Images/Random%20Forest_confusion%20matrix%20PCA.png) |

### SVM

| Pre-PCA | After PCA |
|---|---|
| ![SVM Classification Report (Pre-PCA)](Images/Before%20SVM%20Classification%20report.png) | ![SVM Classification Report](Images/SVM%20classification%20report.png) |
| ![SVM Confusion Matrix (Pre-PCA)](Images/Before%20SVM%20Confusion%20matrix.png) | ![SVM Confusion Matrix](Images/SVM%20Confusion%20matrix.png) |

### XGBoost

| Pre-PCA | After PCA |
|---|---|
| ![XGBoost Classification Report (Pre-PCA)](Images/Pre%20XGBoost%20classification%20report.png) | ![XGBoost Classification Report](Images/XGBoost%20classification%20report.png) |
| ![XGBoost Confusion Matrix (Pre-PCA)](Images/Pre_XGBoost%20Confusion%20matrix.png) | ![XGBoost Confusion Matrix](Images/XGBoost%20Confusion%20Matrix.png) |

### Ensemble (RF + SVM + KNN — Soft Voting)

| Classification Report | Confusion Matrix |
|---|---|
| ![Ensemble Classification Report](Images/Ensemble%20classification%20report.png) | ![Ensemble Confusion Matrix](Images/Ensemble_confusion%20matrix.png) |
