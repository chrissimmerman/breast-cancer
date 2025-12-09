# Breast Cancer Classification using PCA + Logistic Regression  
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Overview  
This project builds an interpretable machine learning model to classify breast tumors as **benign** or **malignant** using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The workflow includes:

- Feature selection to reduce multicollinearity  
- Dimensionality reduction using **Principal Component Analysis (PCA)**  
- Classification using **Logistic Regression**  
- Visual analysis of PCA, model coefficients, and decision boundaries  

The full analysis is available in both a **Jupyter notebook** and a **written PDF report**.

## Repository Contents  

| File | Description |
|------|-------------|
| **`breast-cancer-prediction.ipynb`** | End-to-end notebook including preprocessing, PCA, logistic regression model training, evaluation, and all plots. |
| **`Project Writeup.pdf`** | Formal writeup summarizing the study: methodology, PCA interpretation, logistic regression model equation, results, and conclusions. |

## Project Motivation  
Breast cancer diagnosis benefits from models that are **accurate yet interpretable**. Many features in the dataset are highly correlated, which can destabilize classical statistical models. This project demonstrates:

- How **feature selection** reduces redundancy  
- How **PCA** creates uncorrelated components while preserving variance  
- How **logistic regression** can provide strong predictive performance with full interpretability  

## Methodology

### 1. Data & Preprocessing
- Dataset: **Breast Cancer Wisconsin (Diagnostic)** — 569 samples, 30 numeric features.
- Removed all **standard error (“SE”) features** due to collinearity with mean features.
- Kept **radius** from the radius / area / perimeter group due to stronger clinical interpretability.

### 2. Principal Component Analysis (PCA)
- PCA applied after feature selection.
- **First 7 components explain ~95% of the variance**.
- Component loadings reveal biological/shape-based interpretations such as:
  - Tumor compactness / concavity  
  - Texture variations  
  - Size-related features  

### 3. Logistic Regression
A logistic regression model was trained on the top 7 PCs.

- Model estimates the probability of tumor benignity via:  
  P(y=1)=1/(1+e^{-z})
- Full linear model (z) is documented in the PDF report.
- Logistic regression chosen for its **interpretability, simplicity, and transparency**.

### 4. Performance Results
The model achieves:

| Metric | Malignant | Benign |
|--------|-----------|--------|
| **Precision** | 0.95 | 0.99 |
| **Recall** | 0.98 | 0.97 |
| **F1-score** | 0.96 | 0.98 |

Overall **accuracy:** **97.2%**

Model visualization includes:
- PCA scatter plot (PC1 vs PC2)
- Decision boundary in PCA space
- Coefficient heatmap
- Confusion matrix showing only **4 total misclassifications**

## How to Run the Notebook

### 1. Clone the repository
```bash
git clone https://github.com/chrissimmerman/breast-cancer.git
cd breast-cancer
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook breast-cancer-prediction.ipynb
```

## Key Visualizations
The notebook generates:

- Scree plot + cumulative variance plot  
- PCA loadings table  
- PC1 vs PC2 scatter plot with true tumor labels  
- Logistic decision boundary in PCA space  
- Coefficient heatmap  
- Confusion matrix  

## Dataset Information
The dataset is the **Breast Cancer Wisconsin (Diagnostic)** dataset, originally from the UCI Machine Learning Repository.  
It includes numeric features extracted from digitized FNA (Fine Needle Aspirate) images.

## Future Work & Extensions
- Add **ROC curve** + compute **AUC**  
- Use **k-fold cross-validation** for more robust evaluation  
- Incorporate **calibration curves**  
- Compare against **SVM**, **Random Forest**, **XGBoost**, etc.  
- Explore **LASSO** or mutual information for feature selection  
- Build an **interactive dashboard** with Plotly or Streamlit  

## License  
MIT License.

## Acknowledgments  
- Dataset creators at the University of Wisconsin & UC Irvine.  
- Classic biostatistics workflows combining PCA and logistic regression for interpretable modeling.
