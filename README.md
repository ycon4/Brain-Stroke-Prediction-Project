# Brain Stroke Prediction

A machine learning project that predicts whether a person is likely to have a stroke from demographic, health, and lifestyle data. The notebook compares **k-Nearest Neighbors (kNN)**, **Logistic Regression**, and **Linear SVM** (with L1 and L2 regularization), and pays particular attention to the **class imbalance** in the data, using SMOTE and class weighting.

> This is an academic project. It is not a medical tool and must not be used for diagnosis or clinical decisions.

## Dataset

`full_data.csv` contains **4,981 records** with no missing values.

| Feature | Description |
| ------- | ----------- |
| `gender` | Male or Female |
| `age` | Age in years |
| `hypertension` | 1 if the person has hypertension, otherwise 0 |
| `heart_disease` | 1 if the person has heart disease, otherwise 0 |
| `ever_married` | Yes or No |
| `work_type` | Private, Self-employed, Govt_job, or children |
| `Residence_type` | Urban or Rural |
| `avg_glucose_level` | Average glucose level |
| `bmi` | Body mass index |
| `smoking_status` | never smoked, formerly smoked, smokes, or Unknown |
| `stroke` | **Target:** 1 = had a stroke, 0 = no stroke |

**Class imbalance:** only **248 records (about 5%)** are stroke cases, compared with 4,733 non-stroke cases. This is the central challenge of the project.

## Methodology

1. **Exploratory data analysis:** shape, summary statistics, data types, missing-value check, and target distribution
2. **Feature selection:** `ever_married`, `work_type`, and `Residence_type` were dropped because they showed little clear relationship with stroke occurrence
3. **Preprocessing:** one-hot encoding of `gender` and `smoking_status` (first category dropped), and standardization of all features with `StandardScaler`
4. **Handling class imbalance:**
   - **kNN:** SMOTE (Synthetic Minority Over-sampling Technique), since kNN does not support class weights
   - **Logistic Regression and Linear SVM:** balanced class weights
5. **Model search:** each model was evaluated across test sizes of 0.15, 0.20, 0.25, and 0.30 and random states 1 to 20
   - **kNN:** neighbors from 1 to 9
   - **Logistic Regression (L1 and L2):** C from 0.0001 to 100
   - **Linear SVM (L1 and L2):** C from 0.0001 to 100, using `LinearSVC`
6. **Interpretation:** model coefficients were used to identify the top predictor variable for each linear model

## Results

| Model | Train Accuracy | Test Accuracy | Best Parameter | Top Predictor |
| :---: | :---: | :---: | :---: | :---: |
| kNN | 96% | 94% | k = 7 | N/A |
| Logistic Regression (L2) | 73% | 73% | C = 10 | `age` |
| Logistic Regression (L1) | 95% | 95% | C = 0.0001 | `age` |
| Linear SVM (L1) | 95% | 95% | C = 0.0001 | `age` |
| Linear SVM (L2) | 73% | 72% | C = 1 | `age` |

**Key findings**

- `age` was the strongest predictor in every linear model.
- Accuracy is misleading for this dataset. Because about 95% of records are non-stroke, a model that always predicts "no stroke" would already score roughly 95%. The L1 models' 95% therefore says little about their ability to catch actual stroke cases.
- The class-weighted L2 models score lower on accuracy (72 to 73%) because they trade overall accuracy for more attention to the stroke class.
- For a problem like this, recall, precision, F1, and the confusion matrix for the stroke class are more informative than accuracy.

## Repository Contents

```
.
├── final.ipynb       Main notebook: EDA, preprocessing, modeling, and results
├── full_data.csv     Dataset (4,981 records)
└── throw.ipynb       Scratch notebook with earlier experiments (SMOTE and unweighted variants)
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/ycon4/Brain-Stroke-Prediction-Project.git
cd Brain-Stroke-Prediction-Project

pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn jupyter
```

### Running the Notebook

```bash
jupyter notebook final.ipynb
```

Run the cells from top to bottom. Keep `full_data.csv` in the same folder as the notebook, since it is loaded with a relative path.

## Authors

- Andrei Raagas
- Chavy Martin B. Tapuyao
- Luzinda Niña Panong
- Nathaniel Zaragosa
