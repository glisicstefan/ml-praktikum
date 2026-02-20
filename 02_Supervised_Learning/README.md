# Supervised Learning - Overview

Supervised learning je vrsta mašinskog učenja gde model uči od **labeled podataka** - svaki ulaz (features) ima odgovarajući izlaz (label). Model pravi predikcije, poredi ih sa pravim vrednostima, i iterativno smanjuje grešku.

**Ključna razlika:**
```
Supervised:   X (features) + y (labels) → Model uči pattern → Predikcija
Unsupervised: X (features) samo        → Model traži strukture
```

---

## Classification vs Regression

| Aspekt | Classification | Regression |
|--------|----------------|------------|
| **Output** | Kategorička vrednost | Kontinuirana numerička vrednost |
| **Primeri** | Spam/Ham, Bolest (Da/Ne), Sentiment | Cena kuće, Temperatura, Prihod |
| **Algoritmi** | Logistic, SVM, Random Forest, XGBoost | Linear, Ridge, Random Forest, XGBoost |
| **Metrike** | Accuracy, Precision, Recall, F1, AUC | MAE, MSE, RMSE, R² |

---

## Algoritmi Poređenje

### Classification

| Algoritam | Brzina | Accuracy | Interpretacija | Scaling | Best For |
|-----------|--------|----------|----------------|---------|----------|
| **Logistic Regression** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Required | Binary classification, baseline |
| **Decision Tree** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ Not needed | Interpretability, EDA |
| **Random Forest** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ Not needed | Structured data, robust baseline |
| **XGBoost/LightGBM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ Not needed | **Best performance**, Kaggle |
| **SVM** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ **MUST** | Small datasets, high-dim, clear margin |
| **KNN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ **MUST** | Small datasets, simple baseline |
| **Naive Bayes** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ Not needed | **Text classification**, real-time |

### Regression

| Algoritam | Interpretacija | Najbolji Za |
|-----------|----------------|-------------|
| **Linear Regression** | ⭐⭐⭐⭐⭐ | Linear relationships, baseline |
| **Ridge/Lasso** | ⭐⭐⭐⭐ | Regularization, feature selection |
| **Random Forest** | ⭐⭐ | Non-linear, robust |
| **XGBoost** | ⭐⭐ | Best performance |
| **SVR** | ⭐⭐ | Small datasets, non-linear |

---

## Decision Framework: Koji Algoritam?
```
START
  │
  ├─> Text classification?
  │     └─ YES → Naive Bayes → Logistic Regression → SVM
  │
  ├─> Dataset size?
  │     ├─ Small (<1k)      → Logistic / SVM / KNN
  │     ├─ Medium (1k-100k) → Random Forest / XGBoost
  │     └─ Large (>100k)    → Logistic / LightGBM
  │
  ├─> Interpretability critical?
  │     └─ YES → Logistic Regression / Decision Tree
  │
  ├─> Need best performance?
  │     └─ YES → XGBoost / LightGBM / Ensemble
  │
  └─> Not sure?
        └─ Start: Logistic (linear) + Random Forest (non-linear)
           Compare and iterate
```

---

## Evaluation Metrike

### Classification

| Metrika | Formula | Kada Koristiti |
|---------|---------|----------------|
| **Accuracy** | (TP+TN) / Total | Balanced classes |
| **Precision** | TP / (TP+FP) | Cost of false positives high (spam) |
| **Recall** | TP / (TP+FN) | Cost of false negatives high (cancer) |
| **F1-Score** | 2 × (P×R) / (P+R) | Imbalanced classes |
| **ROC-AUC** | Area under curve | Overall model quality |

### Regression

| Metrika | Interpretacija | Range |
|---------|----------------|-------|
| **MAE** | Average absolute error | [0, ∞), lower better |
| **RMSE** | Penalizes large errors more | [0, ∞), lower better |
| **R²** | Variance explained | [0, 1], higher better |

---

## Supervised Learning Workflow
```
1. DATA COLLECTION
   └─ Collect labeled data (X, y)

2. DATA PREPROCESSING
   ├─ Handle missing values
   ├─ Encode categorical features
   ├─ Scale features (if needed)
   └─ Train/test split (80/20)

3. MODEL SELECTION
   └─ Choose algorithm based on problem

4. TRAINING
   └─ Fit model on training data

5. EVALUATION
   ├─ Test on unseen data
   └─ Calculate metrics

6. HYPERPARAMETER TUNING
   └─ GridSearch / RandomSearch / Optuna

7. DEPLOYMENT
   └─ Serve model in production
```

---

## Folder Content (02_Supervised_Learning)

**Lekcije:**
1. ✅ **Linear Regression** - Osnova regression-a, linear relationships
2. ✅ **Logistic Regression** - Binary i multiclass classification
3. ✅ **Decision Trees** - Interpretable, overfitting prone
4. ✅ **Random Forest** - Ensemble bagging, robust baseline
5. ✅ **Gradient Boosting** - XGBoost/LightGBM/CatBoost, best performance
6. ✅ **Support Vector Machines** - Kernel trick, clear margins
7. ✅ **K-Nearest Neighbors** - Instance-based, simple
8. ✅ **Naive Bayes** - Probabilistic, excellent for text
9. ✅ **Algorithm Comparison** - Side-by-side benchmarks, decision guides

---

## Quick Selection Guide

**Problem → Algorithm:**

| Your Situation | Recommended Algorithm |
|----------------|----------------------|
| Binary classification, need interpretability | Logistic Regression |
| Structured data, want good performance quickly | Random Forest |
| Kaggle competition, need best accuracy | XGBoost / LightGBM |
| Text classification (spam, sentiment) | Naive Bayes |
| Small dataset (<1k), high-dimensional | SVM |
| Need fast predictions, simple baseline | KNN |
| Understand how model decides | Decision Tree |

---

## Key Concepts

**Overfitting vs Underfitting:**
- **Overfitting:** Model memorizes training data (high train acc, low test acc)
- **Underfitting:** Model too simple (low train acc, low test acc)
- **Solution:** Cross-validation, regularization, more data

**Bias-Variance Tradeoff:**
- **High Bias:** Too simple model (underfitting)
- **High Variance:** Too complex model (overfitting)
- **Goal:** Balance both (sweet spot)

**Feature Scaling:**
- **Required:** Logistic, SVM, KNN, Neural Networks
- **Not needed:** Tree-based (Decision Tree, Random Forest, XGBoost)

**Cross-Validation:**
- Don't trust single train/test split
- Use k-fold CV (k=5 or 10)
- More reliable performance estimate

---

## Pros & Cons

**✅ Prednosti:**
- Jasna struktura i cilj učenja
- Lako merljive performanse (metrike)
- Odlične performanse sa kvalitetnim labeled podacima
- Variety of algorithms za različite probleme

**❌ Mane:**
- Zahteva velike količine **labeled podataka** (skupo, time-consuming)
- Model uči samo ono što vidi u training data
- Ne može generalizovati van distribucije training data
- Labeling errors propagiraju u model

---

## Summary Table

| Aspekt | Classification | Regression |
|--------|----------------|------------|
| **Output Type** | Discrete categories | Continuous values |
| **Best Algorithms** | XGBoost, Random Forest, Logistic | XGBoost, Random Forest, Linear |
| **Baseline** | Logistic Regression | Linear Regression |
| **Key Metric** | F1-Score (imbalanced), Accuracy (balanced) | RMSE, R² |
| **Common Use Cases** | Spam detection, fraud, diagnosis | Price prediction, forecasting |

---

**Key Takeaway:** Supervised learning je **najčešći ML pristup** u praksi. Za većinu problema: počni sa **baseline** (Logistic/Linear), probaj **Random Forest** za non-linear patterns, i upgrade na **XGBoost** za best performance. **Feature engineering** i **hyperparameter tuning** često daju veći boost od switching algoritama! 🎯