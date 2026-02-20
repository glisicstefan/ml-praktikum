# Algorithm Comparison

Ovo je **najvažnija lekcija** u Supervised Learning! Ovde ćeš naučiti **kada koristiti koji algoritam** i **kako doneti prave odluke** za tvoj problem.

**Zašto je Algorithm Comparison kritičan?**
- **Ne postoji "best" algoritam** - Zavisi od problema!
- **Slabost jednog = snaga drugog** - Treba znati trade-offs
- **Vreme je novac** - Ne gubiš vreme na pogrešne algoritme
- **Production decisions** - Koji model ide u production?
- **Kaggle competitions** - Koji algoritmi ensemble-ovati?

**Šta ćeš naučiti:**
- ✅ Side-by-side comparison svih algoritama
- ✅ Decision frameworks (flowcharts)
- ✅ Performance benchmarks (actual code)
- ✅ Use case matrix (problem → algorithm)
- ✅ Real-world guidelines

---

## Quick Reference Table

### Classification Algorithms

| Algorithm | Speed (Train) | Speed (Predict) | Interpretability | Handles Non-linear | Scaling Required | Best For |
|-----------|---------------|-----------------|------------------|-------------------|------------------|----------|
| **Logistic Regression** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ✅ Yes | Binary classification, baseline, interpretability |
| **Decision Tree** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ No | Interpretability, EDA, teaching |
| **Random Forest** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ | ❌ No | Structured data, robust baseline |
| **XGBoost/LightGBM** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | ❌ No | **Best performance**, Kaggle, production |
| **SVM** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **MUST** | Small data, high-dim, clear margin |
| **KNN** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ✅ | ✅ **MUST** | Small data, pattern recognition, baseline |
| **Naive Bayes** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ❌ No | **Text classification**, real-time, large scale |

### Regression Algorithms

| Algorithm | Speed (Train) | Speed (Predict) | Interpretability | Handles Non-linear | Scaling Required | Best For |
|-----------|---------------|-----------------|------------------|-------------------|------------------|----------|
| **Linear Regression** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | Recommended | Linear relationships, baseline, interpretability |
| **Decision Tree** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ No | Interpretability, EDA |
| **Random Forest** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ | ❌ No | Structured data, robust |
| **XGBoost/LightGBM** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ | ❌ No | **Best performance** |
| **SVR** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **MUST** | Small data, non-linear |
| **KNN Regressor** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ✅ | ✅ **MUST** | Small data, baseline |

---

## Performance Comparison - Real Benchmark

Hajde da testiramo **SVE algoritme** na **istom dataset-u** i uporedimo performanse!
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ALGORITHM COMPARISON - COMPREHENSIVE BENCHMARK")
print("="*80)

# ==================== 1. LOAD DATA ====================
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"\nDataset: Breast Cancer")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")
print(f"Balance: {np.sum(y==0)}/{np.sum(y==1)} (Malignant/Benign)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (for algorithms that need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 2. DEFINE ALL MODELS ====================
print("\n" + "="*80)
print("MODELS TO COMPARE")
print("="*80)

models_no_scale = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'Naive Bayes': GaussianNB()
}

models_need_scale = {
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("Models without scaling required:")
for name in models_no_scale.keys():
    print(f"  - {name}")

print("\nModels requiring scaling:")
for name in models_need_scale.keys():
    print(f"  - {name}")

# ==================== 3. TRAIN AND EVALUATE ====================
print("\n" + "="*80)
print("TRAINING AND EVALUATION")
print("="*80)

results = []

# Models that don't need scaling
for name, model in models_no_scale.items():
    print(f"\nTraining {name}...")
    
    # Train time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict time
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC (need probabilities)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc else accuracy,
        'CV Mean': cv_mean,
        'CV Std': cv_std,
        'Train Time (s)': train_time,
        'Predict Time (s)': predict_time,
        'Scaling Required': 'No'
    })
    
    print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_mean:.3f} (±{cv_std:.3f})")

# Models that need scaling
for name, model in models_need_scale.items():
    print(f"\nTraining {name}...")
    
    # Train time
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Predict time
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    predict_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': roc_auc if roc_auc else accuracy,
        'CV Mean': cv_mean,
        'CV Std': cv_std,
        'Train Time (s)': train_time,
        'Predict Time (s)': predict_time,
        'Scaling Required': 'Yes'
    })
    
    print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_mean:.3f} (±{cv_std:.3f})")

# ==================== 4. RESULTS SUMMARY ====================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + results_df.to_string(index=False))

# Find best
best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.3f}")
print(f"   F1-Score: {best_model['F1-Score']:.3f}")
print(f"   CV Score: {best_model['CV Mean']:.3f} (±{best_model['CV Std']:.3f})")

# ==================== 5. VISUALIZATIONS ====================
print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

# Accuracy comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy
ax = axes[0, 0]
colors = ['green' if acc > 0.95 else 'orange' if acc > 0.90 else 'red' 
          for acc in results_df['Accuracy']]
ax.barh(results_df['Model'], results_df['Accuracy'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Accuracy')
ax.set_title('Test Accuracy Comparison')
ax.set_xlim([0.85, 1.0])
ax.grid(True, alpha=0.3, axis='x')

# 2. Training time
ax = axes[0, 1]
ax.barh(results_df['Model'], results_df['Train Time (s)'], color='skyblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Time (seconds)')
ax.set_title('Training Time Comparison')
ax.grid(True, alpha=0.3, axis='x')

# 3. Prediction time
ax = axes[1, 0]
ax.barh(results_df['Model'], results_df['Predict Time (s)'], color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Time (seconds)')
ax.set_title('Prediction Time Comparison')
ax.grid(True, alpha=0.3, axis='x')

# 4. CV Score with error bars
ax = axes[1, 1]
ax.barh(results_df['Model'], results_df['CV Mean'], 
        xerr=results_df['CV Std'], color='lightgreen', alpha=0.7, 
        edgecolor='black', capsize=5)
ax.set_xlabel('Cross-Validation Score')
ax.set_title('CV Accuracy (5-Fold) with Std Dev')
ax.set_xlim([0.85, 1.0])
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter: Accuracy vs Speed
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Train Time (s)'], results_df['Accuracy'], 
            s=200, alpha=0.6, edgecolors='black', linewidths=2)

for idx, row in results_df.iterrows():
    plt.annotate(row['Model'], (row['Train Time (s)'], row['Accuracy']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Training Time (seconds)')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs Training Speed\n(Top-right = Fast & Accurate)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Visualizations created!")
```

---

## Decision Framework: Which Algorithm to Use?

### The Ultimate Flowchart
```
START
  │
  ├─> Text classification?
  │     └─ YES → Naive Bayes (MultinomialNB)
  │              └─ Need better? → Logistic Regression → SVM → Deep Learning
  │
  ├─> Dataset size?
  │     ├─ Small (<1k)    → Logistic Regression / SVM / KNN
  │     ├─ Medium (1k-100k) → Random Forest / XGBoost
  │     └─ Large (>100k)   → Logistic Regression / XGBoost / LightGBM
  │
  ├─> Interpretability critical?
  │     └─ YES → Logistic Regression / Decision Tree
  │               └─ Non-linear? → Decision Tree + explain
  │
  ├─> Structured/Tabular data?
  │     └─ YES → Random Forest (baseline) → XGBoost (best performance)
  │
  ├─> High-dimensional? (>50 features)
  │     ├─ Text → Naive Bayes
  │     └─ Numerical → SVM / Logistic Regression
  │
  ├─> Real-time prediction needed?
  │     └─ YES → Logistic Regression / Naive Bayes / KNN
  │
  ├─> Need probabilities?
  │     └─ YES → Logistic Regression / Naive Bayes / Random Forest
  │
  ├─> Kaggle competition?
  │     └─ XGBoost + LightGBM + CatBoost → Ensemble
  │
  └─> Not sure?
        └─ Start: Logistic (linear) + Random Forest (non-linear)
           Compare and iterate
```

---

## Category-Based Comparison

### 1. Linear Models
```python
print("\n" + "="*80)
print("LINEAR MODELS COMPARISON")
print("="*80)

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

# Generate data
X_linear, y_linear = make_classification(n_samples=1000, n_features=20, 
                                          n_informative=15, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

# Scale
scaler_l = StandardScaler()
X_train_l_scaled = scaler_l.fit_transform(X_train_l)
X_test_l_scaled = scaler_l.transform(X_test_l)

linear_models = {
    'Logistic Regression (No Reg)': LogisticRegression(penalty=None, max_iter=1000),
    'Logistic Regression (L2)': LogisticRegression(penalty='l2', C=1.0, max_iter=1000),
    'Logistic Regression (L1)': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
}

print("\n{:30s} {:>10s} {:>10s} {:>15s}".format("Model", "Accuracy", "F1", "Non-Zero Coefs"))
print("-"*70)

for name, model in linear_models.items():
    model.fit(X_train_l_scaled, y_train_l)
    acc = model.score(X_test_l_scaled, y_test_l)
    y_pred = model.predict(X_test_l_scaled)
    f1 = f1_score(y_test_l, y_pred)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
    
    print(f"{name:30s} {acc:10.3f} {f1:10.3f} {n_nonzero:15d}")

print("\n📌 Key Insights:")
print("  • L2 (Ridge): Shrinks all coefficients (none become 0)")
print("  • L1 (Lasso): Feature selection (some coefficients = 0)")
print("  • No Regularization: May overfit with many features")
```

### 2. Tree-Based Models
```python
print("\n" + "="*80)
print("TREE-BASED MODELS COMPARISON")
print("="*80)

from catboost import CatBoostClassifier

tree_models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=0)
}

print("\n{:20s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
    "Model", "Train Acc", "Test Acc", "Train Time", "Overfit Gap"))
print("-"*70)

for name, model in tree_models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    print(f"{name:20s} {train_acc:10.3f} {test_acc:10.3f} {train_time:12.3f}s {gap:11.3f}")

print("\n📌 Key Insights:")
print("  • Decision Tree: Overfits easily (high gap)")
print("  • Random Forest: Reduces overfitting through bagging")
print("  • XGBoost/LightGBM/CatBoost: Best performance, needs tuning")
print("  • LightGBM: Fastest for large datasets")
print("  • CatBoost: Best for categorical features")
```

### 3. Instance-Based vs Parametric
```python
print("\n" + "="*80)
print("INSTANCE-BASED vs PARAMETRIC MODELS")
print("="*80)

instance_models = {
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (K=10)': KNeighborsClassifier(n_neighbors=10),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (Linear)': SVC(kernel='linear')
}

print("\n{:25s} {:>10s} {:>10s} {:>15s} {:>15s}".format(
    "Model", "Accuracy", "Memory", "Train Speed", "Predict Speed"))
print("-"*80)

for name, model in instance_models.items():
    # Train
    start = time.time()
    if 'KNN' in name or 'SVM' in name:
        model.fit(X_train_scaled, y_train)
        X_test_use = X_test_scaled
    else:
        model.fit(X_train, y_train)
        X_test_use = X_test
    train_time = time.time() - start
    
    # Predict
    start = time.time()
    y_pred = model.predict(X_test_use)
    predict_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    
    # Memory (approximate)
    if 'KNN' in name:
        memory = "High (stores all data)"
    else:
        memory = "Low (parameters)"
    
    print(f"{name:25s} {acc:10.3f} {memory:>10s} {train_time:15.3f}s {predict_time:15.4f}s")

print("\n📌 Key Insights:")
print("  • KNN: Instant training, SLOW prediction (must compute distance to all points)")
print("  • Parametric (Logistic/SVM): Learn parameters, fast prediction")
print("  • KNN: High memory (stores entire training set)")
```

---

## Use Case Matrix

### Problem Type → Best Algorithms
```python
use_cases = {
    'Binary Classification (Small Data)': [
        ('Logistic Regression', '⭐⭐⭐⭐⭐', 'Interpretable, fast'),
        ('SVM', '⭐⭐⭐⭐⭐', 'Non-linear boundaries'),
        ('Random Forest', '⭐⭐⭐⭐', 'Robust baseline')
    ],
    
    'Binary Classification (Large Data)': [
        ('Logistic Regression', '⭐⭐⭐⭐⭐', 'Scales well'),
        ('XGBoost/LightGBM', '⭐⭐⭐⭐⭐', 'Best performance'),
        ('Naive Bayes', '⭐⭐⭐⭐', 'Very fast')
    ],
    
    'Multiclass Classification': [
        ('Random Forest', '⭐⭐⭐⭐⭐', 'Handles multiple classes well'),
        ('XGBoost', '⭐⭐⭐⭐⭐', 'Best performance'),
        ('Logistic Regression (OvR)', '⭐⭐⭐⭐', 'Fast, interpretable')
    ],
    
    'Text Classification': [
        ('Naive Bayes', '⭐⭐⭐⭐⭐', 'Fast, excellent baseline'),
        ('Logistic Regression', '⭐⭐⭐⭐⭐', 'Better accuracy'),
        ('SVM', '⭐⭐⭐⭐', 'High-dimensional data')
    ],
    
    'Regression (Linear Relationship)': [
        ('Linear Regression', '⭐⭐⭐⭐⭐', 'Perfect fit'),
        ('Ridge/Lasso', '⭐⭐⭐⭐⭐', 'With regularization'),
        ('XGBoost', '⭐⭐⭐', 'Overkill but works')
    ],
    
    'Regression (Non-linear)': [
        ('Random Forest', '⭐⭐⭐⭐⭐', 'Robust'),
        ('XGBoost', '⭐⭐⭐⭐⭐', 'Best performance'),
        ('SVR', '⭐⭐⭐⭐', 'Small datasets')
    ],
    
    'Imbalanced Data': [
        ('XGBoost (scale_pos_weight)', '⭐⭐⭐⭐⭐', 'Best handling'),
        ('Random Forest (class_weight)', '⭐⭐⭐⭐⭐', 'Easy to use'),
        ('Logistic (class_weight)', '⭐⭐⭐⭐', 'Fast')
    ],
    
    'High-Dimensional Data': [
        ('Logistic Regression (L1)', '⭐⭐⭐⭐⭐', 'Feature selection'),
        ('SVM', '⭐⭐⭐⭐⭐', 'Kernel trick'),
        ('Naive Bayes', '⭐⭐⭐⭐', 'Handles well')
    ],
    
    'Real-Time Predictions': [
        ('Logistic Regression', '⭐⭐⭐⭐⭐', 'Instant'),
        ('Naive Bayes', '⭐⭐⭐⭐⭐', 'Instant'),
        ('Decision Tree', '⭐⭐⭐⭐', 'Fast')
    ],
    
    'Kaggle Competition': [
        ('XGBoost', '⭐⭐⭐⭐⭐', 'Wins most'),
        ('LightGBM', '⭐⭐⭐⭐⭐', 'Fast & accurate'),
        ('Ensemble', '⭐⭐⭐⭐⭐', 'Combine multiple')
    ]
}

print("\n" + "="*80)
print("USE CASE MATRIX: Problem → Best Algorithms")
print("="*80)

for use_case, algorithms in use_cases.items():
    print(f"\n📊 {use_case}:")
    for algo, rating, reason in algorithms:
        print(f"  {rating} {algo:30s} - {reason}")
```

---

## Hyperparameter Importance Rankings

### Which Parameters Matter Most?
```python
hyperparameter_importance = {
    'Logistic Regression': [
        ('C', '⭐⭐⭐⭐⭐', 'Regularization strength'),
        ('penalty', '⭐⭐⭐⭐', 'L1 vs L2'),
        ('class_weight', '⭐⭐⭐⭐', 'For imbalanced data')
    ],
    
    'Random Forest': [
        ('n_estimators', '⭐⭐⭐⭐⭐', '100-500 usually'),
        ('max_depth', '⭐⭐⭐⭐', 'Controls overfitting'),
        ('min_samples_split', '⭐⭐⭐', 'Secondary tuning'),
        ('max_features', '⭐⭐⭐', 'Default sqrt is OK')
    ],
    
    'XGBoost': [
        ('learning_rate', '⭐⭐⭐⭐⭐', '0.01-0.3, critical'),
        ('n_estimators', '⭐⭐⭐⭐⭐', 'With early stopping'),
        ('max_depth', '⭐⭐⭐⭐⭐', '3-10 range'),
        ('subsample', '⭐⭐⭐⭐', 'Row sampling 0.6-1.0'),
        ('colsample_bytree', '⭐⭐⭐⭐', 'Column sampling 0.6-1.0'),
        ('gamma', '⭐⭐⭐', 'Regularization'),
        ('reg_alpha/lambda', '⭐⭐⭐', 'L1/L2 regularization')
    ],
    
    'SVM': [
        ('C', '⭐⭐⭐⭐⭐', 'Regularization'),
        ('gamma', '⭐⭐⭐⭐⭐', 'RBF kernel coefficient'),
        ('kernel', '⭐⭐⭐⭐', 'Linear vs RBF')
    ],
    
    'KNN': [
        ('n_neighbors', '⭐⭐⭐⭐⭐', 'K value, most important'),
        ('weights', '⭐⭐⭐', 'Uniform vs distance'),
        ('metric', '⭐⭐', 'Euclidean usually OK')
    ],
    
    'Decision Tree': [
        ('max_depth', '⭐⭐⭐⭐⭐', 'CRITICAL for overfitting'),
        ('min_samples_split', '⭐⭐⭐⭐', 'Controls splits'),
        ('min_samples_leaf', '⭐⭐⭐', 'Leaf size')
    ]
}

print("\n" + "="*80)
print("HYPERPARAMETER IMPORTANCE RANKINGS")
print("="*80)

for algo, params in hyperparameter_importance.items():
    print(f"\n🔧 {algo}:")
    for param, importance, description in params:
        print(f"  {importance} {param:20s} - {description}")
```

---

## Real-World Guidelines

### Production Considerations
```python
print("\n" + "="*80)
print("PRODUCTION DEPLOYMENT GUIDELINES")
print("="*80)

production_guidelines = {
    'Latency < 1ms (Real-time)': {
        'Best': ['Logistic Regression', 'Naive Bayes'],
        'OK': ['Decision Tree', 'Linear Models'],
        'Avoid': ['KNN', 'SVM (large data)', 'Deep ensembles']
    },
    
    'Latency < 100ms (Interactive)': {
        'Best': ['All tree-based', 'Logistic', 'Naive Bayes'],
        'OK': ['SVM (small data)', 'KNN (small data)'],
        'Avoid': ['KNN (large data)']
    },
    
    'Latency < 1s (Batch)': {
        'Best': ['Any algorithm'],
        'OK': ['Ensemble of multiple models'],
        'Note': 'Focus on accuracy over speed'
    },
    
    'Interpretability Required': {
        'Best': ['Logistic Regression', 'Decision Tree'],
        'OK': ['Linear Regression', 'Naive Bayes'],
        'Avoid': ['XGBoost', 'Random Forest', 'SVM', 'Neural Networks']
    },
    
    'Memory Constrained': {
        'Best': ['Logistic Regression', 'Naive Bayes', 'Decision Tree'],
        'OK': ['SVM'],
        'Avoid': ['KNN', 'Random Forest', 'XGBoost (large)']
    },
    
    'Frequent Retraining': {
        'Best': ['Logistic Regression', 'Naive Bayes', 'Decision Tree'],
        'OK': ['Random Forest'],
        'Avoid': ['XGBoost (slow training)', 'SVM (large data)']
    }
}

for constraint, recommendations in production_guidelines.items():
    print(f"\n📦 {constraint}:")
    for category, algos in recommendations.items():
        if isinstance(algos, list):
            print(f"  {category:15s}: {', '.join(algos)}")
        else:
            print(f"  {category:15s}: {algos}")
```

---

## Common Mistakes When Choosing Algorithms
```python
print("\n" + "="*80)
print("COMMON MISTAKES & HOW TO AVOID THEM")
print("="*80)

mistakes = [
    {
        'mistake': "Using XGBoost for everything",
        'why_bad': "Overkill for simple linear problems, slow to train",
        'better': "Start with Logistic/Linear, upgrade if needed",
        'example': "Linear relationship → Linear Regression is faster & interpretable"
    },
    {
        'mistake': "Not trying simple baselines first",
        'why_bad': "Waste time tuning complex models when simple works",
        'better': "Always: Logistic → Random Forest → XGBoost",
        'example': "Spam detection: Naive Bayes might be sufficient!"
    },
    {
        'mistake': "Using KNN/SVM on large datasets",
        'why_bad': "Prediction is extremely slow",
        'better': "Use tree-based or linear models for large data",
        'example': "1M samples → KNN takes minutes per prediction!"
    },
    {
        'mistake': "Forgetting to scale for SVM/KNN",
        'why_bad': "Features with large values dominate distance",
        'better': "ALWAYS StandardScaler before SVM/KNN",
        'example': "Age (0-100) vs Salary (0-100000) → Salary dominates"
    },
    {
        'mistake': "Using Decision Tree as final model",
        'why_bad': "Overfits easily, unstable",
        'better': "Use Random Forest or XGBoost instead",
        'example': "Single tree: 100% train, 70% test → Overfitting!"
    },
    {
        'mistake': "Ignoring class imbalance",
        'why_bad': "Model predicts majority class always",
        'better': "Use class_weight='balanced' or SMOTE",
        'example': "99% no-fraud → Model predicts all no-fraud = 99% accuracy but useless!"
    },
    {
        'mistake': "Not using regularization",
        'why_bad': "Models overfit with many features",
        'better': "Use L1/L2 for linear, early stopping for boosting",
        'example': "1000 features, 100 samples → Overfitting guaranteed without regularization"
    },
    {
        'mistake': "Choosing algorithm based on name/hype",
        'why_bad': "Different problems need different algorithms",
        'better': "Choose based on: data size, interpretability, speed needs",
        'example': "XGBoost isn't always better than Logistic Regression!"
    }
]

for i, mistake_info in enumerate(mistakes, 1):
    print(f"\n❌ Mistake #{i}: {mistake_info['mistake']}")
    print(f"   Why Bad: {mistake_info['why_bad']}")
    print(f"   ✅ Better: {mistake_info['better']}")
    print(f"   Example: {mistake_info['example']}")
```

---

## Complete Workflow Example
```python
print("\n" + "="*80)
print("COMPLETE ML WORKFLOW: From Problem to Production")
print("="*80)

workflow_code = '''
# ==================== STEP 1: UNDERSTAND PROBLEM ====================
"""
Problem: Predict customer churn (binary classification)
Data: 10,000 customers, 50 features (numerical + categorical)
Goal: 85%+ recall (catch churners), deploy to production
Constraints: Predictions < 100ms, interpretability preferred
"""

# ==================== STEP 2: BASELINE MODELS ====================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Try 2-3 simple models
models_baseline = {
    'Logistic': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_jobs=-1)
}

# Quick evaluation
for name, model in models_baseline.items():
    model.fit(X_train, y_train)
    recall = recall_score(y_test, model.predict(X_test))
    print(f"{name}: Recall = {recall:.3f}")

# Result: Logistic 72%, RF 78% → Need better

# ==================== STEP 3: ADVANCED MODEL ====================
import xgboost as xgb

# XGBoost with class imbalance handling
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
recall_xgb = recall_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost: Recall = {recall_xgb:.3f}")  # 87% ✅

# ==================== STEP 4: THRESHOLD TUNING ====================
# Default threshold 0.5 might not be optimal
y_proba = xgb_model.predict_proba(X_test)[:, 1]

# Find optimal threshold for recall > 85%
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
optimal_threshold = thresholds[np.argmax(recalls >= 0.85)]

y_pred_tuned = (y_proba >= optimal_threshold).astype(int)
recall_tuned = recall_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)

print(f"Tuned threshold {optimal_threshold:.3f}")
print(f"Recall: {recall_tuned:.3f}, Precision: {precision_tuned:.3f}")

# ==================== STEP 5: PRODUCTION READINESS ====================
# Check speed
import time
start = time.time()
_ = xgb_model.predict(X_test[:100])
prediction_time = (time.time() - start) / 100 * 1000  # ms per sample

print(f"Prediction time: {prediction_time:.2f}ms per sample")

if prediction_time < 100:
    print("✅ Meets latency requirement!")
else:
    print("❌ Too slow, consider simpler model")

# ==================== STEP 6: DEPLOY ====================
import joblib
joblib.dump(xgb_model, 'churn_model.pkl')
joblib.dump(optimal_threshold, 'threshold.pkl')

print("✅ Model ready for production!")
'''

print(workflow_code)

print("\n📌 Key Workflow Steps:")
print("  1. Understand problem (classification/regression, size, constraints)")
print("  2. Try simple baselines (Logistic, Random Forest)")
print("  3. If needed, upgrade to XGBoost/LightGBM")
print("  4. Tune threshold for business metric (precision/recall)")
print("  5. Check production requirements (speed, memory)")
print("  6. Deploy with monitoring")
```

---

## Final Recommendations
```python
print("\n" + "="*80)
print("FINAL RECOMMENDATIONS: Algorithm Selection Cheat Sheet")
print("="*80)

recommendations = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ALGORITHM SELECTION GUIDE                          │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 DEFAULT STRATEGY (Works 80% of time):
   1. Start: Logistic Regression (linear baseline)
   2. Upgrade: Random Forest (non-linear baseline)
   3. Final: XGBoost (best performance)
   4. Ensemble: Combine if needed

📊 BY PROBLEM TYPE:

   Binary Classification (Structured):
     → Default: Logistic Regression
     → Best: XGBoost/LightGBM
     → Interpretable: Logistic Regression / Decision Tree

   Multiclass Classification:
     → Default: Random Forest
     → Best: XGBoost/LightGBM
     → Fast: Logistic Regression (OvR)

   Text Classification:
     → Default: Naive Bayes (MultinomialNB)
     → Better: Logistic Regression
     → Best: Fine-tuned Transformer (BERT)

   Regression (Linear):
     → Default: Linear Regression
     → With Regularization: Ridge/Lasso
     → Best: Still Linear Regression

   Regression (Non-linear):
     → Default: Random Forest
     → Best: XGBoost/LightGBM

📏 BY DATA SIZE:

   Small (<1k):
     → Logistic Regression / SVM / KNN

   Medium (1k-100k):
     → Random Forest / XGBoost

   Large (>100k):
     → Logistic Regression / LightGBM / Naive Bayes

   Huge (>1M):
     → Logistic Regression / LightGBM / Online Learning

⚡ BY SPEED REQUIREMENT:

   Real-time (<1ms):
     → Logistic Regression / Naive Bayes / Cached predictions

   Interactive (<100ms):
     → Logistic / Decision Tree / Random Forest

   Batch (seconds OK):
     → XGBoost / Ensemble / Deep Learning

🎓 BY INTERPRETABILITY:

   Must Explain:
     → Logistic Regression / Decision Tree

   Nice to Explain:
     → Random Forest (feature importance)

   Black Box OK:
     → XGBoost / SVM / Neural Networks

💰 BY BUSINESS VALUE:

   High Stakes (medical, financial):
     → Interpretable models + extensive validation
     → Logistic Regression / Decision Tree + SHAP

   Medium Stakes (recommendations):
     → Balance performance & interpretability
     → Random Forest / XGBoost

   Low Stakes (internal tools):
     → Maximize performance
     → XGBoost / Ensemble

🏆 KAGGLE / COMPETITIONS:

   1. XGBoost (multiple with different seeds)
   2. LightGBM (multiple with different seeds)
   3. CatBoost
   4. Neural Networks (if applicable)
   5. Ensemble all above (stacking/blending)

⚠️ REMEMBER:

   • No algorithm is always best
   • Start simple, add complexity only if needed
   • Tune hyperparameters before switching algorithms
   • Cross-validate everything
   • Consider production constraints early
   • Interpretability often beats 1% accuracy gain
"""

print(recommendations)
```

---

## Summary: Quick Lookup Table
```python
print("\n" + "="*80)
print("QUICK LOOKUP: I HAVE [X], WHICH ALGORITHM?")
print("="*80)

quick_lookup = {
    'I have TEXT data': 'Naive Bayes → Logistic Regression',
    'I have SMALL dataset (<1k)': 'Logistic Regression / SVM',
    'I have LARGE dataset (>100k)': 'Logistic Regression / LightGBM',
    'I need INTERPRETABILITY': 'Logistic Regression / Decision Tree',
    'I need BEST PERFORMANCE': 'XGBoost / LightGBM / Ensemble',
    'I need FAST predictions': 'Logistic Regression / Naive Bayes',
    'I have IMBALANCED data': 'XGBoost (scale_pos_weight) / Random Forest (class_weight)',
    'I have HIGH-DIMENSIONAL data': 'Logistic Regression (L1) / SVM / Naive Bayes',
    'I have MANY CATEGORICAL features': 'CatBoost / XGBoost / Random Forest',
    'I have NON-LINEAR relationships': 'Random Forest / XGBoost / SVM (RBF)',
    'I have LINEAR relationships': 'Linear/Logistic Regression',
    'I need PROBABILITIES': 'Logistic Regression / Naive Bayes / Random Forest',
    'I have NO IDEA what I have': 'Start: Logistic + Random Forest, compare',
    'I am doing KAGGLE': 'XGBoost + LightGBM + CatBoost → Ensemble',
    'I am in PRODUCTION': 'Consider: Logistic (speed) or XGBoost (accuracy)',
    'I am LEARNING ML': 'Start: Logistic Regression → Random Forest → XGBoost',
}

for situation, recommendation in quick_lookup.items():
    print(f"\n{'':3s}{situation}")
    print(f"{'':3s}└─> {recommendation}")
```

---

## Zaključak
```python
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)

key_points = """
1️⃣  NO SINGLE "BEST" ALGORITHM
    → Choose based on problem, data, and constraints

2️⃣  ALWAYS START SIMPLE
    → Logistic Regression → Random Forest → XGBoost
    → Don't jump to complex models immediately

3️⃣  SCALING MATTERS
    → SVM & KNN: MUST scale (catastrophic without it)
    → Tree-based: Don't need scaling
    → Linear models: Recommended for interpretation

4️⃣  TREE-BASED = WORKHORSE
    → Random Forest: Great out-of-box performance
    → XGBoost: Best for structured data (wins Kaggle)
    → LightGBM: Fastest for large datasets

5️⃣  TEXT = NAIVE BAYES
    → MultinomialNB: Excellent baseline for text
    → Fast, scalable, hard to beat for spam/sentiment

6️⃣  INTERPRETABILITY COSTS ACCURACY
    → Logistic/Tree: Interpretable but limited
    → XGBoost/Ensemble: Powerful but black box
    → Choose based on business needs

7️⃣  TUNE BEFORE SWITCHING
    → Hyperparameter tuning >> Algorithm switching
    → Well-tuned Random Forest > Default XGBoost

8️⃣  CONSIDER PRODUCTION EARLY
    → Speed requirements
    → Memory constraints
    → Retraining frequency
    → Interpretability needs

9️⃣  ENSEMBLE IF NEEDED
    → Combine strengths of multiple algorithms
    → Stacking/Blending for competitions
    → Diminishing returns in production

🔟  TRUST THE DATA
    → Cross-validate everything
    → Real performance = test set performance
    → Business metrics > Accuracy
"""

print(key_points)

print("\n" + "="*80)
print("🎯 FINAL ADVICE:")
print("="*80)
print("""
For 90% of REAL-WORLD problems:
  
  Step 1: Logistic Regression (fast baseline)
  Step 2: Random Forest (robust baseline)
  Step 3: XGBoost (if you need 2-5% more accuracy)
  Step 4: Tune hyperparameters
  Step 5: If still not enough → Feature engineering > New algorithm

Remember: A well-tuned simple model often beats a poorly-tuned complex one! 🚀
""")

print("="*80)
print("ALGORITHM COMPARISON - COMPLETE! ✅")
print("="*80)
```

---

**Key Takeaway:** Nema "najboljeg" algoritma za sve! **Izbor zavisi od problema, data, i constrainta**. Za **većinu problema**: Logistic Regression (baseline) → Random Forest (robust) → XGBoost (best performance). Za **text**: Naive Bayes. Za **interpretabilnost**: Logistic/Tree. **Start simple, add complexity only when needed!** 🎯💪