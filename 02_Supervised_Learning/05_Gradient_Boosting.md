# Gradient Boosting

Gradient Boosting je **najmoćniji algoritam za structured/tabular data**. Dominira Kaggle competitions i koristi se u production-u svuda!

**Zašto je Gradient Boosting kralj?**
- **Najbolje performanse** na structured data (gotovo uvek top 3)
- **Handles kompleksne veze** - Non-linear, interactions, sve
- **Feature importance** - Vidiš šta utiče na predictions
- **Robustan na outliers** - Bolje od Linear Regression
- **Versatile** - Classification, regression, ranking

**Kada koristiti Gradient Boosting?**
- ✅ Structured/tabular data
- ✅ Želiš najbolje moguće performanse
- ✅ Imaš vreme za hyperparameter tuning
- ✅ Kaggle competitions, production models
- ✅ Non-linear relationships sa complex interactions

**Kada NE koristiti:**
- ❌ Potrebna interpretabilnost (single tree je bolji)
- ❌ Ekstremno mali dataset (<100 samples) → Logistic/Linear
- ❌ Images/Text/Sequences → Neural Networks
- ❌ Potreban instant model (zahteva tuning) → Random Forest

---

## Boosting vs Bagging

### Bagging (Random Forest):
```
Treniraj trees PARALELNO i NEZAVISNO
├─ Tree 1 (bootstrap sample 1)
├─ Tree 2 (bootstrap sample 2)
├─ Tree 3 (bootstrap sample 3)
└─ ...

Final Prediction = AVERAGE/VOTE svih trees

Cilj: Reduce variance (svaki tree je strong, averaging smanjuje greške)
```

### Boosting (Gradient Boosting):
```
Treniraj trees SEKVENCIJALNO, svaki ISPRAVLJA greške prethodnog

Step 1: Tree 1 → Pravi predictions → Izračunaj residuals (greške)
Step 2: Tree 2 → Uči da predvidi residuals od Tree 1
Step 3: Tree 3 → Uči da predvidi residuals od Tree 1 + Tree 2
...
Step N: Tree N → Uči da predvidi residuals svih prethodnih

Final Prediction = Sum svih predictions (weighted)

Cilj: Reduce bias (svaki tree popravlja greške prethodnih)
```

**Key Difference:**
- **Bagging**: Trees rade SVOJ posao nezavisno → combine-ujemo ih
- **Boosting**: Svaki tree radi na **GREŠKAMA** prethodnih trees

---

## Kako Gradient Boosting Radi

### Koncept:
```
Želimo funkciju F(x) koja predviđa y.

Umesto da napravimo JEDAN perfektan model,
pravimo SERIJU slabih modela koji se SABIRU:

F(x) = f₀ + α₁·f₁(x) + α₂·f₂(x) + ... + αₙ·fₙ(x)

gde je:
f₀ = initial prediction (npr. mean za regression)
fᵢ(x) = i-ti tree (mali, shallow tree)
αᵢ = learning rate (koliko "verujemo" i-tom tree-u)
```

### Algoritam (Simplified):
```
1. Initialize: F₀(x) = mean(y)  (za regression)

2. For m = 1 to M:  (M = broj trees)
   
   a) Calculate residuals:
      r = y - F_{m-1}(x)  (koliko je prošli model pogrešio)
   
   b) Fit tree f_m(x) da predvidi residuals r
   
   c) Update model:
      F_m(x) = F_{m-1}(x) + learning_rate × f_m(x)

3. Final prediction: F_M(x)
```

### Primer (Regression):
```
Data: [y_true = [100, 150, 200, 250]]

Step 0: F₀ = mean(y) = 175
        Predictions = [175, 175, 175, 175]
        Residuals = [100-175, 150-175, 200-175, 250-175]
                  = [-75, -25, 25, 75]

Step 1: Tree 1 uči da predvidi residuals [-75, -25, 25, 75]
        Tree 1 predictions: [-70, -20, 20, 70]  (nije perfektno)
        
        Update: F₁(x) = 175 + 0.1 × Tree1(x)  (learning_rate=0.1)
        New predictions = [175-7, 175-2, 175+2, 175+7]
                        = [168, 173, 177, 182]
        New residuals = [100-168, 150-173, 200-177, 250-182]
                      = [-68, -23, 23, 68]

Step 2: Tree 2 uči da predvidi residuals [-68, -23, 23, 68]
        ...
        
Step 100: Nakon 100 trees, residuals su ~0
          Final predictions ≈ [100, 150, 200, 250] ✅
```

**Zašto "Gradient"?**

Matematički, residuals su **negativni gradient** loss funkcije. Tree "ide u smeru" koji **minimizuje loss**.

---

## XGBoost (Extreme Gradient Boosting)

**Najpoznatija i najkorišćenija implementacija!**

### Instalacija:
```bash
pip install xgboost
```

### Python Implementacija:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("Dataset: Breast Cancer")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== XGBOOST CLASSIFIER ====================
print("\n" + "="*60)
print("XGBOOST CLASSIFIER")
print("="*60)

# Train
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'  # Suppress warning
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTrain Accuracy: {train_acc:.3f}")
print(f"Test Accuracy:  {test_acc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, 
                           target_names=cancer.target_names))
```

### Key XGBoost Parameters:
```python
xgb.XGBClassifier(
    # Tree parameters
    n_estimators=100,           # ⭐ Broj trees (50-1000+)
    max_depth=6,                # ⭐ Dubina trees (3-10)
    learning_rate=0.1,          # ⭐ Koliko brzo učimo (0.01-0.3)
    
    # Regularization
    reg_alpha=0,                # L1 regularization
    reg_lambda=1,               # L2 regularization
    gamma=0,                    # Min loss reduction za split
    
    # Stochastic features
    subsample=1.0,              # Row sampling (0.5-1.0)
    colsample_bytree=1.0,       # Column sampling (0.5-1.0)
    colsample_bylevel=1.0,      # Columns per level
    
    # Other
    min_child_weight=1,         # Min sum of weights u child
    scale_pos_weight=1,         # Balance classes (za imbalanced)
    
    random_state=42,
    n_jobs=-1
)
```

---

## Early Stopping (Sprečava Overfitting)

**Problem:** Previše trees → overfitting

**Rešenje:** Zaustavi trening kada validation performance prestane da raste!
```python
from sklearn.model_selection import train_test_split

# Split train → train + validation
X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train sa early stopping
model_es = xgb.XGBClassifier(
    n_estimators=1000,  # Veliki broj (early stopping će zaustaviti pre)
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

model_es.fit(
    X_train_es, y_train_es,
    eval_set=[(X_val_es, y_val_es)],
    early_stopping_rounds=10,  # Zaustavi ako nema improvement za 10 rounds
    verbose=False
)

print(f"\nBest iteration: {model_es.best_iteration}")
print(f"Best score: {model_es.best_score:.3f}")
print(f"Stopped at iteration {model_es.n_estimators} (would've gone to 1000)")

# Test performance
y_test_pred_es = model_es.predict(X_test)
test_acc_es = accuracy_score(y_test, y_test_pred_es)
print(f"Test Accuracy: {test_acc_es:.3f}")
```

---

## Feature Importance (3 Types!)

XGBoost ima **3 načina** da meri feature importance:
```python
# Train model
model_fi = xgb.XGBClassifier(n_estimators=100, random_state=42)
model_fi.fit(X_train, y_train)

# Get importances
importances_weight = model_fi.get_booster().get_score(importance_type='weight')
importances_gain = model_fi.get_booster().get_score(importance_type='gain')
importances_cover = model_fi.get_booster().get_score(importance_type='cover')

print("\nFeature Importance Types:")
print("\n1. WEIGHT (frequency): Koliko puta je feature korišćena")
print(f"   Top feature: {max(importances_weight, key=importances_weight.get)}")

print("\n2. GAIN (importance): Average gain kada je feature korišćena")
print(f"   Top feature: {max(importances_gain, key=importances_gain.get)}")

print("\n3. COVER: Average coverage (# samples) kada je feature korišćena")
print(f"   Top feature: {max(importances_cover, key=importances_cover.get)}")

# Visualize (using gain - most common)
feature_names = cancer.feature_names
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_fi.feature_importances_  # Uses 'gain' by default
}).sort_values('Importance', ascending=False)

print("\n" + importance_df.head(10).to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
top_10 = importance_df.head(10)
plt.barh(top_10['Feature'], top_10['Importance'], 
         color='skyblue', alpha=0.7, edgecolor='black')
plt.xlabel('Importance (Gain)')
plt.title('Top 10 Feature Importances - XGBoost')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# XGBoost built-in plot
xgb.plot_importance(model_fi, max_num_features=10, importance_type='gain')
plt.title('Feature Importance - XGBoost Built-in Plot')
plt.tight_layout()
plt.show()
```

**Preporuka:** Koristi **'gain'** (default) - najinformativniji!

---

## LightGBM (Light Gradient Boosting Machine)

**Brža alternativa XGBoost-u!** Optimizovana za velike datasets.

### Instalacija:
```bash
pip install lightgbm
```

### Ključne Razlike od XGBoost:

| Aspekt | XGBoost | LightGBM |
|--------|---------|----------|
| **Tree Growth** | Level-wise (layer by layer) | Leaf-wise (best leaf) |
| **Speed** | ⭐⭐⭐ Umeren | ⭐⭐⭐⭐⭐ Veoma brz |
| **Memory** | ⭐⭐⭐ Umeren | ⭐⭐⭐⭐⭐ Niska |
| **Categorical Support** | ❌ Ne (mora encode) | ✅ Da (native) |
| **Large Datasets** | ⭐⭐⭐ OK | ⭐⭐⭐⭐⭐ Excellent |
| **Small Datasets** | ⭐⭐⭐⭐ Bolji | ⭐⭐⭐ Može overfitovati |

### Python Implementacija:
```python
import lightgbm as lgb

print("\n" + "="*60)
print("LIGHTGBM CLASSIFIER")
print("="*60)

# Train
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,  # Important: 2^max_depth - 1 je OK
    random_state=42
)
lgb_model.fit(X_train, y_train)

# Predictions
y_test_pred_lgb = lgb_model.predict(X_test)

# Metrics
test_acc_lgb = accuracy_score(y_test, y_test_pred_lgb)
print(f"Test Accuracy: {test_acc_lgb:.3f}")

# Feature importance
importance_lgb = lgb_model.feature_importances_
print(f"\nTop 5 features:")
top_indices = np.argsort(importance_lgb)[-5:][::-1]
for idx in top_indices:
    print(f"  {cancer.feature_names[idx]}: {importance_lgb[idx]:.1f}")
```

### Key LightGBM Parameters:
```python
lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,               # No limit (use num_leaves instead)
    num_leaves=31,              # ⭐ Max broj leaves (2^max_depth)
    min_child_samples=20,       # Min data u leaf
    
    # Regularization
    reg_alpha=0,                # L1
    reg_lambda=0,               # L2
    min_split_gain=0,           # Min gain za split
    
    # Speed optimizations
    subsample=1.0,              # Row sampling
    colsample_bytree=1.0,       # Column sampling
    
    # Categorical
    categorical_feature='auto', # Auto-detect categorical
    
    random_state=42,
    n_jobs=-1
)
```

---

## CatBoost (Categorical Boosting)

**Specijalizovan za categorical features!** Microsoft-ov Yandex algoritam.

### Instalacija:
```bash
pip install catboost
```

### Ključne Prednosti:

- ✅ **Automatic categorical encoding** - Ne mora One-Hot!
- ✅ **Ordered boosting** - Smanjuje overfitting
- ✅ **Symmetric trees** - Brže predictions
- ✅ **Built-in overfitting detection**
- ✅ **Robust defaults** - Malo tuninga potrebno

### Python Implementacija:
```python
from catboost import CatBoostClassifier

print("\n" + "="*60)
print("CATBOOST CLASSIFIER")
print("="*60)

# Train
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    verbose=0  # Suppress output
)
cat_model.fit(X_train, y_train)

# Predictions
y_test_pred_cat = cat_model.predict(X_test)

# Metrics
test_acc_cat = accuracy_score(y_test, y_test_pred_cat)
print(f"Test Accuracy: {test_acc_cat:.3f}")

# Feature importance
importance_cat = cat_model.get_feature_importance()
print(f"\nTop 5 features:")
top_indices_cat = np.argsort(importance_cat)[-5:][::-1]
for idx in top_indices_cat:
    print(f"  {cancer.feature_names[idx]}: {importance_cat[idx]:.1f}")
```

### Categorical Features Example:
```python
# Example with categorical data
data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'LA', 'NYC', 'SF'],      # Categorical
    'education': ['BS', 'MS', 'PhD', 'BS'],  # Categorical
    'salary': [50000, 70000, 90000, 60000],
    'target': [0, 1, 1, 0]
})

# Specify categorical features
cat_features = ['city', 'education']

# CatBoost handles them automatically!
X_cat = data.drop('target', axis=1)
y_cat = data['target']

model_cat_demo = CatBoostClassifier(
    iterations=50,
    cat_features=cat_features,  # Specify categorical columns
    verbose=0
)
model_cat_demo.fit(X_cat, y_cat)

print("\n✅ CatBoost automatically encoded categorical features!")
print("No need for One-Hot Encoding or Label Encoding!")
```

---

## Comparison: XGBoost vs LightGBM vs CatBoost
```python
import time

print("\n" + "="*60)
print("COMPARISON: XGBoost vs LightGBM vs CatBoost")
print("="*60)

models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                  random_state=42, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                    random_state=42),
    'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, 
                                    random_state=42, verbose=0)
}

results = []

for name, model in models.items():
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Train Time (s)': train_time,
        'Predict Time (s)': predict_time
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Train Time: {train_time:.3f}s")
    print(f"  Predict Time: {predict_time:.4f}s")

# Summary
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy
axes[0].bar(results_df['Model'], results_df['Accuracy'], 
            color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Test Accuracy')
axes[0].set_ylim([0.9, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

# Train Time
axes[1].bar(results_df['Model'], results_df['Train Time (s)'], 
            color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Training Time')
axes[1].grid(True, alpha=0.3, axis='y')

# Predict Time
axes[2].bar(results_df['Model'], results_df['Predict Time (s)'], 
            color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
axes[2].set_ylabel('Time (seconds)')
axes[2].set_title('Prediction Time')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

**Tipični Rezultati:**
- **Accuracy**: Svi ~98% (približno jednaki)
- **Train Time**: LightGBM najbrži, CatBoost najsporiji
- **Predict Time**: Svi brzi (~milisekunde)

**Zaključak:**
- **XGBoost**: Solid all-around, najpoznatiji
- **LightGBM**: Najbrži za velike datasets
- **CatBoost**: Najbolji za categorical features

---

## Complete Example: Credit Risk Prediction
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CREDIT RISK PREDICTION - GRADIENT BOOSTING")
print("="*60)

# ==================== 1. GENERATE DATA ====================
X_credit, y_credit = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_classes=2,
    weights=[0.7, 0.3],  # 70% no default, 30% default
    random_state=42
)

print(f"\nDataset: {X_credit.shape}")
print(f"Class distribution:")
print(f"  No Default (0): {np.sum(y_credit == 0)} ({np.sum(y_credit == 0)/len(y_credit)*100:.1f}%)")
print(f"  Default (1):    {np.sum(y_credit == 1)} ({np.sum(y_credit == 1)/len(y_credit)*100:.1f}%)")

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train_cr, X_test_cr, y_train_cr, y_test_cr = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
)

# Validation set for early stopping
X_train_cr2, X_val_cr, y_train_cr2, y_val_cr = train_test_split(
    X_train_cr, y_train_cr, test_size=0.2, random_state=42, stratify=y_train_cr
)

print(f"\nTrain: {X_train_cr2.shape}")
print(f"Val:   {X_val_cr.shape}")
print(f"Test:  {X_test_cr.shape}")

# ==================== 3. BASELINE ====================
print("\n" + "="*60)
print("BASELINE - Random Forest")
print("="*60)

from sklearn.ensemble import RandomForestClassifier

rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_cr, y_train_cr)

from sklearn.metrics import f1_score, roc_auc_score

y_pred_rf = rf_baseline.predict(X_test_cr)
y_proba_rf = rf_baseline.predict_proba(X_test_cr)[:, 1]

rf_acc = accuracy_score(y_test_cr, y_pred_rf)
rf_f1 = f1_score(y_test_cr, y_pred_rf)
rf_auc = roc_auc_score(y_test_cr, y_proba_rf)

print(f"Random Forest:")
print(f"  Accuracy: {rf_acc:.3f}")
print(f"  F1-Score: {rf_f1:.3f}")
print(f"  ROC-AUC:  {rf_auc:.3f}")

# ==================== 4. XGBOOST - DEFAULT ====================
print("\n" + "="*60)
print("XGBOOST - Default Parameters")
print("="*60)

xgb_default = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_default.fit(X_train_cr, y_train_cr)

y_pred_xgb = xgb_default.predict(X_test_cr)
y_proba_xgb = xgb_default.predict_proba(X_test_cr)[:, 1]

xgb_acc = accuracy_score(y_test_cr, y_pred_xgb)
xgb_f1 = f1_score(y_test_cr, y_pred_xgb)
xgb_auc = roc_auc_score(y_test_cr, y_proba_xgb)

print(f"XGBoost (default):")
print(f"  Accuracy: {xgb_acc:.3f}")
print(f"  F1-Score: {xgb_f1:.3f}")
print(f"  ROC-AUC:  {xgb_auc:.3f}")

# ==================== 5. XGBOOST - EARLY STOPPING ====================
print("\n" + "="*60)
print("XGBOOST - Early Stopping")
print("="*60)

xgb_es = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

xgb_es.fit(
    X_train_cr2, y_train_cr2,
    eval_set=[(X_val_cr, y_val_cr)],
    early_stopping_rounds=20,
    verbose=False
)

print(f"Best iteration: {xgb_es.best_iteration}")
print(f"Stopped at: {xgb_es.n_estimators} rounds")

y_pred_xgb_es = xgb_es.predict(X_test_cr)
y_proba_xgb_es = xgb_es.predict_proba(X_test_cr)[:, 1]

xgb_es_acc = accuracy_score(y_test_cr, y_pred_xgb_es)
xgb_es_f1 = f1_score(y_test_cr, y_pred_xgb_es)
xgb_es_auc = roc_auc_score(y_test_cr, y_proba_xgb_es)

print(f"\nXGBoost (early stopping):")
print(f"  Accuracy: {xgb_es_acc:.3f}")
print(f"  F1-Score: {xgb_es_f1:.3f}")
print(f"  ROC-AUC:  {xgb_es_auc:.3f}")

# ==================== 6. HYPERPARAMETER TUNING ====================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - RandomizedSearchCV")
print("="*60)

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 1, 10]
}

xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    param_distributions=param_distributions,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Running RandomizedSearchCV (20 iterations, 3-fold CV)...")
xgb_search.fit(X_train_cr, y_train_cr)

print(f"\nBest parameters:")
for param, value in xgb_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV ROC-AUC: {xgb_search.best_score_:.3f}")

# ==================== 7. FINAL EVALUATION ====================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

best_xgb = xgb_search.best_estimator_
y_pred_best = best_xgb.predict(X_test_cr)
y_proba_best = best_xgb.predict_proba(X_test_cr)[:, 1]

best_acc = accuracy_score(y_test_cr, y_pred_best)
best_f1 = f1_score(y_test_cr, y_pred_best)
best_auc = roc_auc_score(y_test_cr, y_proba_best)

print(f"Best XGBoost (tuned):")
print(f"  Accuracy: {best_acc:.3f}")
print(f"  F1-Score: {best_f1:.3f}")
print(f"  ROC-AUC:  {best_auc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test_cr, y_pred_best,
                           target_names=['No Default', 'Default']))

# Confusion Matrix
cm_cr = confusion_matrix(y_test_cr, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_cr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix - Credit Risk Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== 8. MODEL COMPARISON ====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost (default)', 'XGBoost (early stop)', 'XGBoost (tuned)'],
    'Accuracy': [rf_acc, xgb_acc, xgb_es_acc, best_acc],
    'F1-Score': [rf_f1, xgb_f1, xgb_es_f1, best_f1],
    'ROC-AUC': [rf_auc, xgb_auc, xgb_es_auc, best_auc]
})

print("\n" + comparison.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
colors = ['gray', 'blue', 'orange', 'green']

for idx, metric in enumerate(metrics):
    axes[idx].bar(range(len(comparison)), comparison[metric],
                  color=colors, alpha=0.7, edgecolor='black')
    axes[idx].set_xticks(range(len(comparison)))
    axes[idx].set_xticklabels(comparison['Model'], rotation=15, ha='right')
    axes[idx].set_ylabel(metric)
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].set_ylim([0.8, 1.0])

plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: XGBoost (tuned)")
print(f"   Improvement over Random Forest:")
print(f"     ROC-AUC: +{(best_auc - rf_auc):.3f}")
print(f"     F1: +{(best_f1 - rf_f1):.3f}")

# ==================== 9. FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importances_cr = best_xgb.feature_importances_
feature_names_cr = [f'Feature_{i+1}' for i in range(X_credit.shape[1])]

importance_df_cr = pd.DataFrame({
    'Feature': feature_names_cr,
    'Importance': importances_cr
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features:")
print(importance_df_cr.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_xgb, max_num_features=15, importance_type='gain')
plt.title('Top 15 Feature Importances - XGBoost')
plt.tight_layout()
plt.show()

# ==================== 10. SAVE MODEL ====================
import joblib

joblib.dump(best_xgb, 'xgboost_credit_risk.pkl')
print("\n✅ Model saved: xgboost_credit_risk.pkl")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Best Practices

### ✅ DO:

1. **Start sa Random Forest baseline** - Uporedi XGBoost improvement
2. **Use early stopping** - Sprečava overfitting, štedi vreme
3. **Tune learning_rate i n_estimators zajedno** - Manji LR = više trees
4. **Check feature importance** - Insights i feature selection
5. **Cross-validation** - Za pouzdanu evaluaciju
6. **Scale_pos_weight** za imbalanced - Automatski balansira
7. **Use GPU** ako imaš - 10-50x brže (`tree_method='gpu_hist'`)

### ❌ DON'T:

1. **Ne scale features** - Tree-based ne treba scaling
2. **Ne ignoriši imbalanced data** - `scale_pos_weight` rešava
3. **Ne koristi default parameters za competition** - Tuning je OBAVEZAN
4. **Ne preteruj sa max_depth** - 3-10 je dovoljno (obično 5-7)
5. **Ne zaboravi early stopping** - Štedi vreme i sprečava overfit

---

## Key Hyperparameters Summary

### Najvažniji (Tune OBAVEZNO):
```python
n_estimators      # 100-1000+ (sa early stopping ne brini)
learning_rate     # 0.01-0.3 (manji = sporije ali bolje)
max_depth         # 3-10 (obično 5-7)
```

### Regularization (Anti-overfitting):
```python
gamma             # 0-5 (min loss reduction za split)
reg_alpha         # 0-10 (L1)
reg_lambda        # 0-10 (L2)
min_child_weight  # 1-10 (min sum of weights u child)
```

### Stochastic (Speedup + Regularization):
```python
subsample         # 0.5-1.0 (row sampling)
colsample_bytree  # 0.5-1.0 (column sampling)
```

**Tuning Strategy:**
1. Fix `learning_rate=0.1`, tune `n_estimators` i `max_depth`
2. Tune `subsample` i `colsample_bytree`
3. Tune regularization (`gamma`, `reg_alpha`, `reg_lambda`)
4. Fine-tune `learning_rate` (obično smanji na 0.01-0.05)

**Za detalje, vidi:** `05_Model_Evaluation_and_Tuning/05_Hyperparameter_Tuning.md`

---

## Common Pitfalls

### Greška 1: Default Parameters na Competition
```python
# ❌ LOŠE - Default parameters su OK baseline, ne za final
xgb_bad = xgb.XGBClassifier()
xgb_bad.fit(X_train, y_train)

# ✅ DOBRO - Tune!
xgb_good = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb_good.fit(X_train, y_train)
```

### Greška 2: Previše Trees bez Early Stopping
```python
# ❌ LOŠE - 5000 trees može overfitovati
xgb_bad = xgb.XGBClassifier(n_estimators=5000)
xgb_bad.fit(X_train, y_train)

# ✅ DOBRO - Early stopping
xgb_good = xgb.XGBClassifier(n_estimators=5000)
xgb_good.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50
)
```

### Greška 3: Ignoring Imbalanced Data
```python
# ❌ LOŠE - Ignoriše minority class
xgb_bad = xgb.XGBClassifier()

# ✅ DOBRO
xgb_good = xgb.XGBClassifier(
    scale_pos_weight=len(y[y==0]) / len(y[y==1])  # Ratio majority/minority
)
```

---

## Kada Koristiti Gradient Boosting?

### ✅ Idealno Za:

- **Structured/tabular data** (CSV, databases)
- **Kaggle competitions** (dominira leaderboards)
- **Production models** (bolji od Random Forest)
- **Complex non-linear relationships**
- **Imaš vreme za tuning** (~1-2 sata)
- **Želiš najbolje moguće performanse**

### ❌ Izbegavaj Za:

- **Ekstremno mali datasets** (<100) → Linear/Logistic
- **Potrebna instant interpretabilnost** → Single Tree, Linear
- **Nemaaš vreme za tuning** → Random Forest (bolji out-of-box)
- **Images/Text/Sequences** → Neural Networks
- **Real-time predictions** (latency<1ms) → Linear (brži)

---

## Izbor: XGBoost vs LightGBM vs CatBoost
```
Koji da koristim?

Small-Medium Dataset (<100k rows):
  └─ XGBoost (najstabilniji, najbolji defaults)

Large Dataset (>100k rows):
  └─ LightGBM (najbrži)

Mnogo Categorical Features:
  └─ CatBoost (native categorical support)

Kaggle Competition:
  └─ Probaj sva tri, ensemble-uj!

Production (speed important):
  └─ LightGBM (brži inference)

Production (stability important):
  └─ XGBoost (battle-tested)
```

**Safe Choice:** **XGBoost** - Radi dobro u 90% slučajeva.

---

## Rezime

| Aspekt | Gradient Boosting |
|--------|-------------------|
| **Tip** | Classification & Regression (Ensemble) |
| **Interpretabilnost** | ⭐⭐ Nizak (feature importance OK) |
| **Training Speed** | ⭐⭐ Spor (sequential) |
| **Prediction Speed** | ⭐⭐⭐⭐ Brz |
| **Performance** | ⭐⭐⭐⭐⭐ **Best** za structured data |
| **Handles Non-linearity** | ✅ Excellent |
| **Feature Scaling** | ❌ Ne treba |
| **Overfitting Risk** | ⭐⭐⭐ Umeren (sa early stopping OK) |
| **Hyperparameter Tuning** | ⭐⭐ Dosta potrebno (ali isplati se!) |
| **Best For** | Kaggle, production, najbolje performanse |

---

## Quick Decision Tree
```
Start
  ↓
Structured/tabular data?
  ↓ Yes
Želiš NAJBOLJE moguće performanse?
  ↓ Yes
Imaš vreme za hyperparameter tuning?
  ↓ Yes
→ GRADIENT BOOSTING (XGBoost/LightGBM/CatBoost) ✅

Ako nemasš vreme za tuning:
  └─ Random Forest (brži setup, 95% performanse GB)
```

---

**Key Takeaway:** Gradient Boosting (XGBoost/LightGBM/CatBoost) je **najmoćniji algoritam za structured data**! Dominira Kaggle i production systems. Zahteva **hyperparameter tuning**, ali rezultati su **vredi truda**. Za brzi prototip koristi Random Forest, za finalni model koristi Gradient Boosting! 🏆🎯