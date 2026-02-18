# Handling Imbalanced Data

Imbalanced data je problem gde **jedna klasa ima značajno više primera od druge** klase. Ovo je VEOMA čest problem u realnim aplikacijama i zahteva specijalne tehnike za rukovanje.

**Primeri imbalanced problema:**
- **Fraud detection** - 99.9% legitimne, 0.1% prevare
- **Disease diagnosis** - 95% zdravih, 5% bolesnih
- **Churn prediction** - 90% ostaju, 10% odlazi
- **Spam detection** - 80% legitimni, 20% spam
- **Manufacturing defects** - 99.5% OK, 0.5% defektni

**Zašto je problem?**
- Model uči da **uvek predviđa majority class** → Visoka accuracy, ali beskorisno!
- **Minority class je često važnija** - Koštaju nas false negatives (propuštena prevara, bolest)
- Standardni algoritmi su **biased prema majority class**
- Evaluation metrike poput accuracy **nisu reprezentativne**

**VAŽNO:** Handling imbalance se radi **POSLE** data splitting, ali **PRE** treniranja modela!

---

## Problem sa Imbalanced Data

### Demonstracija Problema:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Imbalanced dataset (95% class 0, 5% class 1)
np.random.seed(42)
X = np.random.randn(1000, 5)
y = np.concatenate([np.zeros(950), np.ones(50)])  # 95/5 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Class distribution:")
print(pd.Series(y_train).value_counts(normalize=True))
# 0.0    0.95
# 1.0    0.05

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")  # Možda 95%!

# ALI pogledaj confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
#     [[285   0]    ← Svi class 0 tačni
#      [ 15   0]]   ← SVI class 1 POGREŠNI!

# Model NIKAD ne predviđa class 1!
print("\nPredictions distribution:")
print(pd.Series(y_pred).value_counts())
# 0.0    300  ← Sve predviđa kao 0!
# (class 1 se ne pojavljuje)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#               precision    recall  f1-score
#          0       0.95      1.00      0.97    ← Odličan
#          1       0.00      0.00      0.00    ← KATASTROFA!
# accuracy                            0.95    ← Obmanjujuće!
```

**Problem:** Model ima 95% accuracy, ALI potpuno IGNORIŠE minority class (class 1)! Za fraud detection, ovo znači da NIJEDNA prevara neće biti detektovana! 🚨

---

## Zašto Accuracy Nije Dobar Metric?

### Dummy Classifier - Baseline:
```python
from sklearn.dummy import DummyClassifier

# "Dummy" model koji UVEK predviđa majority class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

dummy_accuracy = accuracy_score(y_test, dummy_pred)
print(f"Dummy Classifier Accuracy: {dummy_accuracy:.3f}")  # 95%!

# Dummy model ima ISTU accuracy kao "pravi" model!
# Ovo pokazuje da accuracy NE meri pravi kvalitet!
```

### Pravi Metrics za Imbalanced Data:
```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    balanced_accuracy_score
)

# 1. Precision - Od predviđenih pozitivnih, koliko je tačno?
precision = precision_score(y_test, y_pred, zero_division=0)
print(f"Precision: {precision:.3f}")

# 2. Recall (Sensitivity) - Od stvarno pozitivnih, koliko smo uhvatili?
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.3f}")

# 3. F1-Score - Harmonijska sredina precision i recall
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")

# 4. ROC-AUC - Area under ROC curve
# Potrebne su probability predictions
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.3f}")

# 5. PR-AUC - Area under Precision-Recall curve (BOLJI za imbalanced!)
pr_auc = average_precision_score(y_test, y_proba)
print(f"PR-AUC: {pr_auc:.3f}")

# 6. Balanced Accuracy - Average recall za sve klase
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.3f}")
```

**Preporučeni Metrics za Imbalanced:**
- **F1-Score** - Balans precision i recall
- **ROC-AUC** - Overall performance
- **PR-AUC** - NAJBOLJI za extreme imbalance
- **Recall** - Ako su false negatives VEOMA skupi (medical diagnosis)
- **Balanced Accuracy** - Average performance po klasama

---

## Tehnike za Handling Imbalance

---

## 1. Resampling Techniques

**Promena distribucije podataka** - oversampling minority ili undersampling majority.

### A) Random Undersampling (Smanjenje Majority)

**Uklanja random primere iz majority class** dok ne dostigneš željeni ratio.
```python
from imblearn.under_sampling import RandomUnderSampler

# Original distribution
print("Original distribution:")
print(pd.Series(y_train).value_counts())
# 0.0    665
# 1.0     35

# Random undersampling - 1:1 ratio
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print("\nAfter undersampling:")
print(pd.Series(y_train_rus).value_counts())
# 0.0    35  ← Smanjeno na isti broj kao minority!
# 1.0    35

# Train model
model_rus = LogisticRegression()
model_rus.fit(X_train_rus, y_train_rus)

# Evaluate
y_pred_rus = model_rus.predict(X_test)
print("\nClassification Report (Undersampled):")
print(classification_report(y_test, y_pred_rus))

# Sad model VIDI class 1!
```

**Sampling strategy opcije:**
```python
# 1:1 ratio (default)
rus = RandomUnderSampler(sampling_strategy='auto')

# Custom ratio (npr. 2:1)
rus = RandomUnderSampler(sampling_strategy={0: 70, 1: 35})

# Majority class = 2× minority
rus = RandomUnderSampler(sampling_strategy=0.5)  # minority/majority = 0.5
```

**Prednosti:**
- ✅ Brzo i jednostavno
- ✅ Smanjuje trening vreme
- ✅ Balansira classes

**Mane:**
- ❌ **Gubi podatke** - Briše potencijalno korisne primere
- ❌ Može dovesti do underfitting
- ❌ Ne dodaje novu informaciju

**Kada koristiti:**
- Majority class ima MNOGO podataka (>100k)
- Brz test ili baseline

---

### B) Random Oversampling (Povećanje Minority)

**Duplicira random primere iz minority class** dok ne dostigneš željeni ratio.
```python
from imblearn.over_sampling import RandomOverSampler

# Random oversampling - 1:1 ratio
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print("After oversampling:")
print(pd.Series(y_train_ros).value_counts())
# 0.0    665
# 1.0    665  ← Povećano dupliciranjem!

# Train model
model_ros = LogisticRegression()
model_ros.fit(X_train_ros, y_train_ros)

# Evaluate
y_pred_ros = model_ros.predict(X_test)
print("\nClassification Report (Oversampled):")
print(classification_report(y_test, y_pred_ros))
```

**Prednosti:**
- ✅ Ne gubi informacije
- ✅ Koristi sve dostupne podatke
- ✅ Balansira classes

**Mane:**
- ❌ **Overfitting risk** - Duplikati mogu dovesti do memorisanja
- ❌ Povećava trening vreme
- ❌ Ne dodaje novu/sintetičku informaciju

**Kada koristiti:**
- Mali dataset (< 10k samples)
- Jednostavan baseline

---

## 2. SMOTE (Synthetic Minority Over-sampling Technique)

**Kreira SINTETIČKE primere** minority class-e umesto duplikacije - NAJPOZNATIJA tehnika!

### Kako SMOTE Radi?
```
Za svaki minority sample:
1. Nađi K nearest neighbors (iz minority class)
2. Izaberi random neighbor
3. Kreiraj sintetički sample između njih:
   
   new_sample = sample + random(0,1) × (neighbor - sample)
```

### Python Implementacija:
```python
from imblearn.over_sampling import SMOTE

# SMOTE oversampling
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())
# 0.0    665
# 1.0    665  ← Sintetički primeri!

# Train model
model_smote = LogisticRegression()
model_smote.fit(X_train_smote, y_train_smote)

# Evaluate
y_pred_smote = model_smote.predict(X_test)
print("\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote))
```

### Vizualizacija SMOTE:
```python
import matplotlib.pyplot as plt

# 2D example za vizualizaciju
from sklearn.datasets import make_classification

X_toy, y_toy = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1,
    weights=[0.9, 0.1], flip_y=0, random_state=42
)

# Before SMOTE
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_toy[y_toy==0, 0], X_toy[y_toy==0, 1], label='Class 0', alpha=0.6)
plt.scatter(X_toy[y_toy==1, 0], X_toy[y_toy==1, 1], label='Class 1', alpha=0.6)
plt.title('Before SMOTE')
plt.legend()

# After SMOTE
smote_toy = SMOTE(random_state=42)
X_toy_smote, y_toy_smote = smote_toy.fit_resample(X_toy, y_toy)

plt.subplot(1, 2, 2)
plt.scatter(X_toy_smote[y_toy_smote==0, 0], X_toy_smote[y_toy_smote==0, 1], 
            label='Class 0', alpha=0.3)
plt.scatter(X_toy_smote[y_toy_smote==1, 0], X_toy_smote[y_toy_smote==1, 1], 
            label='Class 1 (synthetic)', alpha=0.6)
plt.title('After SMOTE')
plt.legend()

plt.tight_layout()
plt.show()
```

**k_neighbors parametar:**
```python
# k=1 - Duplicira original samples (kao random oversampling)
smote_k1 = SMOTE(k_neighbors=1)

# k=5 - Default (5 najbližih suseda)
smote_k5 = SMOTE(k_neighbors=5)

# k=10 - Više varijabilnosti u sintetičkim primerima
smote_k10 = SMOTE(k_neighbors=10)

# k mora biti <= broj minority samples - 1!
```

**Prednosti:**
- ✅ Kreira nove, različite primere (ne duplicira)
- ✅ Manje overfitting od random oversampling
- ✅ Bolje generalizuje

**Mane:**
- ❌ Može kreirati **noisy samples** između klastera
- ❌ Ne radi dobro sa high-dimensional data
- ❌ Može kreirati outliers

---

### SMOTE Varijante:

#### A) Borderline SMOTE

**Fokusira se na granične primere** - kreira samples blizu decision boundary.
```python
from imblearn.over_sampling import BorderlineSMOTE

# Borderline SMOTE - samo "teški" primeri
bsmote = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train_bsmote, y_train_bsmote = bsmote.fit_resample(X_train, y_train)

# Kreiranje samples SAMO za minority primere blizu granice
```

**Kada koristiti:**
- Decision boundary je nejasna
- Želiš da model fokusira na "teške" primere

#### B) SMOTE + Tomek Links (Cleaning)

**SMOTE + uklanjanje overlap-ujućih samples.**
```python
from imblearn.combine import SMOTETomek

# SMOTE + Tomek Links cleaning
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)

# 1. SMOTE kreira sintetičke samples
# 2. Tomek Links uklanja "problematične" parove (različitih klasa blizu)
```

**Kada koristiti:**
- Želiš čist decision boundary
- Classes imaju overlap

#### C) SMOTE + ENN (Edited Nearest Neighbors)

**SMOTE + agresivnije čišćenje.**
```python
from imblearn.combine import SMOTEENN

# SMOTE + ENN
smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
X_train_se, y_train_se = smote_enn.fit_resample(X_train, y_train)

# 1. SMOTE kreira samples
# 2. ENN uklanja samples čiji najbliži susedi su druge klase
```

#### D) ADASYN (Adaptive Synthetic Sampling)

**Adaptivni SMOTE** - kreira više samples za "teže" minority primere.
```python
from imblearn.over_sampling import ADASYN

# ADASYN - više samples gde je teže klasifikovati
adasyn = ADASYN(sampling_strategy='auto', n_neighbors=5, random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

# Kreira više sintetičkih samples za minority primere 
# koji su okruženi sa više majority samples
```

**Kada koristiti:**
- Dataset ima regione različite težine
- Želiš fokus na "teške" regione

---

## 3. Class Weights

**Penalizuje greške na minority class** - model "plaća" više za pogrešne minority predictions.

### Kako Radi?
```python
# Bez class weights - sve greške jednako koštaju
# Loss = Σ(error)

# Sa class weights - minority greške koštaju više
# Loss = Σ(weight × error)
# weight_minority > weight_majority
```

### Python Implementacija:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# 1. Automatic class weights
model_weighted = LogisticRegression(class_weight='balanced')
model_weighted.fit(X_train, y_train)

# "balanced" automatski računa: n_samples / (n_classes × n_samples_per_class)
# Za 95/5 imbalance: class 0 weight ≈ 0.5, class 1 weight ≈ 9.5

# 2. Manual class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print(f"Class weights: {dict(enumerate(class_weights))}")
# {0: 0.526, 1: 10.0}

# Custom weights
custom_weights = {0: 1, 1: 20}  # 20× više penalizacija za class 1
model_custom = LogisticRegression(class_weight=custom_weights)
model_custom.fit(X_train, y_train)

# Evaluate
y_pred_weighted = model_weighted.predict(X_test)
print("\nClassification Report (Weighted):")
print(classification_report(y_test, y_pred_weighted))
```

### Class Weights za Različite Algoritme:
```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced')

# SVM
from sklearn.svm import SVC
svm = SVC(class_weight='balanced')

# XGBoost
import xgboost as xgb
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)

# LightGBM
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(class_weight='balanced')
```

**Prednosti:**
- ✅ Ne menja podatke
- ✅ Brzo - samo parametar
- ✅ Radi sa originalnim datasetom
- ✅ Ne povećava trening vreme

**Mane:**
- ❌ Ne uvek dovoljno efikasno za extreme imbalance
- ❌ Zahteva tuning weight vrednosti

**Kada koristiti:**
- Preferiš da ne menjati dataset
- Tree-based modeli (RF, XGBoost)
- Kombinuj sa drugim tehnikama

---

## 4. Ensemble Methods

**Kombinovanje više modela** za bolji handling imbalance.

### A) Balanced Random Forest

**Random Forest gde je svaki tree treniran na balanced bootstrap sample.**
```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',  # Balansira svaki bootstrap sample
    replacement=True,
    random_state=42
)

brf.fit(X_train, y_train)

# Evaluate
y_pred_brf = brf.predict(X_test)
print("Classification Report (Balanced RF):")
print(classification_report(y_test, y_pred_brf))
```

### B) Easy Ensemble

**Kreira više balanced subsets undersampling-om, trenira model na svakom.**
```python
from imblearn.ensemble import EasyEnsembleClassifier

# Easy Ensemble - multiple undersampled models
eec = EasyEnsembleClassifier(
    n_estimators=10,  # Broj balanced subsets
    sampling_strategy='auto',
    random_state=42
)

eec.fit(X_train, y_train)

# Evaluate
y_pred_eec = eec.predict(X_test)
print("Classification Report (Easy Ensemble):")
print(classification_report(y_test, y_pred_eec))
```

### C) Balanced Bagging

**Bagging sa balanced bootstrap samples.**
```python
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Balanced Bagging
bbc = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    sampling_strategy='auto',
    random_state=42
)

bbc.fit(X_train, y_train)

y_pred_bbc = bbc.predict(X_test)
print("Classification Report (Balanced Bagging):")
print(classification_report(y_test, y_pred_bbc))
```

---

## 5. Threshold Moving

**Pomeranje decision threshold** - menja se granica predviđanja.

### Kako Radi?
```python
# Default: threshold = 0.5
# prediction = 1 if P(class=1) > 0.5 else 0

# Imbalanced: pomeri threshold NIŽE
# prediction = 1 if P(class=1) > 0.3 else 0
# → Više predictions za minority class!
```

### Python Implementacija:
```python
from sklearn.metrics import precision_recall_curve, roc_curve

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# 1. Default threshold (0.5)
y_pred_default = (y_proba >= 0.5).astype(int)
print("Threshold 0.5:")
print(classification_report(y_test, y_pred_default))

# 2. Lower threshold (0.3)
y_pred_03 = (y_proba >= 0.3).astype(int)
print("\nThreshold 0.3:")
print(classification_report(y_test, y_pred_03))

# 3. Even lower (0.1)
y_pred_01 = (y_proba >= 0.1).astype(int)
print("\nThreshold 0.1:")
print(classification_report(y_test, y_pred_01))
```

### Optimal Threshold - Precision-Recall Curve:
```python
# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# F1-score za svaki threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

# Best threshold (maximum F1)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {best_threshold:.3f}")

# Use optimal threshold
y_pred_optimal = (y_proba >= best_threshold).astype(int)
print("\nOptimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1-Score')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal={best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Tuning')
plt.legend()
plt.grid(True)
plt.show()
```

### Cost-Sensitive Threshold:
```python
# Ako false negative košta više od false positive
# Prilagodi threshold na osnovu cost-a

def find_cost_optimal_threshold(y_true, y_proba, cost_fn, cost_fp):
    """
    Pronalazi optimal threshold na osnovu cost-a.
    
    cost_fn: Cost false negative
    cost_fp: Cost false positive
    """
    thresholds = np.linspace(0, 1, 100)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # tn, fp, fn, tp
        tn, fp, fn, tp = cm.ravel()
        
        # Total cost
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, costs

# Primer: FN košta 10×, FP košta 1×
optimal_threshold, costs = find_cost_optimal_threshold(y_test, y_proba, cost_fn=10, cost_fp=1)
print(f"Cost-optimal threshold: {optimal_threshold:.3f}")

# Visualize costs
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 1, 100), costs)
plt.axvline(optimal_threshold, color='red', linestyle='--')
plt.xlabel('Threshold')
plt.ylabel('Total Cost')
plt.title('Cost-Sensitive Threshold')
plt.show()
```

**Prednosti:**
- ✅ Ne menja trening data
- ✅ Lako prilagoditi
- ✅ Cost-sensitive

**Mane:**
- ❌ Zahteva probability predictions
- ❌ Threshold mora se tunovati

---

## 6. Anomaly Detection Approach

**Za EXTREME imbalance** (0.1% minority) - tretiraj minority kao anomalije.

### One-Class SVM:
```python
from sklearn.svm import OneClassSVM

# Treniraj SAMO na majority class
X_train_majority = X_train[y_train == 0]

# One-Class SVM
ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')  # nu ≈ expected outlier rate
ocsvm.fit(X_train_majority)

# Predict: -1 = outlier (minority), 1 = normal (majority)
y_pred_ocsvm = ocsvm.predict(X_test)
y_pred_ocsvm = (y_pred_ocsvm == -1).astype(int)  # Convert to 0/1

print("Classification Report (One-Class SVM):")
print(classification_report(y_test, y_pred_ocsvm))
```

### Isolation Forest:
```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
iforest = IsolationForest(
    contamination=0.05,  # Expected proportion minority
    random_state=42
)

# Fit na SVIM podacima (može i samo majority)
iforest.fit(X_train)

# Predict
y_pred_if = iforest.predict(X_test)
y_pred_if = (y_pred_if == -1).astype(int)

print("Classification Report (Isolation Forest):")
print(classification_report(y_test, y_pred_if))
```

**Kada koristiti:**
- **Extreme imbalance** (< 1% minority)
- **Fraud detection, outlier detection**
- Minority class je "anomalous"

---

## Kombinovane Strategije

**Najbolji rezultati:** Kombinuj više tehnika!

### Strategy 1: SMOTE + Class Weights
```python
# 1. SMOTE - balance data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Class weights - dodatna penalizacija
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_smote, y_train_smote)
```

### Strategy 2: Undersampling + Ensemble
```python
# 1. Create multiple undersampled subsets
from imblearn.under_sampling import RandomUnderSampler

n_subsets = 5
models = []

for i in range(n_subsets):
    rus = RandomUnderSampler(random_state=i)
    X_subset, y_subset = rus.fit_resample(X_train, y_train)
    
    model = LogisticRegression()
    model.fit(X_subset, y_subset)
    models.append(model)

# 2. Ensemble voting
from scipy.stats import mode

def ensemble_predict(X, models):
    predictions = np.array([model.predict(X) for model in models])
    # Majority vote
    ensemble_pred, _ = mode(predictions, axis=0, keepdims=False)
    return ensemble_pred

y_pred_ensemble = ensemble_predict(X_test, models)
print("Classification Report (Ensemble Undersampling):")
print(classification_report(y_test, y_pred_ensemble))
```

### Strategy 3: SMOTE + Threshold Tuning
```python
# 1. SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Train model
model = LogisticRegression()
model.fit(X_train_smote, y_train_smote)

# 3. Find optimal threshold
y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_threshold = thresholds[np.argmax(f1_scores)]

# 4. Predict sa optimal threshold
y_pred_final = (y_proba >= best_threshold).astype(int)
print("Classification Report (SMOTE + Threshold):")
print(classification_report(y_test, y_pred_final))
```

---

## Evaluation Strategies

### Confusion Matrix Analysis:
```python
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Breakdown
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Rates
print(f"\nTrue Positive Rate (Recall): {tp/(tp+fn):.3f}")
print(f"False Positive Rate: {fp/(fp+tn):.3f}")
print(f"True Negative Rate (Specificity): {tn/(tn+fp):.3f}")
```

### ROC Curve & PR Curve:
```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
axes[0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True)

# PR Curve (BOLJI za imbalanced!)
axes[1].plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
baseline = len(y_test[y_test==1]) / len(y_test)
axes[1].axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

**Za imbalanced data:**
- **PR Curve > ROC Curve** - PR Curve je informativnija!
- **PR-AUC** metric je bolji od ROC-AUC

---

## Decision Framework - Koja Tehnika?
```
Koliko je imbalance ratio?
│
├─→ Mild (60/40 - 70/30)
│   └─→ Class weights često dovoljno
│
├─→ Moderate (80/20 - 90/10)
│   ├─→ SMOTE + Class weights
│   └─→ ILI Random Oversampling
│
├─→ High (95/5 - 99/1)
│   ├─→ SMOTE/ADASYN + Ensemble
│   ├─→ Balanced Random Forest
│   └─→ ILI Easy Ensemble
│
└─→ Extreme (>99.5/0.5)
    ├─→ Anomaly detection (Isolation Forest, One-Class SVM)
    ├─→ Cost-sensitive threshold tuning
    └─→ Multiple undersampling + Ensemble

Koji algoritam koristiš?
│
├─→ Tree-based (RF, XGBoost) → Class weights + SMOTE
├─→ Linear (Logistic) → SMOTE + threshold tuning
├─→ SVM → Class weights
└─→ Neural Networks → Class weights + oversampling

Koliko podataka imaš?
│
├─→ Mali (< 1k) → Oversampling (SMOTE, ADASYN)
├─→ Srednji (1k-100k) → SMOTE + Class weights
└─→ Veliki (> 100k) → Undersampling + Ensemble ili Class weights
```

---

## Best Practices

### ✅ DO:

**1. Split PRE Resampling (KRITIČNO!)**
```python
# ✅ DOBRO - Resample SAMO train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SMOTE samo na train
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Test ostaje original!
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)  # Test NIJE resample-ovan!
```

**2. Stratified Split**
```python
# Održi proporcije u train i test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y  # ✅ Important!
)
```

**3. Use Proper Metrics**
```python
# ❌ LOŠE - Samo accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# ✅ DOBRO - Multiple metrics
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba)}")
```

**4. Cross-Validation sa Stratification**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"F1-Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

**5. Pipeline za Resampling**
```python
from imblearn.pipeline import Pipeline as ImbPipeline

# Pipeline sa resampling
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Cross-validation - resample se dešava UNUTAR svakog fold-a!
scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')
```

### ❌ DON'T:

**1. Ne Resample PRE Split-a (Data Leakage!)**
```python
# ❌ LOŠE - NIKAD ovo!
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)  # Na SVIM podacima!
X_train, X_test = train_test_split(X_resampled, y_resampled)

# Problem: Sintetički samples u train su kreirani iz test podataka!
```

**2. Ne Resample Test Set**
```python
# ❌ LOŠE
smote_train = SMOTE()
X_train_smote, y_train_smote = smote_train.fit_resample(X_train, y_train)

smote_test = SMOTE()
X_test_smote, y_test_smote = smote_test.fit_resample(X_test, y_test)  # NIKAD!

# Test MORA ostati original za validnu evaluaciju!
```

**3. Ne Koristi Samo Accuracy**
```python
# ❌ LOŠE
if accuracy > 0.95:
    print("Great model!")  # Može biti beskorisno za imbalanced!

# ✅ DOBRO
if f1_score > 0.7 and recall > 0.6:
    print("Good model for imbalanced data!")
```

**4. Ne Zaboravi Evaluate na Original Distribution**
```python
# ✅ Resample train, ALI test ostaje original
# Ovo daje realističku procenu kako model radi u produkciji!
```

---

## Common Pitfalls

### Greška 1: Resampling Pre Split-a
```python
# ❌ LOŠE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
X_train, X_test = train_test_split(X_smote, y_smote)

# Sintetički samples iz test podataka mogu biti u train!

# ✅ DOBRO
X_train, X_test = train_test_split(X, y)
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

### Greška 2: Ignoring Test Distribution
```python
# ❌ LOŠE - Test set resampled
# Model se evaluira na balanced data, ali produkcija je imbalanced!

# ✅ DOBRO - Test set original
# Realna evaluacija za production environment
```

### Greška 3: Using Wrong Metric
```python
# ❌ LOŠE
# Model sa 95% accuracy koji nikad ne predviđa minority class

# ✅ DOBRO
# Model sa 85% accuracy, ali 70% recall na minority class
# Ovaj model je BOLJI za imbalanced problem!
```

---

## Complete Example - Best Practices
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Generate imbalanced data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000, 
    n_features=20, 
    n_informative=15,
    n_classes=2,
    weights=[0.95, 0.05],  # 95/5 imbalance
    random_state=42
)

print("Class distribution:")
print(pd.Series(y).value_counts(normalize=True))

# 2. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# 3. Create Pipeline (resample samo u train!)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # Extra protection
        random_state=42
    ))
])

# 4. Cross-Validation (stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')
print(f"\nCV F1-Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 5. Train final model
pipeline.fit(X_train, y_train)

# 6. Predict (test je ORIGINAL imbalanced!)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# 7. Comprehensive Evaluation
print("\n" + "="*50)
print("FINAL EVALUATION (on original imbalanced test set)")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.3f}")

# 8. Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")

print(f"\nRecall (Sensitivity): {tp/(tp+fn):.3f}")
print(f"Specificity: {tn/(tn+fp):.3f}")
print(f"Precision: {tp/(tp+fp):.3f}")

# 9. Threshold tuning (optional)
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nOptimal Threshold: {best_threshold:.3f}")

y_pred_optimal = (y_proba >= best_threshold).astype(int)
print("\nWith Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))
```

---

## Rezime - Handling Imbalanced Data

### Tehnike po Imbalance Ratio:

| Imbalance | Ratio | Tehnike |
|-----------|-------|---------|
| **Mild** | 60/40 - 70/30 | Class weights |
| **Moderate** | 80/20 - 90/10 | SMOTE + Class weights |
| **High** | 95/5 - 99/1 | SMOTE + Ensemble, Balanced RF |
| **Extreme** | >99/1 | Anomaly detection, Cost-sensitive |

### Evaluation Metrics:

| Metric | Kada Koristiti |
|--------|---------------|
| **F1-Score** | Balans precision/recall |
| **Recall** | False negatives su VEOMA skupi (cancer detection) |
| **Precision** | False positives su VEOMA skupi (spam sa važnim emailovima) |
| **PR-AUC** | **NAJBOLJI za imbalanced** - Overall performance |
| **ROC-AUC** | Comparison tool |
| **Balanced Accuracy** | Average performance po klasama |

### Default Strategy:
```
1. Stratified train-test split
2. SMOTE oversampling (na train!)
3. Class weights (model parameter)
4. Evaluate sa F1/PR-AUC (ne accuracy!)
5. Threshold tuning (optional)
6. Cross-validation sa stratification
```

**Key Takeaway:** Imbalanced data je norm, ne exception! Accuracy je OBMANJUJUĆI metric. UVEK koristi F1, Recall, Precision, i PR-AUC. Resample SAMO train set, nikad test! Kombinuj više tehnika za najbolje rezultate. 🎯