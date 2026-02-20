# Logistic Regression

Logistic Regression je **najčešće korišćeni algoritam za binary classification**. Uprkos imenu, ovo je **classification algoritam**, ne regression!

**Zašto je Logistic Regression fundamentalan?**
- **Interpretabilnost** - Koeficijenti pokazuju uticaj svake feature na verovatnoću
- **Probabilistic output** - Daje verovatnoću klase, ne samo labelu
- **Baseline classification model** - Prva stvar koju testirate
- **Brz** - Trenira se veoma brzo
- **Skalabilan** - Radi dobro čak i sa milionima podataka

**Kada koristiti Logistic Regression?**
- ✅ Binary ili multiclass classification
- ✅ Potrebna interpretabilnost (medical, finance, legal)
- ✅ Potrebne verovatnoće (ne samo labels)
- ✅ Linearna separacija između klasa (ili feature engineering)
- ✅ Brz trening i deployment

**Kada NE koristiti:**
- ❌ Kompleksne nelinearne decision boundaries (koristi SVM, tree-based, NN)
- ❌ Visoka multicollinearity koju ne možeš rešiti
- ❌ Ekstremno imbalanced data bez tretmana (vidi folder 08/03_Handling_Imbalanced_Data.md)

---

## Matematička Osnova

### Binary Classification

**Cilj:** Predvideti klasu (0 ili 1) na osnovu features.

**Problem sa Linear Regression za Classification:**
```
Linear: y = β₀ + β₁x₁ + β₂x₂ + ...
Output: Bilo koji broj (-∞ do +∞)

Ali za classification, trebamo:
Output: Verovatnoća između 0 i 1
```

**Rešenje:** **Sigmoid funkcija** koja "zgužva" output u [0, 1]!

### Sigmoid Function (Logistic Function)
```
σ(z) = 1 / (1 + e^(-z))

gde je:
z = β₀ + β₁x₁ + β₂x₂ + ... (linear combination)
```

**Karakteristike:**
- Output: [0, 1] - može se interpretirati kao **verovatnoća**
- σ(0) = 0.5 (decision boundary)
- σ(+∞) → 1 (klasa 1)
- σ(-∞) → 0 (klasa 0)

**Vizualizacija:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot
z = np.linspace(-10, 10, 200)
y = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, linewidth=2, color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Decision Threshold (0.5)')
plt.axvline(x=0, color='green', linestyle='--', linewidth=1, label='z=0')
plt.xlabel('z (linear combination)', fontsize=12)
plt.ylabel('σ(z) (probability)', fontsize=12)
plt.title('Sigmoid Function', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim([-0.1, 1.1])
plt.show()
```

### Model
```
P(y=1|X) = σ(β₀ + β₁x₁ + β₂x₂ + ...)

Prediction:
  - If P(y=1|X) ≥ 0.5 → Predict class 1
  - If P(y=1|X) < 0.5 → Predict class 0
```

### Log-Odds (Logit)
```
log(P / (1-P)) = β₀ + β₁x₁ + β₂x₂ + ...

gde je P = P(y=1|X)

Ovo je LINEARNA funkcija!
```

**Interpretacija koeficijenata:**
- Pozitivan β₁ → Porast x₁ → Porast verovatnoće klase 1
- Negativan β₁ → Porast x₁ → Pad verovatnoće klase 1
- Magnitude koeficijenta → Koliko jak je uticaj

---

## Python Implementacija - Binary Classification

### Primer 1: Simple Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate synthetic data
np.random.seed(42)

# Class 0: mean=2
X_class0 = np.random.randn(100, 2) + np.array([2, 2])
y_class0 = np.zeros(100)

# Class 1: mean=5
X_class1 = np.random.randn(100, 2) + np.array([5, 5])
y_class1 = np.ones(100)

# Combine
X = np.vstack([X_class0, X_class1])
y = np.hstack([y_class0, y_class1])

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Coefficients
print(f"Intercept: {model.intercept_[0]:.3f}")
print(f"Coefficients: {model.coef_[0]}")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")

# Example predictions
print("\nFirst 5 test samples:")
print("True Label | Predicted | P(Class 0) | P(Class 1)")
for i in range(5):
    print(f"    {int(y_test[i])}      |     {int(y_pred[i])}     |   {y_pred_proba[i, 0]:.3f}    |   {y_pred_proba[i, 1]:.3f}")
```

### Decision Boundary Visualization
```python
# Create mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.7, edgecolors='k')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.7, edgecolors='k')

# Decision boundary (gdzie P=0.5)
contour = plt.contour(xx, yy, model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape),
                     levels=[0.5], colors='black', linewidths=2)
plt.clabel(contour, inline=True, fontsize=10, fmt='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Probability Thresholds

**Default threshold: 0.5**  
Ali možeš ga promeniti zavisno od problema!
```python
# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]  # Verovatnoća klase 1

# Different thresholds
thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_threshold)
    
    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_threshold))
```

**Kada koristiti različite thresholds:**
- **0.3 (Lower)** - Želiš više **recall** (catch više positives, even ako imaš false alarms)
  - Primer: Medical diagnosis (bolje detect bolest nego propustiti)
- **0.5 (Default)** - Balansiran approach
- **0.7 (Higher)** - Želiš više **precision** (samo sigurni positives)
  - Primer: Spam detection (bolje propustiti spam nego blokirati important email)

**Za detalje o metrics i threshold tuning, vidi:** `05_Model_Evaluation_and_Tuning/01_Classification_Metrics.md`

---

## Regularization

Logistic Regression može overfitovati sa mnogo features. **Regularization** penalizuje velike koeficijente.
```python
from sklearn.linear_model import LogisticRegression

# C je INVERSE regularization strength
# Manji C = jača regularization

models = {
    'No Regularization (C=1e10)': LogisticRegression(C=1e10, max_iter=1000),
    'Weak Regularization (C=10)': LogisticRegression(C=10, max_iter=1000),
    'Moderate (C=1) [DEFAULT]': LogisticRegression(C=1, max_iter=1000),
    'Strong Regularization (C=0.1)': LogisticRegression(C=0.1, max_iter=1000)
}

for name, model_reg in models.items():
    model_reg.fit(X_train, y_train)
    train_acc = model_reg.score(X_train, y_train)
    test_acc = model_reg.score(X_test, y_test)
    
    print(f"{name}")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy:  {test_acc:.3f}")
    print()
```

**Penalties:**
- **'l2'** (default) - Ridge penalty (smanjuje sve koeficijente)
- **'l1'** - Lasso penalty (neke koeficijente svodi na 0 → feature selection)
- **'elasticnet'** - Kombinacija L1 i L2
```python
# L1 penalty (feature selection)
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
model_l1.fit(X_train, y_train)

print(f"L1 Coefficients: {model_l1.coef_[0]}")
print(f"Non-zero coefficients: {np.sum(model_l1.coef_[0] != 0)}")
```

**Za detalje o Regularization, vidi:** `05_Model_Evaluation_and_Tuning/06_Regularization.md`

---

## Multiclass Classification

Logistic Regression može raditi i sa **više od 2 klase**!

### Strategije:

**1. One-vs-Rest (OvR)** - Trenira N binary classifiers (jedan po klasi)  
**2. Multinomial** - Trenira jedan multi-output model (softmax)
```python
from sklearn.datasets import load_iris

# Load Iris dataset (3 klase)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# One-vs-Rest (default)
model_ovr = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ovr.fit(X_train_i, y_train_i)
acc_ovr = model_ovr.score(X_test_i, y_test_i)

# Multinomial
model_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
model_multi.fit(X_train_i, y_train_i)
acc_multi = model_multi.score(X_test_i, y_test_i)

print(f"One-vs-Rest Accuracy:  {acc_ovr:.3f}")
print(f"Multinomial Accuracy:  {acc_multi:.3f}")

# Predictions
y_pred_multi = model_multi.predict(X_test_i)
y_proba_multi = model_multi.predict_proba(X_test_i)

print("\nFirst 5 predictions:")
print("True | Pred | P(Class 0) | P(Class 1) | P(Class 2)")
for i in range(5):
    print(f" {y_test_i[i]}   |  {y_pred_multi[i]}   |   {y_proba_multi[i, 0]:.3f}    |   {y_proba_multi[i, 1]:.3f}    |   {y_proba_multi[i, 2]:.3f}")
```

---

## Feature Scaling

**Logistic Regression NIJE jako osetljiv na scaling** (za razliku od KNN ili SVM), **ALI:**
- ✅ **Regularization zahteva scaling** (inače features sa većim vrednostima bivaju više penalized)
- ✅ **Brža konvergencija** (gradient descent konvergira brže)
- ✅ **Coefficient interpretation** (lakše uporediti uticaje)
```python
from sklearn.preprocessing import StandardScaler

# Sa scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression()
model_scaled.fit(X_train_scaled, y_train)

# Bez scaling
model_no_scale = LogisticRegression()
model_no_scale.fit(X_train, y_train)

print(f"Scaled Model Accuracy:     {model_scaled.score(X_test_scaled, y_test):.3f}")
print(f"Non-Scaled Model Accuracy: {model_no_scale.score(X_test, y_test):.3f}")
print("\nCoefficients (scaled):     ", model_scaled.coef_[0])
print("Coefficients (non-scaled): ", model_no_scale.coef_[0])
```

**Za detalje o Feature Scaling, vidi:** `01_Data_Preprocessing/06_Feature_Scaling.md`

---

## Imbalanced Data

**Problem:** Ako imaš 95% klase 0 i 5% klase 1, model može predvideti sve kao klasu 0 i imati 95% accuracy!

**Rešenje: Class Weights**
```python
# Imbalanced data
X_imb = np.vstack([np.random.randn(900, 2) + [2, 2],  # Class 0: 900 samples
                   np.random.randn(100, 2) + [5, 5]]) # Class 1: 100 samples
y_imb = np.hstack([np.zeros(900), np.ones(100)])

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42
)

print("Class distribution:")
print(f"  Class 0: {np.sum(y_train_imb == 0)} samples")
print(f"  Class 1: {np.sum(y_train_imb == 1)} samples")

# Default (bez weights)
model_default = LogisticRegression()
model_default.fit(X_train_imb, y_train_imb)

# Balanced class weights
model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train_imb, y_train_imb)

# Evaluate
from sklearn.metrics import classification_report

print("\n" + "="*50)
print("DEFAULT MODEL (bez class weights)")
print("="*50)
print(classification_report(y_test_imb, model_default.predict(X_test_imb), 
                           target_names=['Class 0', 'Class 1']))

print("\n" + "="*50)
print("BALANCED MODEL (sa class weights)")
print("="*50)
print(classification_report(y_test_imb, model_balanced.predict(X_test_imb),
                           target_names=['Class 0', 'Class 1']))
```

**Za detalje o Imbalanced Data, vidi:** `01_Data_Preprocessing/08_Feature_Engineering/03_Handling_Imbalanced_Data.md`

---

## Complete Example: Medical Diagnosis
```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, confusion_matrix, 
                              roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MEDICAL DIAGNOSIS - BREAST CANCER CLASSIFICATION")
print("="*60)

# ==================== 1. LOAD DATA ====================
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target  # 0 = malignant, 1 = benign

print(f"\nDataset: {cancer.filename}")
print(f"Samples: {X_cancer.shape[0]}")
print(f"Features: {X_cancer.shape[1]}")
print(f"\nTarget distribution:")
print(f"  Malignant (0): {np.sum(y_cancer == 0)} ({np.sum(y_cancer == 0)/len(y_cancer)*100:.1f}%)")
print(f"  Benign (1):    {np.sum(y_cancer == 1)} ({np.sum(y_cancer == 1)/len(y_cancer)*100:.1f}%)")

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

print(f"\nTrain set: {X_train_c.shape}")
print(f"Test set:  {X_test_c.shape}")

# ==================== 3. FEATURE SCALING ====================
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

print("\n✅ Features scaled")

# ==================== 4. BASELINE ====================
print("\n" + "="*60)
print("BASELINE - Predict Most Frequent Class")
print("="*60)

most_frequent = np.bincount(y_train_c).argmax()
y_baseline = np.full(len(y_test_c), most_frequent)

baseline_acc = accuracy_score(y_test_c, y_baseline)
print(f"Baseline Accuracy: {baseline_acc:.3f}")

# ==================== 5. LOGISTIC REGRESSION ====================
print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

# Train
model_cancer = LogisticRegression(max_iter=10000, random_state=42)
model_cancer.fit(X_train_c_scaled, y_train_c)

# Predictions
y_train_pred_c = model_cancer.predict(X_train_c_scaled)
y_test_pred_c = model_cancer.predict(X_test_c_scaled)
y_test_proba_c = model_cancer.predict_proba(X_test_c_scaled)[:, 1]

# Metrics
train_acc = accuracy_score(y_train_c, y_train_pred_c)
test_acc = accuracy_score(y_test_c, y_test_pred_c)
test_precision = precision_score(y_test_c, y_test_pred_c)
test_recall = recall_score(y_test_c, y_test_pred_c)
test_f1 = f1_score(y_test_c, y_test_pred_c)
test_auc = roc_auc_score(y_test_c, y_test_proba_c)

print(f"\nTrain Accuracy: {train_acc:.3f}")
print(f"\nTest Performance:")
print(f"  Accuracy:  {test_acc:.3f}")
print(f"  Precision: {test_precision:.3f}")
print(f"  Recall:    {test_recall:.3f}")
print(f"  F1-Score:  {test_f1:.3f}")
print(f"  ROC-AUC:   {test_auc:.3f}")

print(f"\nImprovement over baseline: +{(test_acc - baseline_acc):.3f}")

# ==================== 6. CONFUSION MATRIX ====================
cm = confusion_matrix(y_test_c, y_test_pred_c)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm)
print("\nInterpretacija:")
print(f"  True Negatives (TN):  {cm[0, 0]} (correctly predicted malignant)")
print(f"  False Positives (FP): {cm[0, 1]} (malignant predicted as benign) ⚠️")
print(f"  False Negatives (FN): {cm[1, 0]} (benign predicted as malignant) ⚠️")
print(f"  True Positives (TP):  {cm[1, 1]} (correctly predicted benign)")

# ==================== 7. ROC CURVE ====================
fpr, tpr, thresholds_roc = roc_curve(y_test_c, y_test_proba_c)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {roc_auc_value:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 8. FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

# Top 10 features by absolute coefficient
feature_importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Coefficient': model_cancer.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualization
top_10 = feature_importance.head(10)
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in top_10['Coefficient']]
plt.barh(top_10['Feature'], top_10['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importances\nGreen = Increases probability of Benign')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ==================== 9. CLASSIFICATION REPORT ====================
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test_c, y_test_pred_c, 
                           target_names=['Malignant', 'Benign']))

# ==================== 10. SAVE MODEL ====================
import joblib

joblib.dump(model_cancer, 'logistic_regression_cancer.pkl')
joblib.dump(scaler_c, 'scaler_cancer.pkl')

print("\n✅ Model saved: logistic_regression_cancer.pkl")
print("✅ Scaler saved: scaler_cancer.pkl")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Key Hyperparameters
```python
LogisticRegression(
    penalty='l2',           # 'l1', 'l2', 'elasticnet', None
    C=1.0,                  # Inverse regularization (manji = jača reg)
    solver='lbfgs',         # 'lbfgs', 'liblinear', 'saga'
    max_iter=100,           # Max iterations
    multi_class='auto',     # 'ovr', 'multinomial'
    class_weight=None,      # None ili 'balanced'
    random_state=42
)
```

**Najvažniji:**
- **C** - Regularization strength (tuning range: 0.001 to 100)
- **penalty** - L1 (feature selection) vs L2 (shrinkage)
- **class_weight** - 'balanced' za imbalanced data

**Za Hyperparameter Tuning, vidi:** `05_Model_Evaluation_and_Tuning/05_Hyperparameter_Tuning.md`

---

## Best Practices

### ✅ DO:

1. **Scale features** - Posebno sa regularization
2. **Check class balance** - Koristi `class_weight='balanced'` ako je imbalanced
3. **Tune threshold** - Ne zadržavaj 0.5 ako nije optimalno
4. **Use probabilities** - `predict_proba()` daje više informacija od `predict()`
5. **Regularization** - Skoro uvek koristi (default C=1.0 je OK)
6. **Cross-validation** - Za hyperparameter tuning
7. **Interpretacija** - Koristi koeficijente za business insights

### ❌ DON'T:

1. **Ne ignoriši imbalanced data** - class_weight rešava većinu problema
2. **Ne zaboravi scaling** - Posebno sa regularization!
3. **Ne koristi samo accuracy** - Gledaj precision, recall, F1, ROC-AUC
4. **Ne koristi za kompleksne nelinearne probleme** - Koristi SVM/tree-based/NN
5. **Ne preteruj sa polynomial features** - Brzo overfit-uje

---

## Common Pitfalls

### Greška 1: Ignoring Imbalanced Data
```python
# ❌ LOŠE - Model će predvideti sve kao majority class
model_bad = LogisticRegression()
model_bad.fit(X_imbalanced, y_imbalanced)

# ✅ DOBRO
model_good = LogisticRegression(class_weight='balanced')
model_good.fit(X_imbalanced, y_imbalanced)
```

### Greška 2: Not Scaling sa Regularization
```python
# ❌ LOŠE - Features sa velikim vrednostima bivaju više penalized
model_no_scale = LogisticRegression(C=1.0)
model_no_scale.fit(X_unscaled, y)  # Feature 1: 0-1, Feature 2: 0-10000

# ✅ DOBRO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)
model_scaled = LogisticRegression(C=1.0)
model_scaled.fit(X_scaled, y)
```

### Greška 3: Using Only Accuracy
```python
# ❌ LOŠE - Accuracy ne govori sve!
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# ✅ DOBRO - Gledaj više metrika
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## Kada Koristiti Logistic Regression?

### ✅ Idealno Za:

- **Binary classification** (najčešći use case)
- **Multiclass classification** (sa OvR ili Multinomial)
- **Potrebna interpretabilnost** (medical, finance, legal)
- **Potrebne verovatnoće** (ranking, decision-making)
- **Brz training** (real-time ili large-scale)
- **Linearna separacija** (ili sa feature engineering)

### ❌ Izbegavaj Za:

- **Kompleksne nelinearne decision boundaries** → SVM, Random Forest, XGBoost
- **Image/audio classification** → Neural Networks
- **Ekstremno imbalanced data** → Tree-based + SMOTE/undersampling
- **Više interakcija između features** → Tree-based ili polynomial features

---

## Rezime

| Aspekt | Opis |
|--------|------|
| **Tip** | Supervised Learning - Classification |
| **Output** | Klasa (0/1) + Verovatnoće |
| **Interpretabilnost** | ⭐⭐⭐⭐⭐ Excellent (koeficijenti = uticaj) |
| **Training Speed** | ⭐⭐⭐⭐⭐ Veoma brz |
| **Prediction Speed** | ⭐⭐⭐⭐⭐ Instant |
| **Linearity Required** | Da (ili feature engineering) |
| **Handles Non-linearity** | ❌ (osim sa polynomial features) |
| **Handles Imbalanced** | ✅ (sa class_weight='balanced') |
| **Feature Scaling** | Preporučeno (za regularization) |
| **Regularization** | ✅ Built-in (L1/L2/ElasticNet) |
| **Multiclass** | ✅ (OvR ili Multinomial) |
| **Best For** | Binary classification, interpretabilnost, probabilities |

---

## Quick Decision Tree
```
Start
  ↓
Classification problem?
  ↓ Yes
Potrebne verovatnoće (ne samo labels)?
  ↓ Yes
Interpretabilnost je važna?
  ↓ Yes
Linearna ili skoro-linearna separacija?
  ↓ Yes
→ LOGISTIC REGRESSION ✅

Ako bilo šta "No":
  ├─ Kompleksne nelinearne veze? → SVM (RBF), Random Forest, XGBoost
  ├─ Ne treba interpretacija? → Tree-based, Neural Networks
  ├─ Images/Text/Sequences? → Neural Networks (CNN/RNN/Transformer)
  └─ Samo labels (ne verovatnoće)? → SVM, KNN
```

---

**Key Takeaway:** Logistic Regression je **go-to algoritam za binary classification** kada ti je potrebna **interpretabilnost** i **verovatnoće**. Brz je, robustan, i sa `class_weight='balanced'` radi dobro i na imbalanced data. Za kompleksnije probleme postoje moćniji algoritmi, ali Logistic Regression je **uvek odličan baseline**! 🎯