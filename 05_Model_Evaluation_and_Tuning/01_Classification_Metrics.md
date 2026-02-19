# Classification Metrics
Classification metrics su **metode za evaluaciju performansi classification modela**. Različite metrike mere različite aspekte modela i izbor odgovarajuće metrike zavisi od problema, cost-a grešaka, i distribucije klasa.

**Zašto je izbor metrike kritičan?**
- **Accuracy može biti obmanjujuća** - 95% accuracy zvuči sjajno, ali možda model nikad ne predviđa minority class!
- **Različiti problemi = različite metrike** - Medical diagnosis (Recall!) vs Spam detection (Precision!)
- **Cost različitih grešaka** - False Negative (propuštena bolest) može biti MNOGO skuplje od False Positive
- **Imbalanced data** - Standardne metrike ne funkcionišu dobro

**VAŽNO:** Nikad ne evaluiraj model sa samo jednom metrikom! Koristi više metrika za kompletnu sliku.

---

## Confusion Matrix - Osnova Svega

**Confusion Matrix** prikazuje **stvarne vs predviđene klase** i služi kao osnova za sve ostale metrike.

### Binary Classification (2 klase):
```
                    Predicted
                 Negative  Positive
              ┌──────────┬──────────┐
Actual    Neg │    TN    │    FP    │
          Pos │    FN    │    TP    │
              └──────────┴──────────┘

TN (True Negative)  - Tačno predviđeno kao Negative
FP (False Positive) - Pogrešno predviđeno kao Positive (Type I Error)
FN (False Negative) - Pogrešno predviđeno kao Negative (Type II Error)
TP (True Positive)  - Tačno predviđeno kao Positive
```

### Python Implementacija:
```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Sample predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
#     [[5 1]     ← Row 0: Actual Negative (0)
#      [1 3]]    ← Row 1: Actual Positive (1)
#       ↑  ↑
#    Col0 Col1
#    Pred Pred
#     0    1

# Extract values
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn}")  # 5
print(f"False Positives: {fp}")     # 1
print(f"False Negatives: {fn}")     # 1
print(f"True Positives:  {tp}")     # 3

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()
```

### Heatmap Visualization:
```python
import seaborn as sns

# Pretty heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Sa percentages
cm_normalized = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Normalized)')
plt.show()
```

---

## 1. Accuracy (Preciznost)

**Procenat tačnih predikcija** od ukupnog broja predikcija.

### Formula:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct Predictions / Total Predictions
```

### Python:
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")  # 0.800

# Manual calculation
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
print(f"Manual: {accuracy_manual:.3f}")  # 0.800
```

### Kada Koristiti:

✅ **DOBRO za:**
- Balanced datasets (50/50 ili 60/40)
- Sve greške imaju isti cost
- Quick baseline metric

❌ **LOŠE za:**
- **Imbalanced data** - Velika accuracy ne znači ništa!
- Različiti cost-ovi grešaka
- Medical/fraud detection

### Primer - Zašto Accuracy NE Radi za Imbalanced:
```python
# Extreme imbalance: 99% class 0, 1% class 1
y_true_imb = np.array([0]*990 + [1]*10)  # 990 negative, 10 positive

# "Dumb" model - UVEK predviđa 0
y_pred_dumb = np.array([0]*1000)

accuracy_dumb = accuracy_score(y_true_imb, y_pred_dumb)
print(f"Dumb Model Accuracy: {accuracy_dumb:.1%}")  # 99.0%!

# Izgleda odlično, ALI model je beskoristan!
# NIKAD ne detektuje class 1 (minority)!

cm_dumb = confusion_matrix(y_true_imb, y_pred_dumb)
print("\nConfusion Matrix:")
print(cm_dumb)
#     [[990   0]    ← Svi class 0 tačni
#      [ 10   0]]   ← SVI class 1 POGREŠNI!

# TP = 0! Model potpuno ignoriše minority class!
```

---

## 2. Precision (Pozitivna Prediktivna Vrednost)

**Od predviđenih pozitivnih, koliko je zaista pozitivno?**

### Formula:
```
Precision = TP / (TP + FP)
          = True Positives / Predicted Positives
```

**Pitanje koje Precision odgovara:**
"Kada model kaže Positive, koliko mu verujem?"

### Python:
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.3f}")

# Manual
precision_manual = tp / (tp + fp)
print(f"Manual: {precision_manual:.3f}")
```

### Interpretacija:
```python
# Precision = 0.75 (75%)
# Znači: Kada model predvidi Positive, 75% vremena je u pravu
# 25% su False Positives (Type I Error)
```

### Kada Koristiti Precision:

✅ **PRIORITIZUJ kada:**
- **False Positives su skupi**
- Spam detection - Ne želiš da važan email ide u spam!
- Marketing campaigns - Ne želiš da nerviraš ljude koji nisu zainteresovani
- Fraud alerts - Previše false alarms → ljudi ignorišu
- Cancer screening (initial) - Ne želiš da preplašiš zdrave ljude

### Primer - Spam Detection:
```python
# Email dataset
emails = pd.DataFrame({
    'actual': ['spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam'] * 100,
    'predicted': ['spam', 'spam', 'spam', 'not spam', 'spam', 'not spam'] * 100
})

# Encode
y_true_email = (emails['actual'] == 'spam').astype(int)
y_pred_email = (emails['predicted'] == 'spam').astype(int)

precision_email = precision_score(y_true_email, y_pred_email)
print(f"Spam Precision: {precision_email:.3f}")

# Ako precision = 0.85:
# 85% emailova u spam folderu su zaista spam
# 15% su FALSE POSITIVES - važni emailovi u spam! ❌
```

---

## 3. Recall (Sensitivity, True Positive Rate)

**Od stvarno pozitivnih, koliko smo uhvatili?**

### Formula:
```
Recall = TP / (TP + FN)
       = True Positives / Actual Positives
```

**Pitanje koje Recall odgovara:**
"Od svih pozitivnih slučajeva, koliko smo pronašli?"

### Python:
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.3f}")

# Manual
recall_manual = tp / (tp + fn)
print(f"Manual: {recall_manual:.3f}")
```

### Interpretacija:
```python
# Recall = 0.80 (80%)
# Znači: Od svih pozitivnih cases, uhvatili smo 80%
# 20% su False Negatives (Type II Error) - propustili smo ih!
```

### Kada Koristiti Recall:

✅ **PRIORITIZUJ kada:**
- **False Negatives su VEOMA skupi**
- Cancer/disease detection - Ne smeš propustiti bolest!
- Fraud detection - Bolje 10 false alarms nego propustiti jednu prevaru
- Terrorist detection - Propušteni threat je katastrofalan
- Search engines - Bolje vratiti više rezultata (neke irelevantne) nego propustiti relevantne

### Primer - Cancer Screening:
```python
# Medical screening
patients = pd.DataFrame({
    'actual_cancer': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 50,
    'test_positive': [1, 0, 1, 0, 0, 1, 0, 1, 1, 1] * 50
})

y_true_cancer = patients['actual_cancer']
y_pred_cancer = patients['test_positive']

recall_cancer = recall_score(y_true_cancer, y_pred_cancer)
print(f"Cancer Test Recall: {recall_cancer:.3f}")

# Ako recall = 0.85:
# 85% pacijenata sa rakom je detektovano
# 15% je FALSE NEGATIVES - PROPUŠTENI! ❌❌❌
# Ovo je KATASTROFA u medicinskom kontekstu!
```

---

## 4. F1-Score (Harmonijska Sredina)

**Balans između Precision i Recall** - korisno kada trebaš i jedno i drugo.

### Formula:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonijska sredina precision i recall
```

**Zašto harmonijska, ne aritmetička?**
- Harmonijska sredina PENALIZUJE ekstremne vrednosti
- Ako precision=1.0 i recall=0.1 → F1 će biti niska (~0.18)
- Aritmetička sredina bi bila 0.55 (obmanjujuće visoko!)

### Python:
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.3f}")

# Manual
f1_manual = 2 * (precision * recall) / (precision + recall)
print(f"Manual: {f1_manual:.3f}")
```

### Kada Koristiti F1:

✅ **DOBRO za:**
- **Imbalanced data** - Bolje od accuracy!
- Trebaš balans precision i recall
- Ne znaš koji je važniji (precision vs recall)
- Default metric za binary classification

### F-Beta Score (Weighted F1):
```python
from sklearn.metrics import fbeta_score

# Beta = 0.5: Precision je 2× važniji od Recall
f05 = fbeta_score(y_true, y_pred, beta=0.5)
print(f"F0.5-Score: {f05:.3f}")

# Beta = 2: Recall je 2× važniji od Precision
f2 = fbeta_score(y_true, y_pred, beta=2)
print(f"F2-Score: {f2:.3f}")

# Formula:
# F_beta = (1 + beta²) × (precision × recall) / (beta² × precision + recall)
```

**Kada koristiti F-Beta:**
- **F0.5** - Precision važniji (spam, marketing)
- **F1** - Balans (default)
- **F2** - Recall važniji (medical, fraud)

---

## 5. Specificity (True Negative Rate)

**Od stvarno negativnih, koliko smo tačno identifikovali kao negativne?**

### Formula:
```
Specificity = TN / (TN + FP)
            = True Negatives / Actual Negatives
```

### Python:
```python
# sklearn nema direktnu funkciju, manual calculation
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.3f}")

# Ili sa confusion matrix
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

spec = calculate_specificity(y_true, y_pred)
print(f"Specificity: {spec:.3f}")
```

### Kada Koristiti:

✅ **Važno za:**
- Medical screening - Healthy people correctly identified as healthy
- Krediti - Non-defaulters correctly identified
- Balans sa Sensitivity (Recall)

**Sensitivity vs Specificity:**
- **Sensitivity (Recall)** - Koliko pozitivnih smo uhvatili?
- **Specificity** - Koliko negativnih smo tačno identifikovali?

---

## 6. ROC Curve & ROC-AUC

**Receiver Operating Characteristic** curve pokazuje **trade-off između True Positive Rate (Recall) i False Positive Rate** na različitim threshold-ima.

### Kako Radi:
```
Model daje PROBABILITIES, ne binary predictions!

Threshold = 0.5:  prediction = 1 if P(class=1) > 0.5 else 0
Threshold = 0.3:  prediction = 1 if P(class=1) > 0.3 else 0
Threshold = 0.7:  prediction = 1 if P(class=1) > 0.7 else 0

Za svaki threshold → izračunaj TPR i FPR → plot!
```

### Formule:
```
TPR (True Positive Rate) = Recall = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)

ROC Curve: Plot TPR vs FPR za sve threshold-e
AUC (Area Under Curve): Površina ispod ROC curve (0 to 1)
```

### Python:
```python
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate data
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           weights=[0.7, 0.3], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Probabilities (MORAMO imati probability za ROC!)
y_proba = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.3f}")

# Plot ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR / Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Ili sa RocCurveDisplay
fig, ax = plt.subplots(figsize=(10, 6))
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
plt.title('ROC Curve (sklearn)')
plt.show()
```

### ROC-AUC Interpretacija:
```
AUC = 1.0   - Perfektan model
AUC = 0.9+  - Odličan model
AUC = 0.8-0.9 - Dobar model
AUC = 0.7-0.8 - OK model
AUC = 0.5-0.7 - Loš model
AUC = 0.5   - Random guessing (beskoristan)
AUC < 0.5   - Gori od random (inverzuj predictions!)
```

### Šta AUC Zapravo Meri?
```python
# AUC = Verovatnoća da model rangira
# random pozitivan sample VIŠE od random negativnog sample-a

# AUC = 0.85 znači:
# U 85% slučajeva, model daje višu probability pozitivnom sample-u
# nego negativnom sample-u
```

### Kada Koristiti ROC-AUC:

✅ **DOBRO za:**
- Balanced ili blago imbalanced data
- Comparing modela (viši AUC = bolji)
- Threshold-agnostic evaluation
- Overall model performance

❌ **LOŠE za:**
- **Extreme imbalanced data** - ROC-AUC je previše optimističan!
- Kada te više brine precision (PR-AUC bolji!)

---

## 7. Precision-Recall Curve & PR-AUC

**Precision vs Recall** na različitim threshold-ima - **BOLJI od ROC za imbalanced data!**

### Zašto PR Curve za Imbalanced?
```
ROC Curve koristi FPR = FP / (FP + TN)

Kada TN >> FP (mnogo negative examples), FPR ostaje niska
čak i ako ima dosta FP! ROC izgleda dobro, ali model je loš!

PR Curve koristi Precision = TP / (TP + FP)
Direktno penalizuje FP, ne zavisi od TN!
```

### Python:
```python
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

# Imbalanced data (90/10)
X_imb, y_imb = make_classification(n_samples=1000, n_features=20, 
                                    weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_imb, y_imb, test_size=0.3, 
                                                     stratify=y_imb, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# PR Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# PR-AUC (Average Precision)
pr_auc = average_precision_score(y_test, y_proba)
print(f"PR-AUC: {pr_auc:.3f}")

# Plot PR Curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'Model (AP = {pr_auc:.3f})', linewidth=2)

# Baseline (random)
baseline = y_test.sum() / len(y_test)  # Proportion of positive class
plt.axhline(baseline, color='k', linestyle='--', label=f'Baseline (Random) = {baseline:.3f}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Ili sa PrecisionRecallDisplay
fig, ax = plt.subplots(figsize=(10, 6))
PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
plt.title('Precision-Recall Curve (sklearn)')
plt.show()
```

### ROC vs PR - Side by Side Comparison:
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC
roc_auc = roc_auc_score(y_test, y_proba)
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
axes[0].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')

# PR
pr_auc = average_precision_score(y_test, y_proba)
PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
axes[1].set_title(f'PR Curve (AP = {pr_auc:.3f})')

plt.tight_layout()
plt.show()

# Za imbalanced data: PR-AUC će biti MNOGO niži od ROC-AUC!
# PR-AUC je realističniji!
```

### Kada Koristiti PR-AUC:

✅ **PREFERIRAJ za:**
- **Imbalanced data** (< 30% minority)
- Kada je positive class važnija
- Fraud detection, anomaly detection, rare disease
- Preciznost je važna

---

## 8. Multi-Class Metrics

**Za probleme sa više od 2 klase** - potrebne su agregacione strategije.

### Averaging Strategies:
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Multi-class data
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred_multi = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2, 0])

# 1. MACRO Average - Prosek metrika po klasama (sve klase jednako važne)
precision_macro = precision_score(y_true_multi, y_pred_multi, average='macro')
recall_macro = recall_score(y_true_multi, y_pred_multi, average='macro')
f1_macro = f1_score(y_true_multi, y_pred_multi, average='macro')

print(f"Macro - Precision: {precision_macro:.3f}, Recall: {recall_macro:.3f}, F1: {f1_macro:.3f}")

# 2. WEIGHTED Average - Weighted by support (broj samples po klasi)
precision_weighted = precision_score(y_true_multi, y_pred_multi, average='weighted')
recall_weighted = recall_score(y_true_multi, y_pred_multi, average='weighted')
f1_weighted = f1_score(y_true_multi, y_pred_multi, average='weighted')

print(f"Weighted - Precision: {precision_weighted:.3f}, Recall: {recall_weighted:.3f}, F1: {f1_weighted:.3f}")

# 3. MICRO Average - Aggregate sve TP, FP, FN, zatim compute
precision_micro = precision_score(y_true_multi, y_pred_multi, average='micro')
recall_micro = recall_score(y_true_multi, y_pred_multi, average='micro')
f1_micro = f1_score(y_true_multi, y_pred_multi, average='micro')

print(f"Micro - Precision: {precision_micro:.3f}, Recall: {recall_micro:.3f}, F1: {f1_micro:.3f}")

# 4. Per-Class (None) - Metrike za svaku klasu pojedinačno
precision_per_class = precision_score(y_true_multi, y_pred_multi, average=None)
recall_per_class = recall_score(y_true_multi, y_pred_multi, average=None)
f1_per_class = f1_score(y_true_multi, y_pred_multi, average=None)

print(f"\nPer-Class Precision: {precision_per_class}")
print(f"Per-Class Recall: {recall_per_class}")
print(f"Per-Class F1: {f1_per_class}")
```

### Averaging Strategy Formulas:
```
MACRO:    avg(metric_class_0, metric_class_1, metric_class_2)
          → Sve klase imaju isti weight (good za balanced classes)

WEIGHTED: Σ(metric_class_i × support_class_i) / total_samples
          → Veće klase imaju veći weight (good za imbalanced)

MICRO:    Compute global TP, FP, FN → metric(global)
          → Za multi-class, micro F1 = micro Precision = micro Recall = Accuracy

Per-Class: [metric_class_0, metric_class_1, metric_class_2]
          → Metric za svaku klasu posebno
```

### Kada Koristiti:

| Average | Kada Koristiti |
|---------|---------------|
| **macro** | Balanced data, sve klase jednako važne |
| **weighted** | Imbalanced data, veće klase važnije |
| **micro** | Imbalanced data, overall accuracy je prioritet |
| **None (per-class)** | Potreban uvid u svaku klasu |

### Multi-Class Confusion Matrix:
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Multi-class confusion matrix
cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

print("Multi-Class Confusion Matrix:")
print(cm_multi)
#     [[4 0 0]    ← Class 0: 4 correct
#      [0 2 1]    ← Class 1: 2 correct, 1 confused with class 2
#      [0 1 2]]   ← Class 2: 2 correct, 1 confused with class 1

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_multi, 
                               display_labels=['Class 0', 'Class 1', 'Class 2'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Multi-Class Confusion Matrix')
plt.show()
```

### Multi-Class ROC-AUC:
```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Multi-class probabilities
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                     test_size=0.3, random_state=42)

model_multi = LogisticRegression(max_iter=200)
model_multi.fit(X_train, y_train)
y_proba_multi = model_multi.predict_proba(X_test)

# ROC-AUC za multi-class
# 1. One-vs-Rest (OVR)
roc_auc_ovr = roc_auc_score(y_test, y_proba_multi, multi_class='ovr')
print(f"ROC-AUC (OVR): {roc_auc_ovr:.3f}")

# 2. One-vs-One (OVO)
roc_auc_ovo = roc_auc_score(y_test, y_proba_multi, multi_class='ovo')
print(f"ROC-AUC (OVO): {roc_auc_ovo:.3f}")
```

---

## 9. Log Loss (Cross-Entropy Loss)

**Penalizuje confident wrong predictions** - meri kvalitet probability predictions.

### Formula:
```
Log Loss = -1/N × Σ[y_i × log(p_i) + (1 - y_i) × log(1 - p_i)]

Gde:
y_i = actual label (0 ili 1)
p_i = predicted probability za class 1
N = broj samples
```

### Interpretacija:
```
Log Loss = 0.00   - Perfektna probability
Log Loss = 0.3    - Dobra
Log Loss = 0.7+   - Loša
Log Loss = ∞      - Confident wrong prediction (p=1, actual=0)
```

### Python:
```python
from sklearn.metrics import log_loss

# Binary
y_proba_binary = np.array([0.9, 0.1, 0.8, 0.3, 0.2])
y_true_binary = np.array([1, 0, 1, 0, 0])

logloss = log_loss(y_true_binary, y_proba_binary)
print(f"Log Loss: {logloss:.3f}")

# Penalizuje confidence
y_proba_confident_wrong = np.array([0.99, 0.1, 0.8, 0.01, 0.2])  # 0.99 za class 1, actual 0!
y_true_same = np.array([0, 0, 1, 0, 0])  # First je actual 0!

logloss_wrong = log_loss(y_true_same, y_proba_confident_wrong)
print(f"Log Loss (confident wrong): {logloss_wrong:.3f}")  # VEOMA visok!
```

### Kada Koristiti:

✅ **DOBRO za:**
- Optimizacija probability calibration
- Neural networks (loss function)
- Ranking algorithms
- Kada precision probabilities bitna (not just binary prediction)

---

## 10. Cohen's Kappa

**Agreement između predictions i actual, adjusted for chance.**

### Formula:
```
Kappa = (p_o - p_e) / (1 - p_e)

p_o = Observed agreement (accuracy)
p_e = Expected agreement by chance
```

### Python:
```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.3f}")
```

### Interpretacija:
```
Kappa = 1.0     - Perfect agreement
Kappa = 0.8-1.0 - Almost perfect
Kappa = 0.6-0.8 - Substantial
Kappa = 0.4-0.6 - Moderate
Kappa = 0.2-0.4 - Fair
Kappa = 0.0-0.2 - Slight
Kappa = 0.0     - No agreement (random)
Kappa < 0       - Less than chance (sistemski loš!)
```

### Kada Koristiti:

✅ **Dobro za:**
- Inter-rater agreement (два annotatora)
- Imbalanced data (bolje od accuracy)
- Medical diagnosis agreement

---

## 11. Matthews Correlation Coefficient (MCC)

**Correlation između predictions i actuals** - najbolji single metric za imbalanced!

### Formula:
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

### Python:
```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC: {mcc:.3f}")
```

### Interpretacija:
```
MCC = +1    - Perfect prediction
MCC = 0     - No better than random
MCC = -1    - Total disagreement
```

### Kada Koristiti:

✅ **NAJBOLJI za:**
- **Extreme imbalanced data** (99.9/0.1)
- Sve 4 confusion matrix values su bitne
- Single metric koji obuhvata sve
- Alternative za F1 na imbalanced data

---

## Classification Report - All Metrics at Once
```python
from sklearn.metrics import classification_report

# Comprehensive report
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
print(report)

#               precision    recall  f1-score   support
#
#     Negative       0.88      0.95      0.91       100
#     Positive       0.83      0.65      0.73        50
#
#     accuracy                           0.87       150
#    macro avg       0.86      0.80      0.82       150
# weighted avg       0.86      0.87      0.86       150

# As dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)
print(f"Class 0 F1: {report_dict['0']['f1-score']:.3f}")
```

---

## Decision Framework - Koja Metrika?
```
Koji je problem?
│
├─→ BALANCED data (50/50, 60/40)?
│   └─→ Accuracy, F1-Score, ROC-AUC
│
├─→ IMBALANCED data?
│   ├─→ Moderate (80/20, 90/10)
│   │   └─→ F1-Score, PR-AUC
│   └─→ Extreme (>95/5)
│       └─→ PR-AUC, MCC, Per-class metrics
│
├─→ False Positives SU SKUPI? (spam, marketing)
│   └─→ **Precision** (F0.5 za weighted)
│
├─→ False Negatives SU SKUPI? (medical, fraud)
│   └─→ **Recall** (F2 za weighted)
│
├─→ Trebaš BALANS? (oba važna)
│   └─→ **F1-Score**
│
├─→ Threshold-agnostic evaluation?
│   ├─→ Balanced → ROC-AUC
│   └─→ Imbalanced → PR-AUC
│
├─→ Probabilities bitne? (ranking, calibration)
│   └─→ Log Loss, Brier Score
│
├─→ Multi-class?
│   ├─→ Balanced → Macro F1, Accuracy
│   ├─→ Imbalanced → Weighted F1
│   └─→ Overall → Micro F1
│
└─→ Single BEST metric za imbalanced?
    └─→ **MCC** ili **PR-AUC**
```

---

## Complete Example - Real-World Scenario

### Credit Card Fraud Detection:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt

# ==================== 1. GENERATE IMBALANCED DATA ====================
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000,
    n_features=30,
    n_informative=20,
    n_classes=2,
    weights=[0.997, 0.003],  # 99.7% legitimate, 0.3% fraud (extreme imbalance!)
    random_state=42
)

print(f"Fraud rate: {y.sum() / len(y):.1%}")  # ~0.3%
print(f"Fraud cases: {y.sum()}")
print(f"Legitimate cases: {len(y) - y.sum()}")

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ==================== 3. TRAIN MODEL ====================
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Important za imbalanced!
    random_state=42
)

model.fit(X_train, y_train)

# ==================== 4. PREDICTIONS ====================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ==================== 5. EVALUATION ====================

print("\n" + "="*60)
print("CREDIT CARD FRAUD DETECTION - EVALUATION")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"True Negatives:  {tn:5d} (Correctly identified legitimate)")
print(f"False Positives: {fp:5d} (Legitimate flagged as fraud)")
print(f"False Negatives: {fn:5d} (Fraud missed!) ← CRITICAL!")
print(f"True Positives:  {tp:5d} (Correctly detected fraud)")

# Basic Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nBasic Metrics:")
print(f"Accuracy:  {accuracy:.3f}")   # Misleading for imbalanced!
print(f"Precision: {precision:.3f}")  # Of flagged frauds, how many are real?
print(f"Recall:    {recall:.3f}")     # Of all frauds, how many we caught?
print(f"F1-Score:  {f1:.3f}")

# Advanced Metrics
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"\nAdvanced Metrics:")
print(f"ROC-AUC:   {roc_auc:.3f}")
print(f"PR-AUC:    {pr_auc:.3f}")  # Better for imbalanced!
print(f"MCC:       {mcc:.3f}")     # Best single metric!

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['Legitimate', 'Fraud'],
                           digits=3))

# ==================== 6. VISUALIZATIONS ====================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.1 Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=['Legitimate', 'Fraud'])
disp.plot(ax=axes[0, 0], cmap='Blues', values_format='d')
axes[0, 0].set_title('Confusion Matrix')

# 6.2 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0, 1].plot(fpr, tpr, label=f'Model (AUC = {roc_auc:.3f})', linewidth=2)
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate (Recall)')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 6.3 Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
baseline = y_test.sum() / len(y_test)
axes[1, 0].plot(recall_curve, precision_curve, label=f'Model (AP = {pr_auc:.3f})', linewidth=2)
axes[1, 0].axhline(baseline, color='k', linestyle='--', label=f'Baseline = {baseline:.3f}')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve (Better for Imbalanced!)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 6.4 Threshold Impact
thresholds_test = np.linspace(0, 1, 100)
precisions = []
recalls = []
f1_scores = []

for thresh in thresholds_test:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    
    if y_pred_thresh.sum() > 0:  # Ako ima predviđenih positives
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f = f1_score(y_test, y_pred_thresh, zero_division=0)
    else:
        p, r, f = 0, 0, 0
    
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

axes[1, 1].plot(thresholds_test, precisions, label='Precision', linewidth=2)
axes[1, 1].plot(thresholds_test, recalls, label='Recall', linewidth=2)
axes[1, 1].plot(thresholds_test, f1_scores, label='F1-Score', linewidth=2)
axes[1, 1].axvline(0.5, color='k', linestyle='--', alpha=0.3, label='Default (0.5)')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Metrics vs Threshold')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('fraud_detection_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 7. BUSINESS IMPACT ====================

print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

# Costs
cost_per_fraud_missed = 1000  # Average fraud amount
cost_per_false_alarm = 5      # Customer service investigation

total_fraud_cost = fn * cost_per_fraud_missed
total_false_alarm_cost = fp * cost_per_false_alarm
total_cost = total_fraud_cost + total_false_alarm_cost

print(f"\nMissed Frauds (FN): {fn}")
print(f"Cost of missed frauds: ${total_fraud_cost:,.2f}")
print(f"\nFalse Alarms (FP): {fp}")
print(f"Cost of false alarms: ${total_false_alarm_cost:,.2f}")
print(f"\nTOTAL COST: ${total_cost:,.2f}")

# What if we improve recall to 95%?
target_recall = 0.95
current_recall = tp / (tp + fn)
print(f"\nCurrent Recall: {current_recall:.1%}")
print(f"Target Recall: {target_recall:.1%}")

if current_recall < target_recall:
    frauds_to_catch = int((target_recall - current_recall) * (tp + fn))
    savings = frauds_to_catch * cost_per_fraud_missed
    print(f"Additional frauds to catch: {frauds_to_catch}")
    print(f"Potential savings: ${savings:,.2f}")
```

---

## Rezime - Classification Metrics

### Quick Reference Table:

| Metric | Formula | Best For | Range |
|--------|---------|----------|-------|
| **Accuracy** | (TP+TN)/Total | Balanced data | 0-1 |
| **Precision** | TP/(TP+FP) | FP costly (spam) | 0-1 |
| **Recall** | TP/(TP+FN) | FN costly (medical) | 0-1 |
| **F1-Score** | 2×P×R/(P+R) | Balance, imbalanced | 0-1 |
| **Specificity** | TN/(TN+FP) | True negative rate | 0-1 |
| **ROC-AUC** | Area under ROC | Overall, balanced | 0-1 |
| **PR-AUC** | Area under PR | **Imbalanced!** | 0-1 |
| **MCC** | Correlation | **Best for imbalanced** | -1 to 1 |
| **Log Loss** | Cross-entropy | Probability quality | 0-∞ |
| **Cohen's Kappa** | Adj. agreement | Inter-rater | -1 to 1 |

### Default Metrics By Problem:
```
Balanced Classification:
✅ Primary: Accuracy, F1-Score
✅ Secondary: ROC-AUC
✅ Curves: ROC Curve

Imbalanced Classification:
✅ Primary: F1-Score, PR-AUC
✅ Secondary: MCC, Per-class Recall
✅ Curves: PR Curve
❌ Avoid: Accuracy alone

Medical/Safety-Critical:
✅ Primary: Recall (minimize FN!)
✅ Secondary: F2-Score, Specificity
✅ Monitor: FN count explicitly

Marketing/Spam:
✅ Primary: Precision (minimize FP!)
✅ Secondary: F0.5-Score
✅ Monitor: FP rate

Multi-Class:
✅ Primary: Macro/Weighted F1
✅ Secondary: Per-class metrics
✅ Visual: Confusion Matrix
```

**Key Takeaway:** Ne postoji "best" metric - zavisi od problema! Za imbalanced data, **NIKAD ne koristi samo accuracy**. Kombinuj više metrika (F1 + PR-AUC + Confusion Matrix) za kompletnu sliku. Vizualizuj (ROC/PR curves) da razumeš trade-offs! 🎯