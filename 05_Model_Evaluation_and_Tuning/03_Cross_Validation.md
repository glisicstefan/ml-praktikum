# Cross-Validation

Cross-Validation je tehnika **evaluacije modela korišćenjem više train-test split-ova** da bi se dobila robusnija procena performansi. Umesto jednog split-a, podaci se dele na više načina i model se trenira/evaluira više puta.

**Zašto je cross-validation kritičan?**
- **Jedan train-test split može biti misleading** - "Sreća" u split-u može dati preoptimističnu ili pesimističnu procenu
- **Bolja procena generalizacije** - Više split-ova → stabilnija procena
- **Maksimalno iskorišćenje podataka** - Svaki sample se koristi i za trening i za testiranje
- **Variance u performansama** - Vidimo koliko model zavisi od konkretnog split-a
- **Small datasets** - Posebno važno kada imaš malo podataka

**VAŽNO:** Cross-validation se koristi TOKOM development-a za model selection i tuning. Finalna evaluacija ide na odvojenom test skupu!

---

## Problem sa Single Train-Test Split

### Demonstracija Problema:
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=200, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Multiple random splits
accuracies = []
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i  # Različit random_state!
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(accuracy)

# Analiza
print(f"Mean Accuracy: {np.mean(accuracies):.3f}")
print(f"Std Accuracy:  {np.std(accuracies):.3f}")
print(f"Min Accuracy:  {np.min(accuracies):.3f}")
print(f"Max Accuracy:  {np.max(accuracies):.3f}")
print(f"Range:         {np.max(accuracies) - np.min(accuracies):.3f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(accuracies, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(accuracies), color='red', linestyle='--', 
            linewidth=2, label=f'Mean = {np.mean(accuracies):.3f}')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracy (50 Random Splits)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(50), accuracies, 'o-', alpha=0.6)
plt.axhline(np.mean(accuracies), color='red', linestyle='--', 
            linewidth=2, label='Mean')
plt.fill_between(range(50), 
                  np.mean(accuracies) - np.std(accuracies),
                  np.mean(accuracies) + np.std(accuracies),
                  alpha=0.2, color='red', label='±1 Std')
plt.xlabel('Split Number')
plt.ylabel('Accuracy')
plt.title('Accuracy Variability Across Splits')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Problem: Accuracy varira između 0.75 i 0.95!
# Koji je "pravi" accuracy? Ne znamo sa samo jednim split-om!
```

**Problem:** Sa samo jednim train-test split-om, možemo dobiti "lucky" ili "unlucky" split koji ne predstavlja pravu performansu modela!

**Rešenje:** Cross-validation - koristimo VIŠE split-ova i prosečimo rezultate!

---

## 1. K-Fold Cross-Validation

**Najpoznatija CV metoda** - deli podatke na K fold-ova, svaki fold jednom test, ostali train.

### Kako Radi (K=5):
```
Original Dataset: [||||||||||||||||||||]

Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN] → Score 1
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN] → Score 2
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN] → Score 3
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN] → Score 4
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST] → Score 5

Final Score = Mean(Score 1, Score 2, Score 3, Score 4, Score 5)
```

**Karakteristike:**
- Svaki sample se koristi za testiranje **tačno jednom**
- Svaki sample se koristi za trening **K-1 puta**
- Train size = (K-1)/K × N
- Test size = 1/K × N

### Python Implementacija:
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)

# K-Fold (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("K-Fold Cross-Validation Results:")
print(f"Scores: {cv_scores}")
print(f"Mean:   {cv_scores.mean():.3f}")
print(f"Std:    {cv_scores.std():.3f}")
print(f"95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.3f}, "
      f"{cv_scores.mean() + 1.96*cv_scores.std():.3f}]")
```

### Manual K-Fold (Razumevanje Procesa):
```python
# Manual K-Fold implementation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    fold_scores.append(score)
    
    print(f"Fold {fold}: Train size={len(X_train)}, Test size={len(X_test)}, "
          f"Accuracy={score:.3f}")

print(f"\nMean CV Accuracy: {np.mean(fold_scores):.3f}")
print(f"Std:              {np.std(fold_scores):.3f}")
```

### Shuffle Parameter:
```python
# shuffle=True (PREPORUČENO!)
kf_shuffle = KFold(n_splits=5, shuffle=True, random_state=42)
# Podaci se MEŠAJU pre deljenja na fold-ove
# → Svaki fold ima random uzorak podataka

# shuffle=False
kf_no_shuffle = KFold(n_splits=5, shuffle=False)
# Podaci se NE mešaju
# → Prvi fold = prvi 20%, drugi fold = drugi 20%, itd.
# → Koristi SAMO za vremenske serije ili sortirane podatke!

# Preporuka: UVEK shuffle=True (osim za time series!)
```

---

## 2. Stratified K-Fold (Za Classification)

**K-Fold sa održavanjem proporcija klasa** u svakom fold-u - KRITIČAN za imbalanced data!

### Problem bez Stratification:
```python
# Imbalanced dataset (90% class 0, 10% class 1)
X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20,
    weights=[0.9, 0.1],  # 90/10 imbalance
    random_state=42
)

print(f"Overall distribution: {pd.Series(y_imb).value_counts(normalize=True)}")
# 0: 90%, 1: 10%

# Regular K-Fold (BAD for imbalanced!)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nRegular K-Fold (No Stratification):")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_imb), 1):
    y_train_fold = y_imb[train_idx]
    y_test_fold = y_imb[test_idx]
    
    train_dist = pd.Series(y_train_fold).value_counts(normalize=True)
    test_dist = pd.Series(y_test_fold).value_counts(normalize=True)
    
    print(f"Fold {fold}: Train 1={train_dist[1]:.2%}, Test 1={test_dist[1]:.2%}")
    # Distribucija VARIRA! Nije konzistentna!

# Fold 1: Train 1=10.00%, Test 1=10.00%
# Fold 2: Train 1=9.75%,  Test 1=11.00%  ← Nekonzistentno!
# Fold 3: Train 1=10.25%, Test 1=9.00%
# ...
```

### Sa Stratification:
```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold (GOOD for imbalanced!)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nStratified K-Fold:")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_imb, y_imb), 1):
    y_train_fold = y_imb[train_idx]
    y_test_fold = y_imb[test_idx]
    
    train_dist = pd.Series(y_train_fold).value_counts(normalize=True)
    test_dist = pd.Series(y_test_fold).value_counts(normalize=True)
    
    print(f"Fold {fold}: Train 1={train_dist[1]:.2%}, Test 1={test_dist[1]:.2%}")
    # Distribucija KONZISTENTNA! ✅

# Fold 1: Train 1=10.00%, Test 1=10.00%
# Fold 2: Train 1=10.00%, Test 1=10.00%  ← Konzistentno!
# Fold 3: Train 1=10.00%, Test 1=10.00%
# ...
```

### Cross-Validation sa Stratification:
```python
# UVEK koristi StratifiedKFold za classification!
cv_scores_stratified = cross_val_score(
    model, X_imb, y_imb, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1'  # F1 bolje za imbalanced nego accuracy
)

print(f"Stratified CV F1: {cv_scores_stratified.mean():.3f} ± {cv_scores_stratified.std():.3f}")
```

### Multi-Class Stratification:
```python
# Stratified K-Fold radi i za multi-class!
from sklearn.datasets import load_iris

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

print("Class distribution:")
print(pd.Series(y_iris).value_counts(normalize=True))
# Class 0: 33.3%, Class 1: 33.3%, Class 2: 33.3%

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_iris, y_iris), 1):
    y_test_fold = y_iris[test_idx]
    test_dist = pd.Series(y_test_fold).value_counts(normalize=True).sort_index()
    print(f"Fold {fold}: {test_dist.to_dict()}")
    # Sve tri klase imaju istu proporciju u svakom fold-u! ✅
```

**Kada Koristiti:**

✅ **UVEK za classification probleme** (balanced ili imbalanced)  
✅ **Multi-class classification**  
✅ **Imbalanced data** (KRITIČNO!)

---

## 3. Leave-One-Out Cross-Validation (LOOCV)

**Ekstremni K-Fold** gde je K = N (broj samples) - svaki sample jednom test.

### Kako Radi:
```
Dataset sa N=5 samples:

Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]

N fold-ova = N treniranja!
```

### Python:
```python
from sklearn.model_selection import LeaveOneOut

# SAMO za VEOMA mali dataset (< 100 samples)
X_small = np.random.randn(30, 5)
y_small = np.random.choice([0, 1], 30)

# LOOCV
loo = LeaveOneOut()

print(f"Number of folds: {loo.get_n_splits(X_small)}")  # 30!

# Cross-validation
model = LogisticRegression(max_iter=1000)
cv_scores_loo = cross_val_score(model, X_small, y_small, cv=loo, scoring='accuracy')

print(f"LOOCV Accuracy: {cv_scores_loo.mean():.3f}")
print(f"LOOCV Std:      {cv_scores_loo.std():.3f}")  # Vrlo niska variance
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Maksimalan training data** - Koristi N-1 samples za trening
- ✅ **Deterministic** - Nema randomness u split-u
- ✅ **Niska bias** - Skoro ceo dataset se koristi za trening

**Mane:**
- ❌ **VEOMA SPORO** - N treniranja! (za N=10,000 → 10,000 modela!)
- ❌ **Visoka variance** - Test set ima samo 1 sample
- ❌ **Ne skalira** - Samo za mali dataset (< 100)

### Kada Koristiti:

✅ **SAMO za veoma mali dataset** (< 100 samples)  
✅ **Kada je svaki sample dragocen**  
✅ **Kada imaš vremena i računarske moći**

❌ **NE za velike dataset-e** (koristiti 5-Fold ili 10-Fold)

---

## 4. Leave-P-Out Cross-Validation

**Generalizacija LOOCV** - P samples u test setu umesto 1.

### Python:
```python
from sklearn.model_selection import LeavePOut

# Leave-2-Out (svaka kombinacija od 2 samples je test)
lpo = LeavePOut(p=2)

X_small = np.random.randn(10, 5)
y_small = np.random.choice([0, 1], 10)

n_splits = lpo.get_n_splits(X_small)
print(f"Number of folds (Leave-2-Out): {n_splits}")  # C(10,2) = 45

# Ovo EKSPLODIRA za veće N!
# Leave-2-Out za N=100 → C(100,2) = 4,950 fold-ova
# Leave-3-Out za N=100 → C(100,3) = 161,700 fold-ova! 🤯
```

**Kada Koristiti:**

✅ **Ekstremno retko** - Skoro nikad u praksi  
❌ **Previše skupo** - Broj fold-ova je kombinatorijski

**Preporuka:** Koristi K-Fold umesto!

---

## 5. Shuffle Split (Random Permutations)

**Random train-test splits sa kontrolom nad size-om** - ne garantuje da svaki sample bude u test skupu.

### Kako Radi:
```python
from sklearn.model_selection import ShuffleSplit

# ShuffleSplit
ss = ShuffleSplit(
    n_splits=10,        # Broj random splits
    test_size=0.2,      # 20% test
    random_state=42
)

X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=ss, scoring='accuracy')

print(f"ShuffleSplit CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Provera - da li svaki sample bio u test?
test_counts = np.zeros(len(X))
for train_idx, test_idx in ss.split(X):
    test_counts[test_idx] += 1

print(f"\nMin times in test: {test_counts.min():.0f}")
print(f"Max times in test: {test_counts.max():.0f}")
print(f"Mean times in test: {test_counts.mean():.1f}")
# Ne garantuje da svaki sample bude u test skupu!
```

### ShuffleSplit vs K-Fold:
```python
# K-Fold: Svaki sample u test TAČNO JEDNOM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_counts_kf = np.zeros(len(X))
for train_idx, test_idx in kf.split(X):
    test_counts_kf[test_idx] += 1
print(f"K-Fold - All samples in test: {(test_counts_kf == 1).all()}")  # True!

# ShuffleSplit: Random samples u test, možda više puta ili nikad
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
test_counts_ss = np.zeros(len(X))
for train_idx, test_idx in ss.split(X):
    test_counts_ss[test_idx] += 1
print(f"ShuffleSplit - All samples in test: {(test_counts_ss >= 1).all()}")  # Može biti False!
```

### Stratified Shuffle Split:
```python
from sklearn.model_selection import StratifiedShuffleSplit

# Stratified version - održava proporcije klasa
sss = StratifiedShuffleSplit(
    n_splits=10,
    test_size=0.2,
    random_state=42
)

cv_scores_sss = cross_val_score(model, X_imb, y_imb, cv=sss, scoring='f1')
print(f"Stratified ShuffleSplit F1: {cv_scores_sss.mean():.3f} ± {cv_scores_sss.std():.3f}")
```

### Kada Koristiti:

✅ **Želiš kontrolu nad train/test ratio**  
✅ **Želiš više iteracija od K-Fold**  
✅ **Brzi eksperimenti** - Manje fold-ova od LOOCV ali više od K-Fold

❌ **Ne garantuje da svaki sample bude testiran**

---

## 6. Time Series Cross-Validation

**Za temporalne podatke** - mora poštovati vremenski redosled (ne sme shuffle!)

### Time Series Split:
```python
from sklearn.model_selection import TimeSeriesSplit

# Time series data (sortiran po datumu!)
n_samples = 100
X_ts = np.arange(n_samples).reshape(-1, 1)
y_ts = np.sin(X_ts.ravel() / 10) + np.random.randn(n_samples) * 0.1

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

print("Time Series Cross-Validation Splits:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts), 1):
    print(f"Fold {fold}:")
    print(f"  Train: indices {train_idx[0]:3d} to {train_idx[-1]:3d} (size={len(train_idx):3d})")
    print(f"  Test:  indices {test_idx[0]:3d} to {test_idx[-1]:3d} (size={len(test_idx):3d})")

# Fold 1:
#   Train: indices   0 to  19 (size= 20)
#   Test:  indices  20 to  39 (size= 20)
# Fold 2:
#   Train: indices   0 to  39 (size= 40)
#   Test:  indices  40 to  59 (size= 20)
# Fold 3:
#   Train: indices   0 to  59 (size= 60)
#   Test:  indices  60 to  79 (size= 20)
# ...

# Train UVEK ide PRE test-a (vremenski redosled!)
```

### Visualizing Time Series Splits:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
    # Create train/test indicator
    indices = np.zeros(len(X_ts))
    indices[train_idx] = 1  # Train
    indices[test_idx] = 2   # Test
    
    # Plot
    axes[fold].scatter(range(len(X_ts)), indices, c=indices, cmap='RdYlGn', 
                       s=50, alpha=0.8)
    axes[fold].set_yticks([0, 1, 2])
    axes[fold].set_yticklabels(['', 'Train', 'Test'])
    axes[fold].set_ylabel(f'Fold {fold+1}')
    axes[fold].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time Index')
plt.suptitle('Time Series Cross-Validation Splits', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### ⚠️ KRITIČNO za Time Series:
```python
# ❌ NIKAD NE RADI OVO za time series:
kf = KFold(n_splits=5, shuffle=True)  # shuffle=True je LOŠE za TS!
# Koristi budućnost za trening → Data leakage!

# ✅ UVEK koristi TimeSeriesSplit ili shuffle=False
tscv = TimeSeriesSplit(n_splits=5)
# ILI
kf_ts = KFold(n_splits=5, shuffle=False)  # Održava vremenski redosled
```

### Kada Koristiti:

✅ **UVEK za time series data** (stock prices, sales, weather)  
✅ **Sequentially ordered data**  
✅ **Kada vreme ima značenje**

---

## 7. Group K-Fold (Za Grouped Data)

**Za podatke sa grupama** - cela grupa mora biti u jednom skupu (train ILI test).

### Problem bez Group Awareness:
```python
# Dataset sa grupama (npr. pacijenti sa više merenja)
patients = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4])
X_groups = np.random.randn(10, 5)
y_groups = np.random.choice([0, 1], 10)

# Regular K-Fold (BAD for grouped data!)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

print("Regular K-Fold (Ignores Groups):")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_groups), 1):
    train_patients = set(patients[train_idx])
    test_patients = set(patients[test_idx])
    
    # Provera overlap-a
    overlap = train_patients & test_patients
    print(f"Fold {fold}: Train patients={sorted(train_patients)}, "
          f"Test patients={sorted(test_patients)}, "
          f"Overlap={sorted(overlap) if overlap else 'None'}")

# Fold 1: Train patients=[1, 2, 4], Test patients=[1, 3], Overlap=[1]  ← BAD!
# Patient 1 je I u train I u test → Data leakage!
```

### Sa Group K-Fold:
```python
from sklearn.model_selection import GroupKFold

# GroupKFold
gkf = GroupKFold(n_splits=3)

print("\nGroup K-Fold (Respects Groups):")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X_groups, y_groups, groups=patients), 1):
    train_patients = set(patients[train_idx])
    test_patients = set(patients[test_idx])
    
    # Provera overlap-a
    overlap = train_patients & test_patients
    print(f"Fold {fold}: Train patients={sorted(train_patients)}, "
          f"Test patients={sorted(test_patients)}, "
          f"Overlap={sorted(overlap) if overlap else 'None'}")

# Fold 1: Train patients=[1, 2, 4], Test patients=[3], Overlap=None  ✅
# Fold 2: Train patients=[1, 3, 4], Test patients=[2], Overlap=None  ✅
# Fold 3: Train patients=[2, 3], Test patients=[1, 4], Overlap=None  ✅

# NEMA overlap-a! Svaki pacijent je U JEDNOM skupu!
```

### Cross-Validation sa Groups:
```python
# Cross-validation sa groups
cv_scores_grouped = cross_val_score(
    model, X_groups, y_groups,
    cv=GroupKFold(n_splits=3),
    groups=patients  # MORA proslediti groups!
)

print(f"Group CV: {cv_scores_grouped.mean():.3f} ± {cv_scores_grouped.std():.3f}")
```

### Leave-One-Group-Out:
```python
from sklearn.model_selection import LeaveOneGroupOut

# Leave-One-Group-Out - Svaka grupa jednom test
logo = LeaveOneGroupOut()

print(f"\nLeave-One-Group-Out:")
print(f"Number of folds: {logo.get_n_splits(groups=patients)}")  # 4 (broj unique grupa)

for fold, (train_idx, test_idx) in enumerate(logo.split(X_groups, y_groups, groups=patients), 1):
    test_group = patients[test_idx][0]
    train_groups = sorted(set(patients[train_idx]))
    
    print(f"Fold {fold}: Train groups={train_groups}, Test group=[{test_group}]")

# Fold 1: Train groups=[2, 3, 4], Test group=[1]
# Fold 2: Train groups=[1, 3, 4], Test group=[2]
# Fold 3: Train groups=[1, 2, 4], Test group=[3]
# Fold 4: Train groups=[1, 2, 3], Test group=[4]
```

### Kada Koristiti:

✅ **Data ima groups** (pacijenti, korisnici, prodavnice)  
✅ **Multiple measurements per entity**  
✅ **Avoiding data leakage** kroz grupe

**Primeri:**
- **Medical data** - Pacijenti sa više testova
- **User behavior** - Korisnici sa više sesija
- **Image data** - Različite slike istog objekta

---

## 8. Nested Cross-Validation

**CV unutar CV** - za hyperparameter tuning BEZ data leakage!

### Problem sa Inner Tuning bez Nested CV:
```python
# ❌ LOŠE - Tuning na istom CV kao i evaluacija
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 500, 1000]}

# GridSearch SA CV
grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,  # Inner CV za tuning
    scoring='accuracy'
)

# Evaluate SA istim CV
cv_scores = cross_val_score(grid_search, X, y, cv=5, scoring='accuracy')
# Problem: Isti CV fold-ovi se koriste za tuning I evaluaciju!
# → Optimističan bias! Model je "video" test data tokom tuning-a!

print(f"CV (with tuning leak): {cv_scores.mean():.3f}")  # Previše optimističan!
```

### Sa Nested CV (PRAVILNO!):
```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Nested CV structure:
# Outer CV: Evaluacija
# Inner CV: Hyperparameter tuning

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Model sa hyperparameter search
param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 500, 1000]}
grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=inner_cv,  # Inner CV (3-fold)
    scoring='accuracy'
)

# Outer CV evaluacija
nested_cv_scores = cross_val_score(
    grid_search, X, y,
    cv=outer_cv,  # Outer CV (5-fold)
    scoring='accuracy'
)

print(f"Nested CV: {nested_cv_scores.mean():.3f} ± {nested_cv_scores.std():.3f}")

# Proces za svaki outer fold:
# 1. Split data na outer train i outer test
# 2. Na outer train: GridSearch sa inner CV (3-fold) → Best params
# 3. Train model sa best params na outer train
# 4. Evaluate na outer test
# 5. Repeat za sve outer folds
```

### Manual Nested CV (Razumevanje):
```python
from sklearn.model_selection import GridSearchCV

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

outer_scores = []
best_params_per_fold = []

for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X), 1):
    print(f"\nOuter Fold {fold}:")
    
    # Outer split
    X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
    y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
    
    # Inner CV: Hyperparameter tuning SAMO na outer train
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid={'C': [0.1, 1, 10]},
        cv=inner_cv,
        scoring='accuracy'
    )
    grid_search.fit(X_outer_train, y_outer_train)
    
    # Best params from inner CV
    best_params = grid_search.best_params_
    best_params_per_fold.append(best_params)
    print(f"  Best params (inner CV): {best_params}")
    
    # Evaluate best model on outer test
    outer_score = grid_search.score(X_outer_test, y_outer_test)
    outer_scores.append(outer_score)
    print(f"  Outer test score: {outer_score:.3f}")

print(f"\n{'='*60}")
print(f"Nested CV Mean: {np.mean(outer_scores):.3f}")
print(f"Nested CV Std:  {np.std(outer_scores):.3f}")
print(f"\nBest params varied across folds: {best_params_per_fold}")
# Different outer folds mogu imati različite "best" params!
```

### Kada Koristiti:

✅ **UVEK za proper hyperparameter tuning evaluation**  
✅ **Reporting unbiased performance**  
✅ **Model selection sa tuning**

❌ **NE za production training** (koristi običan GridSearch na SVIM podacima)

**Note:** Nested CV je SPOR (outer_folds × inner_folds × param_combinations treniranja!)

---

## 9. cross_val_score vs cross_validate

### cross_val_score (Simple):
```python
from sklearn.model_selection import cross_val_score

# Single metric
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='accuracy'  # Samo jedan metric
)

print(f"Scores: {scores}")
print(f"Mean:   {scores.mean():.3f}")
```

### cross_validate (Advanced):
```python
from sklearn.model_selection import cross_validate

# Multiple metrics + timing
cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],  # Više metrika!
    return_train_score=True,  # Vrati i train score (check overfitting)
    return_estimator=True     # Vrati trenirane modele
)

print("Cross-Validate Results:")
print(f"Test Accuracy:  {cv_results['test_accuracy'].mean():.3f}")
print(f"Test Precision: {cv_results['test_precision'].mean():.3f}")
print(f"Test Recall:    {cv_results['test_recall'].mean():.3f}")
print(f"Test F1:        {cv_results['test_f1'].mean():.3f}")

print(f"\nTrain Accuracy: {cv_results['train_accuracy'].mean():.3f}")
# Provera overfitting: Train vs Test gap

print(f"\nFit Time:   {cv_results['fit_time'].mean():.3f}s")
print(f"Score Time: {cv_results['score_time'].mean():.3f}s")

# Access individual trained models
trained_models = cv_results['estimator']
print(f"\nTrained {len(trained_models)} models")
```

### Overfitting Detection:
```python
# Provera overfitting sa cross_validate
cv_results = cross_validate(
    model, X, y, cv=5,
    scoring='accuracy',
    return_train_score=True
)

train_scores = cv_results['train_accuracy']
test_scores = cv_results['test_accuracy']

print(f"Train Accuracy: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
print(f"Test Accuracy:  {test_scores.mean():.3f} ± {test_scores.std():.3f}")
print(f"Gap:            {(train_scores.mean() - test_scores.mean()):.3f}")

if (train_scores.mean() - test_scores.mean()) > 0.1:
    print("⚠️  WARNING: Likely overfitting! (Train >> Test)")
elif (train_scores.mean() - test_scores.mean()) < -0.05:
    print("⚠️  WARNING: Data leakage? (Test > Train)")
else:
    print("✅ Model seems well-balanced")
```

---

## 10. Custom Cross-Validation Splitters

### Creating Custom Splitter:
```python
from sklearn.model_selection import BaseCrossValidator

class CustomSplit(BaseCrossValidator):
    """
    Custom CV splitter - primer: 60/20/20 train/val/test repeated splits.
    """
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Set seed za reproducibility
            np.random.seed(self.random_state + i if self.random_state else None)
            
            # Shuffle
            np.random.shuffle(indices)
            
            # Split 60/20/20
            train_end = int(0.6 * n_samples)
            val_end = int(0.8 * n_samples)
            
            train_idx = indices[:train_end]
            test_idx = indices[val_end:]  # Koristimo validation kao "test" za CV
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Korišćenje
custom_cv = CustomSplit(n_splits=5, random_state=42)
scores = cross_val_score(model, X, y, cv=custom_cv, scoring='accuracy')
print(f"Custom CV: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Best Practices

### ✅ DO:

**1. Izbor Pravog CV za Problem:**
```python
# Classification (balanced)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Classification (imbalanced)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time series
cv = TimeSeriesSplit(n_splits=5)

# Grouped data
cv = GroupKFold(n_splits=5)

# Small dataset
cv = LeaveOneOut()
```

**2. Set random_state za Reproducibility:**
```python
# ✅ DOBRO
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ❌ LOŠE
cv = KFold(n_splits=5, shuffle=True)  # Različiti rezultati svaki put!
```

**3. Report Mean AND Std:**
```python
scores = cross_val_score(model, X, y, cv=5)
print(f"CV: {scores.mean():.3f} ± {scores.std():.3f}")
# Std pokazuje variance/stability!
```

**4. Check Train-Test Gap:**
```python
cv_results = cross_validate(model, X, y, cv=5, return_train_score=True)
train_mean = cv_results['train_score'].mean()
test_mean = cv_results['test_score'].mean()

if train_mean - test_mean > 0.1:
    print("⚠️ Overfitting detected!")
```

**5. Use Appropriate K:**
```python
# Small dataset (< 100): K=10 ili LOOCV
# Medium dataset (100-10k): K=5 ili K=10
# Large dataset (> 10k): K=3 ili K=5

# Default: K=5 (good trade-off)
```

### ❌ DON'T:

**1. Ne Shuffle Time Series:**
```python
# ❌ LOŠE
cv = KFold(n_splits=5, shuffle=True)  # Za time series!

# ✅ DOBRO
cv = TimeSeriesSplit(n_splits=5)
```

**2. Ne Zaboravi Stratify za Imbalanced:**
```python
# ❌ LOŠE
cv = KFold(n_splits=5)  # Za imbalanced classification

# ✅ DOBRO
cv = StratifiedKFold(n_splits=5, shuffle=True)
```

**3. Ne Koristi LOOCV za Velike Datasets:**
```python
# ❌ LOŠE - Traje VEČNOST!
cv = LeaveOneOut()  # Za 10,000 samples = 10,000 treniranja!

# ✅ DOBRO
cv = KFold(n_splits=5, shuffle=True)
```

**4. Ne Ignoriši Groups:**
```python
# ❌ LOŠE - Data leakage kroz grupe!
cv = KFold(n_splits=5)  # Za grouped data

# ✅ DOBRO
cv = GroupKFold(n_splits=5)
```

**5. Ne Koristi Isti CV za Tuning i Evaluation:**
```python
# ❌ LOŠE
grid_search = GridSearchCV(model, params, cv=5)
scores = cross_val_score(grid_search, X, y, cv=5)  # Isti CV!

# ✅ DOBRO - Nested CV
outer_cv = KFold(5)
inner_cv = KFold(3)
grid_search = GridSearchCV(model, params, cv=inner_cv)
scores = cross_val_score(grid_search, X, y, cv=outer_cv)
```

---

## Complete Example - Model Comparison sa CV
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# ==================== 1. LOAD DATA ====================
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y, return_counts=True)}")

# ==================== 2. MODELS ====================
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=42))
    ])
}

# ==================== 3. CROSS-VALIDATION ====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Multiple metrics
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=True,
        n_jobs=-1
    )
    
    results[name] = cv_results
    
    # Print results
    print(f"  Test Accuracy:  {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"  Test Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
    print(f"  Test Recall:    {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
    print(f"  Test F1:        {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print(f"  Test ROC-AUC:   {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
    
    # Overfitting check
    train_acc = cv_results['train_accuracy'].mean()
    test_acc = cv_results['test_accuracy'].mean()
    gap = train_acc - test_acc
    print(f"  Train-Test Gap: {gap:.3f}", end='')
    if gap > 0.1:
        print(" ⚠️ Possible overfitting")
    else:
        print(" ✅")

# ==================== 4. VISUALIZATION ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 3, idx % 3]
    
    # Extract scores for each model
    data = []
    labels = []
    for name in models.keys():
        scores = results[name][f'test_{metric}']
        data.append(scores)
        labels.append(name)
    
    # Box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Distribution (5-Fold CV)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(labels, rotation=15, ha='right')

# Train vs Test comparison
ax = axes[1, 2]
model_names = list(models.keys())
train_means = [results[name]['train_accuracy'].mean() for name in model_names]
test_means = [results[name]['test_accuracy'].mean() for name in model_names]

x = np.arange(len(model_names))
width = 0.35

ax.bar(x - width/2, train_means, width, label='Train', alpha=0.8)
ax.bar(x + width/2, test_means, width, label='Test', alpha=0.8)

ax.set_ylabel('Accuracy')
ax.set_title('Train vs Test Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cv_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 5. BEST MODEL ====================
best_model_name = max(results.keys(), 
                      key=lambda k: results[k]['test_f1'].mean())
best_f1 = results[best_model_name]['test_f1'].mean()

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print(f"F1-Score: {best_f1:.3f}")
print("="*60)
```

---

## Rezime - Cross-Validation

### Quick Reference:

| CV Method | When to Use | K value | Train Size | Pros | Cons |
|-----------|-------------|---------|------------|------|------|
| **K-Fold** | Default | 5-10 | (K-1)/K | Fast, low variance | Some samples not used |
| **Stratified K-Fold** | **Imbalanced classification** | 5-10 | (K-1)/K | **Best for classification** | Classification only |
| **LOOCV** | Small dataset | N | (N-1)/N | Maximum data use | Very slow, high variance |
| **TimeSeriesSplit** | **Time series** | 5 | Growing | **Respects time order** | Not for non-temporal |
| **GroupKFold** | **Grouped data** | 3-5 | Variable | **No group leakage** | Need group info |
| **ShuffleSplit** | Custom ratio | 10+ | Custom | Flexible | No guarantee all tested |

### Default Strategy:
```
Classification (balanced):
✅ StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Classification (imbalanced):
✅ StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

Regression:
✅ KFold(n_splits=5, shuffle=True, random_state=42)

Time Series:
✅ TimeSeriesSplit(n_splits=5)

Grouped Data:
✅ GroupKFold(n_splits=5)

Small Dataset (< 100):
✅ LeaveOneOut() ili KFold(n_splits=10)

Large Dataset (> 10k):
✅ KFold(n_splits=3) ili ShuffleSplit(n_splits=5, test_size=0.2)
```

**Key Takeaway:** Nikad ne evaluiraj model sa samo jednim train-test split-om! Cross-validation daje robusniju, stabilniju procenu performansi. Za classification UVEK koristi Stratified K-Fold. Za time series NIKAD ne shuffle. Report mean AND std - variance je bitna! 🎯