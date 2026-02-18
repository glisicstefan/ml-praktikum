# Data Splitting

Data Splitting je proces **deljenja dataseta na različite skupove** za treniranje, validaciju i testiranje modela. Ovo je **KRITIČAN korak** za procenu koliko dobro model generalizuje na neviđene podatke.

**Zašto je splitting bitan?**
- **Procena generalizacije** - Kako model radi na novim, neviđenim podacima?
- **Sprečavanje overfitting-a** - Model ne sme "pamtiti" test podatke
- **Hyperparameter tuning** - Validacioni skup za optimizaciju parametara
- **Unbiased evaluacija** - Test skup daje realan skor performansi

**KRITIČNO:** Splitting se MOŽE raditi na početku (pre svih preprocessing koraka) ili nakon feature engineering-a, ali UVEK pre finalnog treniranja!

---

## Redosled Operacija - Dve Strategije

### Strategija 1: Split na Početku (Preporučeno!) ✅
```
1. Data Collection
2. Data Splitting  ← OVDE! (na raw podacima)
3. Data Cleaning (fit na train)
4. EDA (samo train)
5. Data Transformation (fit na train)
6. Encoding (fit na train)
7. Feature Scaling (fit na train)
8. Model Training (samo train)
9. Evaluation (samo test)
```

**Prednosti:**
- Garantuje NO data leakage
- Test skup je "locked" - nikad se ne dodiruje
- Preprocessing parametri (mean, median, encoding mappings) dolaze SAMO iz train

### Strategija 2: Split Posle Feature Engineering
```
1. Data Collection
2. Data Cleaning
3. EDA
4. Feature Engineering (kreiranje features)
5. Data Splitting  ← OVDE!
6. Encoding (fit na train)
7. Feature Scaling (fit na train)
8. Model Training
9. Evaluation
```

**Prednosti:**
- Feature engineering na celom dataset-u (lakše)
- Manje komplikovano za handlovati

**Oprez:** MORA paziti da feature engineering ne koristi target!

---

## 1. Train-Test Split

**Osnovni split** - deli dataset na **train** (za učenje) i **test** (za evaluaciju).

### Standardni Ratio:
- **80/20** - 80% train, 20% test (najčešće)
- **70/30** - 70% train, 30% test
- **90/10** - 90% train, 10% test (veliki dataset)

### Python Implementacija:
```python
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Dataset
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'target': np.random.choice([0, 1], 100)
})

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Basic train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% test
    random_state=42     # Reproducibility
)

print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
# Train size: 80 (80.0%)
# Test size: 20 (20.0%)

# Provera distribucije target-a
print(f"\nTrain target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nTest target distribution:\n{y_test.value_counts(normalize=True)}")
```

### Parametri train_test_split:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # Procenat ili apsolutni broj za test
    train_size=None,      # Ako je None, automatski = 1 - test_size
    random_state=42,      # Seed za reproducibility
    shuffle=True,         # Da li mešati podatke pre split-a?
    stratify=y            # Održava proporciju klasa (za classification)
)

# test_size opcije:
test_size=0.2      # 20% podataka u test
test_size=0.3      # 30% podataka u test
test_size=100      # Tačno 100 samples u test (apsolutni broj)

# random_state:
random_state=42    # Isti split svaki put (reproducible)
random_state=None  # Različit split svaki put
```

### Shuffle Parameter:
```python
# SA shuffle=True (default)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    shuffle=True,      # Podaci se mešaju PRE split-a
    random_state=42
)

# BEZ shuffle (za time series!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    shuffle=False      # Zadržava redosled (prvi 80% train, zadnjih 20% test)
)
# ⚠️ Koristi shuffle=False SAMO za temporalne podatke!
```

---

## 2. Stratified Split (Za Classification)

**Stratified split** održava **istu proporciju klasa** u train i test skupu.

### Zašto je Bitan?
```python
# Dataset sa imbalanced classes
y = pd.Series([0]*90 + [1]*10)  # 90% class 0, 10% class 1

# BEZ stratify - random split može biti loš!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("WITHOUT stratify:")
print(f"Train: {y_train.value_counts(normalize=True)}")
# Može biti: 0: 88%, 1: 12% (nije reprezentativno!)
print(f"Test: {y_test.value_counts(normalize=True)}")
# Može biti: 0: 95%, 1: 5% (loše za evaluaciju!)

# SA stratify - garantuje proporcije!
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y           # ✅ Održava 90/10 u oba skupa!
)

print("\nWITH stratify:")
print(f"Train: {y_train_strat.value_counts(normalize=True)}")
# 0: 90%, 1: 10% ✅
print(f"Test: {y_test_strat.value_counts(normalize=True)}")
# 0: 90%, 1: 10% ✅
```

### Kada Koristiti Stratify?

✅ **UVEK za classification sa imbalanced classes**  
✅ **Multi-class classification**  
✅ **Kada je test skup mali** (< 100 samples)  
✅ **Binary classification** (preporučeno)

❌ **NE za regression** (stratify ne radi za continuous targets)

### Multi-Class Stratification:
```python
# Multi-class target
y_multi = pd.Series(['A']*50 + ['B']*30 + ['C']*20)  # 50%, 30%, 20%

X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi, 
    test_size=0.2, 
    stratify=y_multi,    # Održava sve tri proporcije!
    random_state=42
)

print("Train distribution:")
print(y_train.value_counts(normalize=True))
# A: 50%, B: 30%, C: 20% ✅

print("\nTest distribution:")
print(y_test.value_counts(normalize=True))
# A: 50%, B: 30%, C: 20% ✅
```

---

## 3. Train-Validation-Test Split

**Tri skupa** - train (treniranje), validation (hyperparameter tuning), test (finalna evaluacija).

### Zašto Tri Skupa?
```
Train Set (60-70%)      → Treniranje modela
    ↓
Validation Set (15-20%) → Hyperparameter tuning, model selection
    ↓
Test Set (15-20%)       → FINALNA evaluacija (jednom!)
```

**Validation skup** sprečava **overfitting na test skupu** tokom tuning-a!

### Python Implementacija:
```python
# Metoda 1: Dva split-a
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.4,       # 40% za validation + test
    random_state=42,
    stratify=y
)

# Split temp na validation i test (50/50 od 40% = 20% svaki)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,       # 50% od temp = 20% od total
    random_state=42,
    stratify=y_temp
)

print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
# Train: 600 (60.0%)
# Validation: 200 (20.0%)
# Test: 200 (20.0%)

# Metoda 2: Manual indexing
from sklearn.model_selection import train_test_split

# First split: 60% train, 40% temp
train_idx, temp_idx = train_test_split(
    np.arange(len(X)), 
    test_size=0.4, 
    random_state=42,
    stratify=y
)

# Second split: 50% validation, 50% test (od temp)
val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.5, 
    random_state=42,
    stratify=y.iloc[temp_idx]
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]
```

### Korišćenje Validation Skupa:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hyperparameter tuning sa validation skupom
best_score = 0
best_params = {}

for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, 20]:
        # Train na train
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluacija na VALIDATION (ne test!)
        val_score = accuracy_score(y_val, model.predict(X_val))
        
        if val_score > best_score:
            best_score = val_score
            best_params = {
                'n_estimators': n_estimators, 
                'max_depth': max_depth
            }

print(f"Best params: {best_params}")
print(f"Best validation score: {best_score:.3f}")

# Finalni model sa najboljim parametrima
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# FINALNA evaluacija na test (JEDNOM!)
test_score = accuracy_score(y_test, final_model.predict(X_test))
print(f"Final test score: {test_score:.3f}")
```

---

## 4. Cross-Validation (K-Fold)

**Najrobusniji pristup** - deli podatke na **K fold-ova**, svaki fold jednom test, ostali train.

### Kako K-Fold Radi?
```
5-Fold Cross-Validation:

Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN] → Score 1
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN] → Score 2
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN] → Score 3
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN] → Score 4
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST] → Score 5

Final Score = Average(Score 1, Score 2, Score 3, Score 4, Score 5)
```

### Python Implementacija:
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold Cross-Validation (K=5)
cv_scores = cross_val_score(
    model, X, y, 
    cv=5,              # 5 folds
    scoring='accuracy' # Metric
)

print(f"CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std: {cv_scores.std():.3f}")
# CV Scores: [0.85 0.87 0.83 0.86 0.84]
# Mean: 0.850
# Std: 0.014

# Manual KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for train_idx, val_idx in kf.split(X):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_val_fold, y_val_fold)
    scores.append(score)

print(f"Manual CV Mean: {np.mean(scores):.3f}")
```

### Koliko Folds?

| K | Train Size | Validation Size | Variance | Bias | Use Case |
|---|------------|-----------------|----------|------|----------|
| **K=3** | 67% | 33% | High | Low | Veliki dataset, brz test |
| **K=5** | 80% | 20% | Medium | Medium | **Default choice** ⭐ |
| **K=10** | 90% | 10% | Low | Medium-High | Standardno, robusto |
| **K=20** | 95% | 5% | Very Low | High | Mali dataset, duže traje |
| **K=N (LOOCV)** | (N-1)/N | 1 sample | Minimum | Maximum | Ekstremno mali dataset |

**Preporuka:** K=5 ili K=10 (najbolji trade-off)

---

## 5. Stratified K-Fold (Za Imbalanced Data)

**K-Fold sa održavanjem proporcija klasa** u svakom fold-u.

### Python Implementacija:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Imbalanced target
y_imbalanced = pd.Series([0]*90 + [1]*10)  # 90/10

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_imbalanced), 1):
    y_train_fold = y_imbalanced.iloc[train_idx]
    y_val_fold = y_imbalanced.iloc[val_idx]
    
    print(f"\nFold {fold}:")
    print(f"Train: {y_train_fold.value_counts(normalize=True).to_dict()}")
    print(f"Val: {y_val_fold.value_counts(normalize=True).to_dict()}")
    # Svaki fold ima ~90/10 ratio! ✅

# Cross-validation sa stratification
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(
    model, X, y_imbalanced, 
    cv=skf,              # Stratified KFold objekat
    scoring='f1'         # F1 score za imbalanced
)

print(f"\nStratified CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### Regular KFold vs Stratified KFold:
```python
from sklearn.model_selection import KFold, StratifiedKFold

# Regular KFold - može imati različite proporcije
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"Fold {fold} - Val class 1: {y_imbalanced.iloc[val_idx].sum()}")
# Fold 1 - Val class 1: 3
# Fold 2 - Val class 1: 1  ← Nekonzistentno!
# Fold 3 - Val class 1: 2
# Fold 4 - Val class 1: 2
# Fold 5 - Val class 1: 2

# Stratified KFold - konzistentne proporcije
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_imbalanced), 1):
    print(f"Fold {fold} - Val class 1: {y_imbalanced.iloc[val_idx].sum()}")
# Fold 1 - Val class 1: 2  ← Konzistentno! ✅
# Fold 2 - Val class 1: 2
# Fold 3 - Val class 1: 2
# Fold 4 - Val class 1: 2
# Fold 5 - Val class 1: 2
```

---

## 6. Time Series Split

**Za temporalne podatke** - ne sme biti shuffle, mora poštovati vremenski redosled!

### Kako Radi?
```
Time Series Split (5 folds):

Fold 1: [TRAIN                    ][TEST]
Fold 2: [TRAIN                         ][TEST]
Fold 3: [TRAIN                              ][TEST]
Fold 4: [TRAIN                                   ][TEST]
Fold 5: [TRAIN                                        ][TEST]

Svaki fold: Train na prošlosti, Test na budućnosti
```

### Python Implementacija:
```python
from sklearn.model_selection import TimeSeriesSplit

# Time series data (sorted by date)
df_timeseries = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100, freq='D'),
    'sales': np.random.randn(100).cumsum(),
    'promotion': np.random.choice([0, 1], 100)
})
df_timeseries = df_timeseries.sort_values('date').reset_index(drop=True)

X_ts = df_timeseries[['promotion']]
y_ts = df_timeseries['sales']

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts), 1):
    print(f"\nFold {fold}:")
    print(f"Train: indices {train_idx[0]} to {train_idx[-1]} (size={len(train_idx)})")
    print(f"Test:  indices {test_idx[0]} to {test_idx[-1]} (size={len(test_idx)})")
    print(f"Train dates: {df_timeseries.iloc[train_idx]['date'].min()} to {df_timeseries.iloc[train_idx]['date'].max()}")
    print(f"Test dates:  {df_timeseries.iloc[test_idx]['date'].min()} to {df_timeseries.iloc[test_idx]['date'].max()}")

# Fold 1:
# Train: indices 0 to 14 (size=15)
# Test:  indices 15 to 29 (size=15)
# ...

# Cross-validation na time series
from sklearn.linear_model import LinearRegression

model = LinearRegression()
cv_scores = cross_val_score(
    model, X_ts, y_ts, 
    cv=tscv,
    scoring='neg_mean_squared_error'
)

print(f"\nTime Series CV MSE: {-cv_scores.mean():.3f}")
```

### ⚠️ KRITIČNO za Time Series:
```python
# ❌ NIKAD za time series - Shuffle=True
X_train, X_test, y_train, y_test = train_test_split(
    X_ts, y_ts, 
    test_size=0.2, 
    shuffle=True  # ❌ LOŠE - Koristi budućnost za trening!
)

# ✅ DOBRO - Shuffle=False ili TimeSeriesSplit
X_train, X_test, y_train, y_test = train_test_split(
    X_ts, y_ts, 
    test_size=0.2, 
    shuffle=False  # ✅ Poslednji 20% = test
)

# ILI još bolje - TimeSeriesSplit za CV
tscv = TimeSeriesSplit(n_splits=5)
```

---

## 7. Leave-One-Out Cross-Validation (LOOCV)

**Ekstremni K-Fold** - K = N (svaki sample jednom test).

### Kako Radi?
```
Dataset sa N=5 samples:

Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]

N folds = N evaluacija
```

### Kada Koristiti?

✅ **VEOMA mali dataset** (< 100 samples)  
✅ **Maksimalan training data** - Koristi N-1 samples za trening  
✅ **Deterministic** - Nema randomness u split-u

❌ **NE koristiti za:**
- Velike dataset-e (SPORO - N iteracija!)
- Computationally expensive modeli

### Python Implementacija:
```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Mali dataset
X_small = np.random.randn(30, 5)
y_small = np.random.choice([0, 1], 30)

# Leave-One-Out
loo = LeaveOneOut()

print(f"Number of folds: {loo.get_n_splits(X_small)}")  # 30 folds!

# Cross-validation
model = RandomForestClassifier(n_estimators=50, random_state=42)
cv_scores = cross_val_score(
    model, X_small, y_small, 
    cv=loo,
    scoring='accuracy'
)

print(f"LOOCV Accuracy: {cv_scores.mean():.3f}")
print(f"LOOCV Std: {cv_scores.std():.3f}")  # Vrlo mala variance

# Comparison: LOOCV vs 5-Fold
cv5_scores = cross_val_score(model, X_small, y_small, cv=5, scoring='accuracy')

print(f"\n5-Fold Accuracy: {cv5_scores.mean():.3f}")
print(f"5-Fold Std: {cv5_scores.std():.3f}")  # Veća variance
```

---

## 8. Group-Based Splitting

**Za podatke sa grupama** - ceo grupa mora biti u istom skupu (train ili test).

### Primeri Grupa:
- **Medicinski podaci** - Pacijenti (više merenja po pacijentu)
- **Time series po entitetima** - Korisnici, prodavnice (višestruki zapisi)
- **Genomic data** - Familije (članovi ne smeju biti split-ovani)

### Zašto je Bitan?
```python
# Problem: Patient A ima 5 merenja
# Ako su 3 u train i 2 u test → DATA LEAKAGE!
# Model uči karakteristike Patient A iz train, testira na Patient A u test!

# Rešenje: SVI merenja Patient A u JEDNOM skupu (train ILI test)
```

### Python Implementacija:
```python
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

# Dataset sa grupama
df_groups = pd.DataFrame({
    'patient_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'feature1': np.random.randn(10),
    'feature2': np.random.randn(10),
    'target': np.random.choice([0, 1], 10)
})

X_groups = df_groups[['feature1', 'feature2']]
y_groups = df_groups['target']
groups = df_groups['patient_id']

# GroupKFold - Grupe ne preklapaju se između fold-ova
gkf = GroupKFold(n_splits=3)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_groups, y_groups, groups), 1):
    train_groups = groups.iloc[train_idx].unique()
    test_groups = groups.iloc[test_idx].unique()
    
    print(f"\nFold {fold}:")
    print(f"Train groups: {sorted(train_groups)}")
    print(f"Test groups: {sorted(test_groups)}")
    
    # Check: Nema overlap!
    assert len(set(train_groups) & set(test_groups)) == 0, "Groups overlap!"

# Fold 1:
# Train groups: [2, 3, 4]
# Test groups: [1]
# ...

# Cross-validation sa groups
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(
    model, X_groups, y_groups, 
    cv=gkf.split(X_groups, y_groups, groups),
    scoring='accuracy'
)

print(f"\nGroupKFold CV: {cv_scores.mean():.3f}")
```

### LeaveOneGroupOut:
```python
# Leave-One-Group-Out - Svaka grupa jednom test
logo = LeaveOneGroupOut()

print(f"Number of folds: {logo.get_n_splits(groups=groups)}")  # 4 (broj unique grupa)

for fold, (train_idx, test_idx) in enumerate(logo.split(X_groups, y_groups, groups), 1):
    test_group = groups.iloc[test_idx].unique()[0]
    train_groups = groups.iloc[train_idx].unique()
    
    print(f"Fold {fold}: Train={sorted(train_groups)}, Test=[{test_group}]")

# Fold 1: Train=[2, 3, 4], Test=[1]
# Fold 2: Train=[1, 3, 4], Test=[2]
# Fold 3: Train=[1, 2, 4], Test=[3]
# Fold 4: Train=[1, 2, 3], Test=[4]
```

---

## Izbor Strategije - Decision Tree
```
Koji je tip problema?
│
├─→ TIME SERIES?
│   └─→ TimeSeriesSplit (shuffle=False!)
│
├─→ Ima GROUPS (patients, users)?
│   └─→ GroupKFold ili LeaveOneGroupOut
│
└─→ Standardan ML problem
    │
    ├─→ Dataset veličina?
    │   │
    │   ├─→ VEOMA mali (< 100)?
    │   │   └─→ LOOCV ili K-Fold (K=10)
    │   │
    │   ├─→ Mali (100-1000)?
    │   │   └─→ K-Fold (K=10) ili Stratified K-Fold
    │   │
    │   ├─→ Srednji (1000-10,000)?
    │   │   └─→ K-Fold (K=5) ili Train-Val-Test split
    │   │
    │   └─→ Veliki (>10,000)?
    │       └─→ Train-Test split (80/20) ili K-Fold (K=3-5)
    │
    └─→ Imbalanced classes?
        ├─→ DA → Stratified K-Fold ili Stratified Train-Test
        └─→ NE → Regular K-Fold ili Train-Test
```

---

## Best Practices - Splitting Checklist

### ✅ DO:

**1. Random State za Reproducibility**
```python
# UVEK postavi random_state!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  # ✅ Isti rezultati svaki put
)
```

**2. Stratify za Classification**
```python
# Za imbalanced ili multi-class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y  # ✅ Održava proporcije klasa
)
```

**3. Check Distribucije Nakon Split-a**
```python
print("Train distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest distribution:")
print(y_test.value_counts(normalize=True))

# Trebaju biti slične!
```

**4. Test Skup je "Locked"**
```python
# Test skup se koristi SAMO JEDNOM - za finalnu evaluaciju!

# ✅ DOBRO
# 1. Train-val split za tuning
# 2. Finalni model sa najboljim params
# 3. Jedna evaluacija na test

# ❌ LOŠE
# Tuning na test skupu → Overfit na test!
```

**5. Cross-Validation za Robusnost**
```python
# Ne oslanjaj se na jedan train-test split
# Koristi CV za bolju procenu

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### ❌ DON'T:

**1. Ne Shuffle Time Series**
```python
# ❌ LOŠE - Koristi budućnost za trening!
X_train, X_test, y_train, y_test = train_test_split(
    X_ts, y_ts, 
    shuffle=True  # NIKAD za time series!
)

# ✅ DOBRO
X_train, X_test, y_train, y_test = train_test_split(
    X_ts, y_ts, 
    shuffle=False
)
```

**2. Ne Zaboravi Stratify za Imbalanced**
```python
# ❌ LOŠE - Može biti bias
X_train, X_test, y_train, y_test = train_test_split(
    X, y_imbalanced, 
    test_size=0.2
)

# ✅ DOBRO
X_train, X_test, y_train, y_test = train_test_split(
    X, y_imbalanced, 
    test_size=0.2, 
    stratify=y_imbalanced
)
```

**3. Ne Split Groups**
```python
# ❌ LOŠE - Patient A u train I test!
X_train, X_test, y_train, y_test = train_test_split(
    X_patients, y_patients, 
    test_size=0.2
)

# ✅ DOBRO - Koristi GroupKFold
gkf = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    model, X_patients, y_patients, 
    cv=gkf.split(X_patients, y_patients, groups=patient_ids)
)
```

**4. Ne Tune na Test Skupu**
```python
# ❌ LOŠE
for param in param_grid:
    model.set_params(**param)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # NIKAD tune na test!
    if score > best_score:
        best_params = param

# ✅ DOBRO - Tune na validation ili CV
for param in param_grid:
    model.set_params(**param)
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    if cv_score > best_score:
        best_params = param
```

**5. Ne Zaboravi Size Balance**
```python
# ❌ LOŠE - Test previše mali
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.05  # Samo 5% - nestabilan skor!
)

# ✅ DOBRO - 20-30% je optimalno
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2  # 20% - dovoljan za evaluaciju
)
```

---

## Common Pitfalls (Česte Greške)

### Greška 1: Data Leakage kroz Split
```python
# ❌ LOŠE - Preprocessing na SVIM podacima!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # FIT na celom dataset-u!
X_train, X_test = train_test_split(X_scaled, test_size=0.2)

# Problem: Mean i std dolaze iz CELOG dataset-a (uključujući test)!

# ✅ DOBRO - Split FIRST
X_train, X_test = train_test_split(X, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)  # Fit SAMO na train
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Greška 2: Overusing Test Set
```python
# ❌ LOŠE - Multiple evaluations na test
scores = []
for model in [model1, model2, model3]:
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)  # Overfit na test!
    scores.append(test_score)

# Biraš model sa najboljim test score → Indirect overfitting!

# ✅ DOBRO - Koristi validation ili CV
for model in [model1, model2, model3]:
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    scores.append(cv_score)

best_model = models[np.argmax(scores)]
best_model.fit(X_train, y_train)

# Test JEDNOM - finalna evaluacija
final_score = best_model.score(X_test, y_test)
```

### Greška 3: Small Test Set
```python
# ❌ LOŠE - Test set previše mali
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=10  # Samo 10 samples!
)

# Problem: Nestabilan accuracy (1 greška = 10% drop!)

# ✅ DOBRO - Bar 20% ili minimum 50-100 samples
test_size = max(0.2 * len(X), 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=int(test_size)
)
```

---

## Practical Examples

### Primer 1: Complete ML Workflow sa Proper Splitting
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Split FIRST (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 3. Preprocessing (fit na train!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Cross-validation za hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold CV
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# 5. Finalni model
final_model = grid_search.best_estimator_

# 6. Test JEDNOM - finalna evaluacija
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"FINAL TEST ACCURACY: {test_accuracy:.3f}")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Primer 2: Time Series sa Rolling Window
```python
from sklearn.metrics import mean_squared_error

# Time series data (sorted)
df_ts = pd.read_csv('sales_data.csv', parse_dates=['date'])
df_ts = df_ts.sort_values('date').reset_index(drop=True)

# Rolling window CV
window_size = 30  # Train window
horizon = 7       # Forecast horizon

scores = []
for i in range(window_size, len(df_ts) - horizon):
    # Train window: [i-window_size : i]
    # Test window: [i : i+horizon]
    
    train_data = df_ts.iloc[i-window_size:i]
    test_data = df_ts.iloc[i:i+horizon]
    
    X_train = train_data[['feature1', 'feature2']]
    y_train = train_data['sales']
    X_test = test_data[['feature1', 'feature2']]
    y_test = test_data['sales']
    
    # Train i predict
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Score
    mse = mean_squared_error(y_test, y_pred)
    scores.append(mse)

print(f"Average Rolling Window MSE: {np.mean(scores):.3f}")
```

---

## Splitting Summary - Quick Reference

| Metoda | Kada Koristiti | Prednosti | Mane |
|--------|---------------|-----------|------|
| **Train-Test (80/20)** | Default, veliki dataset | Brz, jednostavan | Jedan split - variance |
| **Train-Val-Test (60/20/20)** | Hyperparameter tuning | Validation za tuning | Manje training data |
| **K-Fold (K=5)** | Standardan ML, robustan skor | Niska variance | Sporije (K iteracija) |
| **Stratified K-Fold** | Imbalanced classes | Održava proporcije | - |
| **Time Series Split** | Temporalni podaci | Poštuje vreme | Ne shuffle! |
| **LOOCV** | VEOMA mali dataset | Maksimalan training | Ekstremno spor |
| **GroupKFold** | Data sa grupama | Spreč ava leakage | Zahteva group info |

**Default Strategy:**
1. **Large dataset (>10k):** Train-Test (80/20) sa stratify
2. **Medium dataset (1k-10k):** 5-Fold Cross-Validation
3. **Small dataset (<1k):** 10-Fold Cross-Validation ili LOOCV
4. **Imbalanced:** UVEK stratify!
5. **Time series:** TimeSeriesSplit, shuffle=False
6. **Groups:** GroupKFold

**Golden Rules:**
- ✅ Split BEFORE preprocessing (ili fit preprocessing na train)
- ✅ Test set je locked - koristi JEDNOM
- ✅ Uvek stratify za imbalanced classes
- ✅ Random state za reproducibility
- ❌ NIKAD shuffle time series
- ❌ NIKAD tune na test skupu
- ❌ NIKAD fit preprocessing na test

**Key Takeaway:** Proper splitting je osnova valjanog ML projekta. Bez dobrog split-a, tvoj model skor ne znači ništa! Test skup MORA biti neviđen tokom celog development procesa. 🎯