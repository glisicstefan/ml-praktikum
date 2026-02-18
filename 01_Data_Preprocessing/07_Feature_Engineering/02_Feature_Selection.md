# Feature Selection

Feature Selection je proces **biranja najbitnijih features** iz skupa svih dostupnih features. Cilj je zadržati features koje najviše doprinose predviđanju, a ukloniti one koji dodaju šum ili su redundantni.

**Zašto je feature selection bitan?**
- **Smanjuje overfitting** - Manje features = manja šansa za memorisanje šuma
- **Poboljšava performanse** - Model se fokusira na bitne signale
- **Ubrzava trening** - Manje features = brži trening i inference
- **Smanjuje kompleksnost** - Jednostavniji model je lakši za razumevanje i maintain
- **Curse of dimensionality** - Previše features zahteva eksponencijalno više podataka

**VAŽNO:** Feature selection se radi **POSLE** feature creation, a često **PRE ili TOKOM** model training-a!

---

## Problem sa Previše Features

### Curse of Dimensionality:
```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstracija: Potreban broj samples raste eksponencijalno
dimensions = np.arange(1, 21)
samples_needed = 10 ** dimensions  # Približno

plt.figure(figsize=(10, 6))
plt.semilogy(dimensions, samples_needed)
plt.xlabel('Number of Features')
plt.ylabel('Samples Needed (log scale)')
plt.title('Curse of Dimensionality')
plt.grid(True)
plt.show()

# Sa 10 features → Treba 10^10 samples!
# Sa 20 features → Treba 10^20 samples! (nemoguće)
```

### Overfitting Problem:
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate data sa mnogo features
X, y = make_classification(
    n_samples=100,      # Malo samples
    n_features=200,     # MNOGO features!
    n_informative=10,   # Samo 10 su zaista informativne
    n_redundant=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train)):.3f}")
print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
# Train: 1.000 (perfect!)
# Test: 0.700 (loše - overfitting!)

# Problem: 200 features, ali samo 10 su informativne!
```

---

## Tipovi Feature Selection
```
Feature Selection Methods
│
├─→ 1. Filter Methods (Pre-training)
│   ├─ Variance Threshold
│   ├─ Correlation
│   ├─ Statistical Tests (Chi², ANOVA)
│   └─ Mutual Information
│
├─→ 2. Wrapper Methods (Iterative sa modelom)
│   ├─ Recursive Feature Elimination (RFE)
│   ├─ Forward Selection
│   ├─ Backward Elimination
│   └─ Exhaustive Search
│
└─→ 3. Embedded Methods (Tokom training-a)
    ├─ L1 Regularization (Lasso)
    ├─ Tree-based Importance
    └─ Ridge Regression coefficients
```

---

## 1. Filter Methods

**Statistička evaluacija features** nezavisno od modela - brzo i jednostavno.

### A) Variance Threshold

**Uklanja features sa malom varijansom** - ako se feature ne menja, ne može biti informativan!
```python
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],           # Varijacija
    'feature2': [100, 100, 100, 100, 100], # Konstanta - nema varijansu!
    'feature3': [1, 1, 1, 1, 2],           # Skoro konstanta
    'feature4': [10, 20, 30, 40, 50],      # Dobra varijacija
    'target': [0, 1, 0, 1, 0]
})

X = df.drop('target', axis=1)
y = df['target']

# Variance Threshold (ukloni features sa variance < threshold)
selector = VarianceThreshold(threshold=0.1)  # Threshold = 10% variance
X_selected = selector.fit_transform(X)

# Koje features su zadržane?
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features.tolist()}")
# ['feature1', 'feature4'] - feature2 i feature3 uklonjeni!

# DataFrame sa selected features
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
print(X_selected_df)
```

**Kada koristiti:**
- **Preprocessing step** - Pre detaljnije selekcije
- **Konstantne/quasi-konstantne features** - Brzo uklanjanje
- **Veliki dataset** - Brza eliminacija očiglednih nebitnih features

**Parametri:**
```python
# threshold = 0 → Ukloni SAMO konstantne (variance = 0)
selector_const = VarianceThreshold(threshold=0)

# threshold = 0.01 → Ukloni ako variance < 1%
selector_low = VarianceThreshold(threshold=0.01)

# Za binary features: threshold = p(1-p)
# Ukloni ako frekvencija jedne vrednosti > 80%
# threshold = 0.8 * (1 - 0.8) = 0.16
selector_binary = VarianceThreshold(threshold=0.16)
```

---

### B) Correlation Analysis

**Uklanja visoko korelisane features** - redundantne informacije.

#### Correlation sa Target-om:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation sa target
correlations = df.corr()['target'].drop('target').sort_values(ascending=False)

print("Correlation with target:")
print(correlations)

# Visualize
plt.figure(figsize=(10, 6))
correlations.plot(kind='barh')
plt.xlabel('Correlation with Target')
plt.title('Feature Importance - Correlation')
plt.show()

# Select features sa |correlation| > threshold
threshold = 0.2
important_features = correlations[abs(correlations) > threshold].index.tolist()
print(f"Important features: {important_features}")
```

#### Feature-Feature Correlation (Redundancy):
```python
# Correlation matrix
corr_matrix = df.drop('target', axis=1).corr().abs()

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Identify highly correlated pairs
def get_correlated_features(corr_matrix, threshold=0.9):
    """
    Pronalazi parove features sa visokom korelacijom.
    """
    # Upper triangle (bez duplikata)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features sa correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return to_drop

# Features za uklanjanje
drop_features = get_correlated_features(corr_matrix, threshold=0.9)
print(f"Features to drop (high correlation): {drop_features}")

# Remove correlated features
X_uncorrelated = X.drop(columns=drop_features)
```

**Pravilo:**
- **correlation > 0.9** → Definitivno izbaci jedan
- **correlation > 0.7** → Razmotri brisanje
- **correlation < 0.5** → Zadrži oba

---

### C) Statistical Tests

**Testira statističku značajnost** između feature-a i target-a.

#### Chi-Square Test (Za Kategoričke Features):
```python
from sklearn.feature_selection import SelectKBest, chi2

# Categorical features (one-hot encoded) i categorical target
df_cat = pd.DataFrame({
    'category_A': [1, 0, 1, 0, 1],
    'category_B': [0, 1, 0, 1, 0],
    'category_C': [1, 1, 0, 0, 1],
    'target': [0, 1, 0, 1, 0]
})

X_cat = df_cat.drop('target', axis=1)
y_cat = df_cat['target']

# Chi-square test - meri zavisnost između kategoričkih promenljivih
selector = SelectKBest(chi2, k=2)  # Selektuj top 2 features
X_selected = selector.fit_transform(X_cat, y_cat)

# Scores
scores = pd.DataFrame({
    'feature': X_cat.columns,
    'chi2_score': selector.scores_,
    'p_value': selector.pvalues_
}).sort_values('chi2_score', ascending=False)

print(scores)
#       feature  chi2_score   p_value
# 1  category_B        5.00      0.025  ← Significant!
# 0  category_A        1.67      0.197
# 2  category_C        0.60      0.439

# Selected features
selected_features = X_cat.columns[selector.get_support()]
print(f"Selected: {selected_features.tolist()}")
```

#### ANOVA F-test (Za Numeričke Features):
```python
from sklearn.feature_selection import f_classif, f_regression

# Numerical features i categorical target (classification)
X_num = df[['feature1', 'feature4']]
y_class = df['target']

# F-test za classification
selector_f = SelectKBest(f_classif, k=1)  # Top 1 feature
X_selected_f = selector_f.fit_transform(X_num, y_class)

# Scores
f_scores = pd.DataFrame({
    'feature': X_num.columns,
    'f_score': selector_f.scores_,
    'p_value': selector_f.pvalues_
}).sort_values('f_score', ascending=False)

print(f_scores)

# Za regression target
# selector_reg = SelectKBest(f_regression, k=5)
```

**p-value interpretacija:**
- **p < 0.01** → Veoma značajna (highly significant)
- **p < 0.05** → Značajna (significant)
- **p > 0.05** → Nije značajna (not significant) → Razmotri brisanje

---

### D) Mutual Information

**Meri zavisnost** između feature-a i target-a (linearne I nelinearne).
```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Mutual Information za classification
mi_scores = mutual_info_classif(X, y, random_state=42)

# Results
mi_df = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(mi_df)
#       feature  mi_score
# 3    feature4      0.42  ← Najviše informacija!
# 0    feature1      0.18
# 2    feature3      0.05
# 1    feature2      0.00  ← Nema informacije

# Select top K
selector_mi = SelectKBest(mutual_info_classif, k=2)
X_selected_mi = selector_mi.fit_transform(X, y)

# Vizualizacija
plt.figure(figsize=(10, 6))
plt.barh(mi_df['feature'], mi_df['mi_score'])
plt.xlabel('Mutual Information Score')
plt.title('Feature Importance - Mutual Information')
plt.show()
```

**Mutual Information vs Correlation:**
- **Correlation** - Samo **linearne** odnose
- **Mutual Information** - I **linearne I nelinearne** odnose
```python
# Primer: Non-linear relationship
X_nonlinear = np.random.randn(100, 1)
y_nonlinear = (X_nonlinear ** 2).ravel() + np.random.randn(100) * 0.1

# Correlation - niska (linearni odnos je slab)
correlation = np.corrcoef(X_nonlinear.ravel(), y_nonlinear)[0, 1]
print(f"Correlation: {correlation:.3f}")  # ~0

# Mutual Information - visoka (postoji zavisnost!)
mi = mutual_info_regression(X_nonlinear, y_nonlinear, random_state=42)
print(f"Mutual Information: {mi[0]:.3f}")  # Visoka vrednost!
```

---

## 2. Wrapper Methods

**Iterativno evaluiraju kombinacije features** koristeći ML model - sporo ali precizno.

### A) Recursive Feature Elimination (RFE)

**Najpoznatija wrapper metoda** - iterativno uklanja najgore features.

#### Kako RFE Radi?
```
1. Treniraj model sa svim features
2. Ranguj features po važnosti
3. Ukloni najgori feature
4. Repeat korake 1-3 dok ne ostane k features
```

#### Python Implementacija:
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Sample data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500, 
    n_features=20, 
    n_informative=10, 
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RFE - Select top 10 features
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

# Selected features
selected_features = np.where(rfe.support_)[0]
print(f"Selected feature indices: {selected_features}")

# Feature ranking (1 = najbolji)
ranking = pd.DataFrame({
    'feature': range(X.shape[1]),
    'ranking': rfe.ranking_
}).sort_values('ranking')

print(ranking.head(10))
#    feature  ranking
# 0        0        1  ← Top feature
# 1        1        1
# 9        9        1
# ...

# Transform data
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Evaluate
model.fit(X_train_rfe, y_train)
print(f"Accuracy with RFE: {model.score(X_test_rfe, y_test):.3f}")
```

#### RFECV - RFE sa Cross-Validation:
```python
from sklearn.feature_selection import RFECV

# RFE sa automatskim brojem features (koristi CV za izbor)
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=5,                    # 5-fold CV
    scoring='accuracy',
    min_features_to_select=5 # Minimum features
)

rfecv.fit(X_train, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {np.where(rfecv.support_)[0]}")

# Plot CV scores vs number of features
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
         rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features')
plt.ylabel('CV Score')
plt.title('RFECV - Optimal Number of Features')
plt.axvline(rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_}')
plt.legend()
plt.grid(True)
plt.show()
```

**Kada koristiti:**
- **Medium dataset** (< 10,000 samples, < 100 features)
- **Želiš optimalan broj features** - RFECV automatski
- **Wrapper za finalni model** - Koristi isti model za selekciju i trening

**⚠️ Sporo za velike dataset-e!**

---

### B) Forward Selection

**Počinje sa 0 features** i dodaje najbolje jednu po jednu.
```python
from sklearn.feature_selection import SequentialFeatureSelector

# Forward Selection
sfs_forward = SequentialFeatureSelector(
    estimator=LogisticRegression(max_iter=1000),
    n_features_to_select=10,
    direction='forward',  # Dodaj features
    scoring='accuracy',
    cv=5
)

sfs_forward.fit(X_train, y_train)

# Selected features
selected_forward = np.where(sfs_forward.get_support())[0]
print(f"Forward Selection: {selected_forward}")
```

**Proces:**
```
Start: {} (no features)
Step 1: Test sve features pojedinačno → Dodaj najbolji
Step 2: Test sve preostale sa najboljim → Dodaj najbolji
...
Stop: Kada dostigneš n_features_to_select
```

---

### C) Backward Elimination

**Počinje sa svim features** i uklanja najgore jednu po jednu.
```python
# Backward Elimination
sfs_backward = SequentialFeatureSelector(
    estimator=LogisticRegression(max_iter=1000),
    n_features_to_select=10,
    direction='backward',  # Ukloni features
    scoring='accuracy',
    cv=5
)

sfs_backward.fit(X_train, y_train)

# Selected features
selected_backward = np.where(sfs_backward.get_support())[0]
print(f"Backward Elimination: {selected_backward}")
```

**Proces:**
```
Start: {all features}
Step 1: Test sve features uklanjanjem jedne → Ukloni najgoru
Step 2: Test preostale uklanjanjem jedne → Ukloni najgoru
...
Stop: Kada dostigneš n_features_to_select
```

---

### D) Exhaustive Feature Selection

**Testira SVE moguće kombinacije** - garantovano optimalno, ali VEOMA sporo!
```python
from mlxtend.feature_selection import ExhaustiveFeatureSelector

# ⚠️ SAMO za mali broj features! (< 15)
efs = ExhaustiveFeatureSelector(
    estimator=LogisticRegression(max_iter=1000),
    min_features=5,
    max_features=10,
    scoring='accuracy',
    cv=3
)

# UPOZORENJE: Ovo može trajati VEOMA dugo!
# Za 20 features, testira C(20,10) = 184,756 kombinacija!

# efs.fit(X_train_small, y_train_small)
```

**Broj kombinacija:**
- 10 features, select 5: C(10,5) = 252 kombinacije
- 20 features, select 10: C(20,10) = 184,756 kombinacije
- 30 features, select 15: C(30,15) = 155,117,520 kombinacije! 🤯

**Kada koristiti:**
- **VEOMA mali broj features** (< 15)
- **Kritična aplikacija** - Treba garantovano najbolja kombinacija
- **Imaš vremena** - Exhaustive search traje!

---

## 3. Embedded Methods

**Feature selection se dešava TOKOM treniranja** modela - balans brzine i preciznosti.

### A) L1 Regularization (Lasso)

**L1 penalty** gura koeficijente ka nuli - automatski feature selection!
```python
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

# MORA scaling pre L1!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Regression (L1)
lasso = Lasso(alpha=0.1)  # Veći alpha = više features → 0
lasso.fit(X_train_scaled, y_train)

# Features sa non-zero coefficients
non_zero_coef = np.where(lasso.coef_ != 0)[0]
print(f"Non-zero features: {non_zero_coef}")
print(f"Number of selected features: {len(non_zero_coef)}")

# Visualize coefficients
plt.figure(figsize=(12, 6))
plt.bar(range(len(lasso.coef_)), lasso.coef_)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficients (L1 Regularization)')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# Logistic Regression sa L1 (za classification)
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
logreg_l1.fit(X_train_scaled, y_train)

# Selected features
selected_l1 = np.where(logreg_l1.coef_[0] != 0)[0]
print(f"L1 Logistic selected: {selected_l1}")
```

#### Alpha Tuning - Koliko Features?
```python
# Test različite alpha vrednosti
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
n_features_list = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    n_features = np.sum(lasso.coef_ != 0)
    n_features_list.append(n_features)
    print(f"Alpha={alpha:.3f}: {n_features} features")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alphas, n_features_list, marker='o')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Number of Selected Features')
plt.title('L1 Regularization - Feature Selection')
plt.grid(True)
plt.show()

# Alpha ↑ → Fewer features
```

**Kada koristiti:**
- **Linear models** - Regression, Logistic Regression
- **Automatski selection** - Tokom treniranja
- **Interpretability** - Koeficijenti pokazuju važnost

---

### B) Tree-Based Feature Importance

**Decision Trees** i **Random Forests** računaju importance tokom treniranja.
```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# DataFrame
importance_df = pd.DataFrame({
    'feature': range(X_train.shape[1]),
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Visualize
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest - Feature Importance')
plt.show()

# Select top K features
k = 10
top_features = indices[:k]
print(f"Top {k} features: {top_features}")

X_train_selected = X_train[:, top_features]
X_test_selected = X_test[:, top_features]

# Re-train sa selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

print(f"All features accuracy: {rf.score(X_test, y_test):.3f}")
print(f"Top {k} features accuracy: {rf_selected.score(X_test_selected, y_test):.3f}")
```

#### SelectFromModel - Automatic Threshold:
```python
from sklearn.feature_selection import SelectFromModel

# Automatic feature selection based on importance
selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='mean',  # Features sa importance > mean
    prefit=False
)

selector.fit(X_train, y_train)

# Selected features
selected_features = selector.get_support()
print(f"Number of features selected: {np.sum(selected_features)}")

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

**Threshold opcije:**
```python
# median - Features sa importance > median
selector = SelectFromModel(estimator=rf, threshold='median')

# mean - Features sa importance > mean
selector = SelectFromModel(estimator=rf, threshold='mean')

# Custom value
selector = SelectFromModel(estimator=rf, threshold=0.01)

# Top K features (indirektno)
selector = SelectFromModel(estimator=rf, max_features=10)
```

**Kada koristiti:**
- **Tree-based modeli** - RF, XGBoost, LightGBM
- **Brza selekcija** - Jedan trening daje importance
- **Non-linear relationships** - Trees hvat aju kompleksne obrasce

---

### C) XGBoost/LightGBM Importance

**Gradient Boosting** modeli daju još preciznije importance.
```python
import xgboost as xgb

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Feature importance (više tipova!)
# 1. Weight - Number of times feature appears in trees
importance_weight = xgb_model.get_booster().get_score(importance_type='weight')

# 2. Gain - Average gain of splits using this feature
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')

# 3. Cover - Average coverage of splits
importance_cover = xgb_model.get_booster().get_score(importance_type='cover')

# Visualize
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)
plt.title('XGBoost Feature Importance (Gain)')
plt.show()

# Select top K
from sklearn.feature_selection import SelectFromModel

selector_xgb = SelectFromModel(xgb_model, threshold='median', prefit=True)
X_train_selected = selector_xgb.transform(X_train)
X_test_selected = selector_xgb.transform(X_test)

print(f"Selected features: {np.sum(selector_xgb.get_support())}")
```

**Importance types:**
- **weight** - Koliko puta se feature koristi za split (frequency)
- **gain** - Prosečan gain kada se feature koristi (quality)
- **cover** - Prosečan broj samples koji se splituju (coverage)

**Preporuka:** Koristi **gain** - najbolja mera kvaliteta feature-a!

---

## Kombinovane Strategije

**Najbolji pristup:** Kombinuj više metoda!

### Pipeline: Filter → Wrapper → Embedded
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Step-by-step feature selection pipeline
X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Original features: {X_train.shape[1]}")

# Step 1: Variance Threshold (filter)
var_threshold = VarianceThreshold(threshold=0.01)
X_train_var = var_threshold.fit_transform(X_train)
X_test_var = var_threshold.transform(X_test)
print(f"After Variance Threshold: {X_train_var.shape[1]} features")

# Step 2: Statistical test (filter)
k_best = SelectKBest(f_classif, k=50)
X_train_kbest = k_best.fit_transform(X_train_var, y_train)
X_test_kbest = k_best.transform(X_test_var)
print(f"After SelectKBest: {X_train_kbest.shape[1]} features")

# Step 3: RFE (wrapper)
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=20)
X_train_rfe = rfe.fit_transform(X_train_kbest, y_train)
X_test_rfe = rfe.transform(X_test_kbest)
print(f"After RFE: {X_train_rfe.shape[1]} features")

# Step 4: Train final model (embedded)
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train_rfe, y_train)

print(f"\nFinal Accuracy: {final_model.score(X_test_rfe, y_test):.3f}")
```

### Voting Strategy:
```python
# Sakupljanje glasova od više metoda
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Pretpostavimo binary features za chi2
X_binary = (X > X.mean()).astype(int)

# Method 1: Chi-square
chi2_selector = SelectKBest(chi2, k=30)
chi2_selector.fit(X_binary, y)
chi2_support = chi2_selector.get_support()

# Method 2: F-test
f_selector = SelectKBest(f_classif, k=30)
f_selector.fit(X, y)
f_support = f_selector.get_support()

# Method 3: Mutual Information
mi_selector = SelectKBest(mutual_info_classif, k=30)
mi_selector.fit(X, y)
mi_support = mi_selector.get_support()

# Method 4: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_support = rf.feature_importances_ > np.median(rf.feature_importances_)

# Voting: Feature je selected ako minimum N metoda ga izaberu
vote_threshold = 3  # Minimum 3 od 4 metoda
votes = chi2_support.astype(int) + f_support.astype(int) + mi_support.astype(int) + rf_support.astype(int)
selected_by_voting = votes >= vote_threshold

print(f"Features selected by voting (>= {vote_threshold}/4): {np.sum(selected_by_voting)}")

X_voted = X[:, selected_by_voting]
```

---

## Evaluation - Da li je Selection Pomogao?

### Before vs After Comparison:
```python
from sklearn.model_selection import cross_val_score

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Before selection (all features)
scores_all = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"All features ({X.shape[1]}): {scores_all.mean():.3f} ± {scores_all.std():.3f}")

# After selection
X_selected = X[:, selected_by_voting]
scores_selected = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
print(f"Selected features ({X_selected.shape[1]}): {scores_selected.mean():.3f} ± {scores_selected.std():.3f}")

# Improvement
improvement = scores_selected.mean() - scores_all.mean()
print(f"\nImprovement: {improvement:+.3f}")

# Training time comparison
import time

start = time.time()
model.fit(X, y)
time_all = time.time() - start

start = time.time()
model.fit(X_selected, y)
time_selected = time.time() - start

print(f"\nTraining time - All: {time_all:.3f}s")
print(f"Training time - Selected: {time_selected:.3f}s")
print(f"Speedup: {time_all/time_selected:.2f}x")
```

---

## Decision Framework - Koja Metoda?
```
Koliko features imaš?
│
├─→ < 20 features
│   └─→ Exhaustive ili jednostavno probaj sve
│
├─→ 20-100 features
│   ├─→ Brz test? → Filter methods (Variance, Correlation, Statistical)
│   └─→ Optimalno? → Wrapper (RFE, RFECV)
│
└─→ > 100 features
    ├─→ Filter FIRST (smanjiti na ~50)
    └─→ PA Embedded (L1, Tree importance)

Koji model koristiš?
│
├─→ Linear model → L1 Regularization (Lasso)
├─→ Tree model → Tree importance (RF, XGBoost)
└─→ Bilo koji → RFE sa tim modelom

Koliko vremena imaš?
│
├─→ Malo → Filter methods (brzo)
├─→ Srednje → Embedded (RF importance)
└─→ Mnogo → Wrapper (RFE, Forward/Backward)
```

---

## Best Practices

### ✅ DO:

**1. Split PRE Feature Selection**
```python
# ✅ DOBRO - Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature selection SAMO na train
selector = SelectKBest(k=10)
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)  # ISTI selector!
```

**2. Use Multiple Methods**
```python
# Kombinuj filter + wrapper/embedded
# 1. Filter - brzo eliminišeš očigledne
# 2. Wrapper/Embedded - finalni izbor
```

**3. Cross-Validation**
```python
# Ne oslanjaj se na jedan train-test split
# Koristi CV za robusnu evaluaciju
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_selected, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

**4. Track Feature Names**
```python
# Čuvaj koje features su selected
selected_indices = selector.get_support()
selected_feature_names = X.columns[selected_indices]

print(f"Selected features: {selected_feature_names.tolist()}")

# Save za production
import json
with open('selected_features.json', 'w') as f:
    json.dump(selected_feature_names.tolist(), f)
```

**5. Dokumentuj Razloge**
```python
selection_log = {
    'method': 'Random Forest Importance',
    'threshold': 'median',
    'original_features': 100,
    'selected_features': 25,
    'improvement': 0.03,
    'selected_list': selected_feature_names.tolist()
}
```

### ❌ DON'T:

**1. Ne Selektuj Pre Split-a (Data Leakage!)**
```python
# ❌ LOŠE - Koristi test data za selection!
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # FIT na SVIM podacima!
X_train, X_test = train_test_split(X_selected)

# ✅ DOBRO
X_train, X_test = train_test_split(X, y)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

**2. Ne Brisaj Features Preuranjeno**
```python
# ❌ LOŠE - Možda je feature koristan u interakciji!
# Feature može biti nekoristan sam, ali moćan u kombinaciji

# Primer: pojedinačno loši, ali zajedno dobri
# feature1: noise
# feature2: noise
# feature1 × feature2: informativno!

# ✅ DOBRO - Testiraj I interactions
```

**3. Ne Ignoriši Domain Knowledge**
```python
# ❌ LOŠE - Izbaciti "age" jer ima nisku importance
# ALI domain knowledge kaže da je age kritičan!

# ✅ DOBRO - Force include kritične features
# Pa onda selektuj ostale
```

**4. Ne Koristi Samo Jedan Metric**
```python
# ❌ LOŠE - Samo accuracy
# Feature može biti dobar za precision, loš za accuracy

# ✅ DOBRO - Multiple metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluiraj sa više metrika
```

---

## Common Pitfalls

### Greška 1: Feature Selection na Celom Datasetu
```python
# ❌ LOŠE
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Koristi TEST data!
X_train, X_test = train_test_split(X_selected, y)

# Problem: Selector je "video" test data tokom selection-a!

# ✅ DOBRO
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector = SelectKBest(k=10)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

### Greška 2: Zaboravljanje Scaling Pre L1
```python
# ❌ LOŠE - L1 bez scaling-a
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)  # Features na različitim skalama!

# Problem: Features sa većim range-om dobijaju veće koeficijente

# ✅ DOBRO
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
```

### Greška 3: Premalo Features
```python
# ❌ LOŠE - Aggressive selection
selector = SelectKBest(k=2)  # Samo 2 features od 100!

# Problem: Gubi previše informacija → Underfitting

# ✅ DOBRO - Postepeno smanjuj
# Start sa više, PA test kako accuracy pada
for k in [50, 30, 20, 10, 5]:
    selector = SelectKBest(k=k)
    X_selected = selector.fit_transform(X_train, y_train)
    score = cross_val_score(model, X_selected, y_train, cv=5).mean()
    print(f"k={k}: {score:.3f}")
```

---

## Rezime - Feature Selection Quick Reference

### Metode po Brzini:

| Metoda | Brzina | Preciznost | Best For |
|--------|--------|-----------|----------|
| **Variance Threshold** | ⚡⚡⚡ Najbrža | ⭐ Niska | Quick cleanup |
| **Correlation** | ⚡⚡⚡ Brza | ⭐⭐ Srednja | Redundancy removal |
| **Statistical Tests** | ⚡⚡ Brza | ⭐⭐ Srednja | Filter step |
| **Mutual Information** | ⚡⚡ Brza | ⭐⭐⭐ Dobra | Non-linear data |
| **L1 Regularization** | ⚡⚡ Srednja | ⭐⭐⭐ Dobra | Linear models |
| **Tree Importance** | ⚡ Srednja | ⭐⭐⭐⭐ Vrlo dobra | Tree models |
| **RFE** | 🐌 Spora | ⭐⭐⭐⭐ Vrlo dobra | Small-medium data |
| **RFECV** | 🐌🐌 Vrlo spora | ⭐⭐⭐⭐⭐ Najbolja | Critical applications |
| **Exhaustive** | 🐌🐌🐌 Ekstremno spora | ⭐⭐⭐⭐⭐ Garantovano najbolja | < 15 features |

### Default Strategy:
```
1. Variance Threshold (remove constants)
2. Correlation (remove redundancy > 0.9)
3. SelectKBest ili Tree Importance (reduce to ~30-50)
4. RFE ili L1 (finalni izbor ~10-20)
5. Validate sa CV
```

**Key Takeaway:** Feature Selection je balans između accuracy i complexity. Najbolji model nije uvek onaj sa svim features - često je jednostavniji i brži model sa pažljivo odabranim features bolji! Manje je često više! 🎯