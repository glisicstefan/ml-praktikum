# Feature Scaling

Feature Scaling je proces **transformacije numeričkih features na sličan opseg vrednosti**. Različite features često imaju različite jedinice merenja i opsege (npr. starost 20-80, plata 20,000-100,000), što može uticati na performanse mnogih ML algoritama.

**Zašto je scaling bitan?**
- Većina algoritama **ne radi dobro** kada features imaju drastično različite opsege
- **Distance-based algoritmi** (KNN, SVM, K-Means) su posebno osetljivi
- **Gradient descent** konvergira brže sa skaliranim features
- **Regularizacija** (L1/L2) zahteva skaliranje jer penalizuje koeficijente

**KRITIČNO:** Scaling se radi **POSLE** encoding-a, ali **PRE** treniranja modela!

---

## Kada Raditi Scaling?

### Redosled Preprocessing Koraka:
```
1. Data Cleaning
2. EDA
3. Data Transformation (log, sqrt, Box-Cox)
4. Encoding (Label, One-Hot, Target)
5. Feature Scaling  ← OVDE!
6. Train-Test Split (ILI split pre svega, pa fit na train)
7. Model Training
```

---

## Algoritmi koji Zahtevaju Scaling

### ✅ **MORAJU biti skalirani:**

**Distance-Based Algoritmi:**
- **K-Nearest Neighbors (KNN)** - Koristi Euclidean distance
- **K-Means Clustering** - Koristi distance za klastere
- **Hierarchical Clustering** - Distance-based
- **DBSCAN** - Density + distance

**Gradient Descent Algoritmi:**
- **Linear Regression** - Brža konvergencija
- **Logistic Regression** - Brža konvergencija
- **Neural Networks** - Kritično za brzi trening!
- **Support Vector Machines (SVM)** - Kernel functions osetljive na scale

**Regularization:**
- **Ridge Regression (L2)** - Penalizuje koeficijente
- **Lasso Regression (L1)** - Penalizuje koeficijente
- **Elastic Net** - Kombinacija L1 i L2

**Dimensionality Reduction:**
- **PCA (Principal Component Analysis)** - Variance-based
- **LDA (Linear Discriminant Analysis)** - Distance-based

### ❌ **NE moraju biti skalirani:**

**Tree-Based Algoritmi:**
- **Decision Trees** - Prave splits, scale nije bitna
- **Random Forest** - Ensemble trees
- **Gradient Boosting (XGBoost, LightGBM, CatBoost)** - Tree-based
- **AdaBoost** - Ensemble trees

**Naive-Based:**
- **Naive Bayes** - Probabilistički, ne koristi distance

**Rule-Based:**
- **Association Rules** - Frekvencija, ne distance

---

## Tipovi Feature Scaling

---

## 1. Standardization (Z-score Normalization)

**Najčešće korišćena tehnika** - transformiše podatke da imaju **mean = 0** i **standard deviation = 1**.

### Formula:
```
X_scaled = (X - μ) / σ

gde je:
μ = mean (prosek)
σ = standard deviation (standardna devijacija)
```

### Karakteristike:
- **Centar distribucije**: 0
- **Opseg**: Obično između -3 i +3 (ali može izvan)
- **Outliers**: Nisu bounded, ostaju outliers
- **Distribucija**: Čuva oblik distribucije

### Kada Koristiti?
✅ **Normalna/Gaussian distribucija** - Podaci su približno normalno distribuirani  
✅ **Outliers su validni** - Ne želiš da ih "ublaziš"  
✅ **Default choice** - Kad ne znaš šta da koristiš, počni sa ovim  
✅ **Neural Networks** - Preporučeno  
✅ **PCA** - Preporučeno  
✅ **Linear models sa L1/L2** - Preporučeno

### Python Implementacija:
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [30000, 40000, 50000, 60000, 70000],
    'experience': [2, 5, 8, 12, 15]
})

print("Original data:")
print(df.describe())
#        age    salary  experience
# mean  35.0   50000.0        8.4
# std    7.9   15811.4        5.0

# Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("\nScaled data:")
print(df_scaled.describe())
#        age  salary  experience
# mean   0.0     0.0         0.0
# std    1.0     1.0         1.0

print("\nScaled values:")
print(df_scaled.head())
#        age    salary  experience
# 0 -1.265  -1.265      -1.282
# 1 -0.632  -0.632      -0.680
# 2  0.000   0.000      -0.078
# 3  0.632   0.632       0.723
# 4  1.265   1.265       1.325

# Parametre možeš sačuvati
print(f"\nMean: {scaler.mean_}")
print(f"Std: {scaler.scale_}")  # Ovo je std

# Inverse transformation (vraćanje u original)
df_original = scaler.inverse_transform(df_scaled)
print("\nInverse transformed:")
print(pd.DataFrame(df_original, columns=df.columns))
```

### Proper Train-Test Split:
```python
from sklearn.model_selection import train_test_split

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    df[['age', 'salary', 'experience']], 
    df['target'], 
    test_size=0.2, 
    random_state=42
)

# Fit scaler SAMO na train
scaler = StandardScaler()
scaler.fit(X_train)

# Transform train i test SA ISTIM parametrima
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚠️ NE RADI OVO - Data Leakage!
# scaler.fit(X_test)  # NIKAD!
# scaler.fit_transform(pd.concat([X_train, X_test]))  # NIKAD!
```

---

## 2. Min-Max Normalization (Normalization)

Skalira podatke na **fiksni opseg**, obično **[0, 1]**.

### Formula:
```
X_scaled = (X - X_min) / (X_max - X_min)

Ili za custom opseg [a, b]:
X_scaled = a + (X - X_min) × (b - a) / (X_max - X_min)
```

### Karakteristike:
- **Opseg**: [0, 1] ili [a, b]
- **Bounded**: Sve vrednosti unutar opsega
- **Outliers**: Sabijaju ostale vrednosti (problem!)
- **Distribucija**: Čuva oblik

### Kada Koristiti?
✅ **Bounded opseg je potreban** - Neural Networks (aktivacione funkcije)  
✅ **Image data** - Pixel values (0-255 → 0-1)  
✅ **Nema outliers** - Ili su outliers već tretirani  
✅ **Uniformna distribucija** - Podaci su uniformno distribuirani  
✅ **Algorithms sa bounded inputs** - Neural nets sa sigmoid/tanh

❌ **NE koristiti sa:**
- Outliers (sabiju sve ostale vrednosti)
- Novi test data može biti izvan [min, max] train data

### Python Implementacija:
```python
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [30000, 40000, 50000, 60000, 70000]
})

# Min-Max scaling [0, 1]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

print("Scaled to [0, 1]:")
print(pd.DataFrame(df_scaled, columns=df.columns))
#    age  salary
# 0  0.0     0.0
# 1  0.25    0.25
# 2  0.5     0.5
# 3  0.75    0.75
# 4  1.0     1.0

# Custom opseg [-1, 1]
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
df_scaled_custom = scaler_custom.fit_transform(df)

print("\nScaled to [-1, 1]:")
print(pd.DataFrame(df_scaled_custom, columns=df.columns))
#    age  salary
# 0 -1.0    -1.0
# 1 -0.5    -0.5
# 2  0.0     0.0
# 3  0.5     0.5
# 4  1.0     1.0

# Parametri
print(f"\nMin: {scaler.data_min_}")
print(f"Max: {scaler.data_max_}")
print(f"Scale: {scaler.scale_}")
```

### Problem sa Outliers:
```python
# Data sa outlier
df_outlier = pd.DataFrame({
    'salary': [30000, 40000, 50000, 60000, 1000000]  # Outlier!
})

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_outlier)

print("Scaled with outlier:")
print(df_scaled)
# [[0.000]   ← 30,000
#  [0.010]   ← 40,000  Svi sabijeni blizu 0!
#  [0.021]   ← 50,000
#  [0.031]   ← 60,000
#  [1.000]]  ← 1,000,000

# Rešenje: Tretiraj outliers PRE scaling-a ili koristi Robust Scaler!
```

---

## 3. Robust Scaler

**Scaler otporan na outliers** - koristi **median** i **IQR** umesto mean i std.

### Formula:
```
X_scaled = (X - median) / IQR

gde je:
IQR = Q3 - Q1 (Interquartile Range)
Q1 = 25th percentile
Q3 = 75th percentile
```

### Karakteristike:
- **Centar**: Median (ne mean)
- **Spread**: IQR (ne std)
- **Outliers**: Ne utiču na scaling!
- **Opseg**: Nije bounded

### Kada Koristiti?
✅ **Data ima outliers** - Outliers ne smeju uticati na scaling  
✅ **Skewed distribucije** - Kada mean/std nisu reprezentativni  
✅ **Robustnost je prioritet** - Novi data može imati outliers  
✅ **Nakon outlier detection** - Kada znaš da ima outliers ali ih držiš

### Python Implementacija:
```python
from sklearn.preprocessing import RobustScaler

# Data sa outliers
df = pd.DataFrame({
    'salary': [30000, 35000, 40000, 45000, 50000, 1000000]  # Outlier!
})

# Robust Scaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

print("Robust Scaled:")
print(pd.DataFrame(df_scaled, columns=['salary']))
#     salary
# 0   -1.0     ← 30,000
# 1   -0.5     ← 35,000
# 2    0.0     ← 40,000
# 3    0.5     ← 45,000
# 4    1.0     ← 50,000
# 5   95.0     ← 1,000,000 (outlier, ali ne utiče na ostale!)

# Parametri
print(f"\nMedian (center): {scaler.center_}")
print(f"IQR (scale): {scaler.scale_}")

# Poređenje sa StandardScaler
scaler_std = StandardScaler()
df_std = scaler_std.fit_transform(df)

print("\nStandardScaler (za poređenje):")
print(pd.DataFrame(df_std, columns=['salary']))
# Outlier drastično utiče na mean i std, sabija sve ostale!
```

### Poređenje: RobustScaler vs StandardScaler sa Outliers:
```python
import matplotlib.pyplot as plt

# Kreiranje podataka sa outlierima
np.random.seed(42)
normal_data = np.random.normal(50, 10, 100)
outliers = [200, 250, 300]
data = np.concatenate([normal_data, outliers])

# Scaling
robust = RobustScaler().fit_transform(data.reshape(-1, 1))
standard = StandardScaler().fit_transform(data.reshape(-1, 1))

# Vizualizacija
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(data, bins=30)
axes[0].set_title('Original Data')

axes[1].hist(standard, bins=30, color='orange')
axes[1].set_title('StandardScaler (outliers utiču)')

axes[2].hist(robust, bins=30, color='green')
axes[2].set_title('RobustScaler (outliers ne utiču)')

plt.show()
```

---

## 4. MaxAbsScaler

Skalira prema **maksimalnoj apsolutnoj vrednosti** u svakoj feature.

### Formula:
```
X_scaled = X / max(|X|)
```

### Karakteristike:
- **Opseg**: [-1, 1]
- **Čuva zeros**: 0 ostaje 0
- **Čuva znak**: Pozitivni ostaju pozitivni, negativni ostaju negativni
- **Sparse data friendly**: Ne ruši sparse matrice

### Kada Koristiti?
✅ **Sparse data** - CSR/CSC matrice (ne ruši sparsity)  
✅ **Data već centrirana oko 0** - Ne pomera centar  
✅ **Pozitivni i negativni su važni** - Čuva znak vrednosti  
✅ **Text data (TF-IDF)** - Nakon TF-IDF transformacije

### Python Implementacija:
```python
from sklearn.preprocessing import MaxAbsScaler

# Data sa pozitivnim i negativnim vrednostima
df = pd.DataFrame({
    'temperature_change': [-10, -5, 0, 5, 10, 15],
    'profit': [-2000, -1000, 0, 3000, 5000, 8000]
})

# MaxAbsScaler
scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(df)

print("MaxAbs Scaled:")
print(pd.DataFrame(df_scaled, columns=df.columns))
#    temperature_change  profit
# 0        -0.667       -0.250
# 1        -0.333       -0.125
# 2         0.000        0.000   ← Zero je sacuvan!
# 3         0.333        0.375
# 4         0.667        0.625
# 5         1.000        1.000

# Max absolute vrednosti
print(f"\nMax abs: {scaler.max_abs_}")
# [15. 8000.]

# Sparse matrix primer
from scipy.sparse import csr_matrix

sparse_data = csr_matrix([[0, 1, 0], [0, 0, 3], [5, 0, 0]])
print(f"Sparse before: {sparse_data.nnz} non-zero elements")

scaled_sparse = scaler.fit_transform(sparse_data)
print(f"Sparse after: {scaled_sparse.nnz} non-zero elements")
# Sparsity je sacuvana!
```

---

## 5. Normalizer (L2 Normalization)

**Row-wise normalization** - skalira **svaki red** (sample) da ima **normu = 1**.

### Formula:
```
X_normalized = X / ||X||

gde je ||X|| norma (L1, L2, ili max)
```

### Karakteristike:
- **Skalira po redu**, ne po koloni!
- **Svaki sample** ima normu = 1
- **Direction** je važan, ne magnitude

### Kada Koristiti?
✅ **Text classification** - TF-IDF + Cosine similarity  
✅ **Clustering** - Kada je direction bitnija od magnitude  
✅ **Image features** - Histogram normalization  
✅ **Cosine similarity** - KNN sa cosine metric

❌ **NE koristiti za:**
- Tipične ML probleme (koristi StandardScaler)
- Kada magnitude sadrži informaciju

### Python Implementacija:
```python
from sklearn.preprocessing import Normalizer

# Sample data (2 samples, 3 features)
X = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original:")
print(X)

# L2 normalization (default)
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)

print("\nL2 Normalized:")
print(X_normalized)
# [[0.267 0.535 0.802]  ← sqrt(1² + 2² + 3²) = 3.74, svaki deli sa 3.74
#  [0.456 0.570 0.684]]

# Provera - norma svakog reda je 1
row_norms = np.linalg.norm(X_normalized, axis=1)
print(f"\nRow norms: {row_norms}")  # [1. 1.]

# L1 normalization
normalizer_l1 = Normalizer(norm='l1')
X_l1 = normalizer_l1.fit_transform(X)

print("\nL1 Normalized:")
print(X_l1)
# [[0.167 0.333 0.5]    ← sum = 1 + 2 + 3 = 6, svaki deli sa 6
#  [0.267 0.333 0.4]]
```

---

## Poređenje Scaling Tehnika

### Vizuelno Poređenje:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Generate data
np.random.seed(42)
data = np.random.normal(50, 15, 100)
data = np.append(data, [150, 200])  # Add outliers

# Apply scalers
original = data.reshape(-1, 1)
standard = StandardScaler().fit_transform(original)
minmax = MinMaxScaler().fit_transform(original)
robust = RobustScaler().fit_transform(original)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(original, bins=30, color='blue', alpha=0.7)
axes[0, 0].set_title('Original Data')
axes[0, 0].axvline(original.mean(), color='red', linestyle='--', label='Mean')

axes[0, 1].hist(standard, bins=30, color='orange', alpha=0.7)
axes[0, 1].set_title('StandardScaler (mean=0, std=1)')
axes[0, 1].axvline(0, color='red', linestyle='--')

axes[1, 0].hist(minmax, bins=30, color='green', alpha=0.7)
axes[1, 0].set_title('MinMaxScaler [0,1] - outliers sabijaju!')
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].axvline(1, color='red', linestyle='--')

axes[1, 1].hist(robust, bins=30, color='purple', alpha=0.7)
axes[1, 1].set_title('RobustScaler - otporan na outliers')
axes[1, 1].axvline(0, color='red', linestyle='--', label='Median')

plt.tight_layout()
plt.show()
```

### Tabela Poređenja:

| Scaler | Opseg | Centar | Spread | Outliers | Use Case |
|--------|-------|--------|--------|----------|----------|
| **StandardScaler** | Unbounded | Mean → 0 | Std → 1 | Utiču | Default choice, Normal dist |
| **MinMaxScaler** | [0, 1] | N/A | N/A | Sabijaju! | Bounded needed, No outliers |
| **RobustScaler** | Unbounded | Median → 0 | IQR | Ne utiču! | Data sa outliers |
| **MaxAbsScaler** | [-1, 1] | N/A | Max abs | Utiču | Sparse data, čuva zeros |
| **Normalizer** | [0, 1] per row | N/A | L1/L2 norm | N/A | Text, direction matters |

---

## Kada Koristiti Koji Scaler? - Decision Tree
```
Da li imaš outliers u podacima?
│
├─→ DA, i validni su (ne brišeš ih)
│   └─→ RobustScaler
│
├─→ DA, ali nemaš vremena da ih tretiraš
│   └─→ RobustScaler
│
└─→ NE (ili već tretirani)
    │
    ├─→ Trebaš bounded opseg [0,1]?
    │   ├─→ DA (Neural Networks, Images)
    │   │   └─→ MinMaxScaler
    │   │
    │   └─→ NE
    │       │
    │       ├─→ Sparse data (text, TF-IDF)?
    │       │   └─→ MaxAbsScaler
    │       │
    │       ├─→ Row-wise normalization (text classification)?
    │       │   └─→ Normalizer
    │       │
    │       └─→ Default / Ne znaš
    │           └─→ StandardScaler (safest choice)
```

---

## Best Practices - Scaling Checklist

### ✅ DO:

**1. Uvek Fit na Train, Transform na Test**
```python
# ✅ DOBRO
scaler = StandardScaler()
scaler.fit(X_train)  # Fit SAMO na train

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ LOŠE - Data Leakage!
scaler.fit(X_test)  # NIKAD!
scaler.fit(pd.concat([X_train, X_test]))  # NIKAD!
```

**2. Sacuvaj Scaler za Production**
```python
import joblib

# Save
joblib.dump(scaler, 'scaler.pkl')

# Load u production
scaler = joblib.load('scaler.pkl')
new_data_scaled = scaler.transform(new_data)
```

**3. Scale SAMO Numerical Features**
```python
from sklearn.compose import ColumnTransformer

# Automatic selection
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Scale samo numerical
ct = ColumnTransformer([
    ('scaler', StandardScaler(), numerical_cols),
    ('passthrough', 'passthrough', categorical_cols)
])

X_transformed = ct.fit_transform(X)
```

**4. Check Algorithm Requirements**
```python
# Tree-based? NO scaling needed
if algorithm in ['DecisionTree', 'RandomForest', 'XGBoost']:
    X_final = X  # No scaling
else:
    X_final = scaler.fit_transform(X)
```

**5. Dokumentuj Scaling Strategy**
```python
scaling_config = {
    'method': 'StandardScaler',
    'features': ['age', 'salary', 'experience'],
    'reason': 'Using Logistic Regression - requires scaling',
    'params': {
        'mean': scaler.mean_,
        'std': scaler.scale_
    }
}
```

### ❌ DON'T:

**1. Ne scale Target Variable (regression)**
```python
# ❌ LOŠE (osim u posebnim slučajevima)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

# ✅ DOBRO - ostavi target kao što jeste
# Ili ako skaliraš, inverse_transform predictions!
```

**2. Ne koristi fit_transform na test**
```python
# ❌ LOŠE
X_test_scaled = scaler.fit_transform(X_test)  # NIKAD fit_transform na test!

# ✅ DOBRO
X_test_scaled = scaler.transform(X_test)  # Samo transform
```

**3. Ne scale već skalirane features**
```python
# ❌ LOŠE - Double scaling
X_scaled = StandardScaler().fit_transform(X)
X_double_scaled = MinMaxScaler().fit_transform(X_scaled)  # Nepotrebno!

# ✅ DOBRO - Izaberi jedan scaler
X_scaled = StandardScaler().fit_transform(X)
```

**4. Ne zaboravi scale categorical (nakon encoding)**
```python
# ❌ LOŠE
X = pd.get_dummies(X)  # One-hot
X_scaled = scaler.fit_transform(X)  # Scale I binary kolone!

# ✅ DOBRO - Scale samo numerical, ne one-hot
numerical_features = X.select_dtypes(include=[np.number]).columns
X[numerical_features] = scaler.fit_transform(X[numerical_features])
```

**5. Ne zaboravi inverse_transform za interpretaciju**
```python
# Ako si skalirao features za trening:
model.fit(X_train_scaled, y_train)

# Coefficients su u scaled prostoru!
coefficients_scaled = model.coef_

# Za interpretaciju, vrati u original scale
coefficients_original = coefficients_scaled / scaler.scale_

print(f"Original coefficients: {coefficients_original}")
```

---

## Practical Examples

### Primer 1: Complete Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'salary': [30000, 40000, 50000, 60000, 70000, 80000],
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade', 'Niš', 'Novi Sad'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
    'target': [0, 0, 1, 0, 1, 1]
})

# Define columns
numerical_features = ['age', 'salary']
categorical_features = ['city', 'education']

# Split
X = df[numerical_features + categorical_features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit i predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"Predictions: {predictions}")
print(f"Accuracy: {pipeline.score(X_test, y_test)}")

# Save pipeline
joblib.dump(pipeline, 'full_pipeline.pkl')
```

### Primer 2: Manual Scaling sa Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Dataset
X_train = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [30000, 40000, 50000, 60000, 70000],
    'experience': [2, 5, 8, 12, 15],
    'hours_worked': [40, 45, 50, 55, 60]
})
y_train = pd.Series([0, 0, 1, 1, 1])

# 1. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# 2. Feature selection (na scaled data!)
selector = SelectKBest(f_classif, k=2)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)

# Koje features su izabrane?
selected_features = X_train.columns[selector.get_support()]
print(f"Selected features: {selected_features.tolist()}")

# 3. Train model
model = LogisticRegression()
model.fit(X_train_selected, y_train)

# Za test data:
X_test_scaled = scaler.transform(X_test)  # Koristi isti scaler
X_test_selected = selector.transform(X_test_scaled)  # Koristi isti selector
predictions = model.predict(X_test_selected)
```

### Primer 3: Custom Scaler (Mixed Strategy)
```python
from sklearn.base import BaseEstimator, TransformerMixin

class MixedScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler koji koristi različite strategije za različite features.
    """
    def __init__(self, robust_features=None, standard_features=None):
        self.robust_features = robust_features or []
        self.standard_features = standard_features or []
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        if self.robust_features:
            self.robust_scaler.fit(X_df[self.robust_features])
        if self.standard_features:
            self.standard_scaler.fit(X_df[self.standard_features])
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        if self.robust_features:
            X_df[self.robust_features] = self.robust_scaler.transform(
                X_df[self.robust_features]
            )
        if self.standard_features:
            X_df[self.standard_features] = self.standard_scaler.transform(
                X_df[self.standard_features]
            )
        
        return X_df.values

# Korišćenje
scaler = MixedScaler(
    robust_features=['salary'],  # Ima outliers
    standard_features=['age', 'experience']  # Nema outliers
)

X_scaled = scaler.fit_transform(X_train)
```

---

## Common Pitfalls (Česte Greške)

### Greška 1: Scaling Pre Split
```python
# ❌ LOŠE - Data Leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit na SVIM podacima!
X_train, X_test = train_test_split(X_scaled)

# ✅ DOBRO
X_train, X_test = train_test_split(X)  # Split FIRST
scaler = StandardScaler()
scaler.fit(X_train)  # Fit samo na train
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Greška 2: Različiti Scalers za Train i Test
```python
# ❌ LOŠE
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)  # Različiti parametri!

# ✅ DOBRO
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ISTI scaler!
```

### Greška 3: Scaling Tree-Based Algoritama
```python
# ❌ Nepotrebno (ali ne škodi)
X_scaled = StandardScaler().fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)  # Random Forest ne treba scaling

# ✅ DOBRO - štedi vreme
model = RandomForestClassifier()
model.fit(X, y)  # Direktno bez scaling-a
```

### Greška 4: Zaboravljanje Inverse Transform
```python
# ❌ LOŠE - Coefficients su u scaled prostoru
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_scaled, y_train)

print(f"Coefficients: {model.coef_}")  # Teško interpretirati!

# ✅ DOBRO - Vrati u original scale
coefficients_original = model.coef_ / scaler.scale_
print(f"Original coefficients: {coefficients_original}")
```

---

## Scaling Impact - Visualization
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def compare_scaling_impact(X_train, X_test, y_train, y_test):
    """
    Uporedi impact scaling-a na KNN accuracy.
    """
    # Bez scaling-a
    knn_no_scale = KNeighborsClassifier(n_neighbors=5)
    knn_no_scale.fit(X_train, y_train)
    acc_no_scale = accuracy_score(y_test, knn_no_scale.predict(X_test))
    
    # Sa StandardScaler
    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(X_train)
    X_test_std = scaler_std.transform(X_test)
    knn_std = KNeighborsClassifier(n_neighbors=5)
    knn_std.fit(X_train_std, y_train)
    acc_std = accuracy_score(y_test, knn_std.predict(X_test_std))
    
    # Sa MinMaxScaler
    scaler_mm = MinMaxScaler()
    X_train_mm = scaler_mm.fit_transform(X_train)
    X_test_mm = scaler_mm.transform(X_test)
    knn_mm = KNeighborsClassifier(n_neighbors=5)
    knn_mm.fit(X_train_mm, y_train)
    acc_mm = accuracy_score(y_test, knn_mm.predict(X_test_mm))
    
    # Rezultati
    print(f"No Scaling:        {acc_no_scale:.3f}")
    print(f"StandardScaler:    {acc_std:.3f}")
    print(f"MinMaxScaler:      {acc_mm:.3f}")
    
    # Bar chart
    import matplotlib.pyplot as plt
    
    scalers = ['No Scaling', 'StandardScaler', 'MinMaxScaler']
    accuracies = [acc_no_scale, acc_std, acc_mm]
    
    plt.figure(figsize=(10, 6))
    plt.bar(scalers, accuracies, color=['red', 'green', 'blue'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Impact of Scaling on KNN Accuracy')
    plt.ylim([0, 1])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.show()

# Korišćenje
compare_scaling_impact(X_train, X_test, y_train, y_test)
```

---

## Rezime - Scaling Quick Reference

| Scaler | Formula | Range | Best For | Avoid For |
|--------|---------|-------|----------|-----------|
| **StandardScaler** | (X - μ) / σ | Unbounded | Default, Normal dist, PCA, NN | Data sa outliers |
| **MinMaxScaler** | (X - min) / (max - min) | [0, 1] | Bounded needed, Images, NN | Data sa outliers |
| **RobustScaler** | (X - median) / IQR | Unbounded | Data sa outliers, Skewed | - |
| **MaxAbsScaler** | X / max(\|X\|) | [-1, 1] | Sparse data, Text (TF-IDF) | Dense data |
| **Normalizer** | X / \|\|X\|\| | [0, 1] per row | Text classification, Cosine similarity | Typical ML |

**Default Strategy:**
1. Check for outliers (EDA)
2. Outliers? → RobustScaler
3. No outliers? → StandardScaler (safest)
4. Special case (bounded, sparse, text)? → Choose accordingly

**Critical Rules:**
- ✅ ALWAYS: Fit on train, transform on test
- ✅ ALWAYS: Save scaler for production
- ✅ ALWAYS: Scale ONLY numerical features
- ❌ NEVER: fit_transform on test data
- ❌ NEVER: Scale tree-based algorithms (optional, ne škodi ali nepotrebno)

**Key Takeaway:** Scaling je KRITIČAN za distance-based i gradient-based algoritme, ali NIJE potreban za tree-based. Izbor scaler-a zavisi od outliers i distribucije podataka. StandardScaler je safe default! 🎯