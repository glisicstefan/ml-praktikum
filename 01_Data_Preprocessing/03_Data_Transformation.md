# Data Transformation

Data Transformation je proces **promena oblika ili strukture podataka** da bi se poboljšala njihova pogodnost za mašinsko učenje. Ovo NIJE isto što i Feature Scaling - transformacije menjaju distribuciju i prirodu podataka, ne samo njihov opseg.

**Zašto radimo transformacije?**
- Većina ML algoritama pretpostavlja **normalnu distribuciju** podataka
- **Smanjenje skewness** (asimetrije) poboljšava performanse linearnih modela
- **Stabilizacija variance** kroz dataset
- **Linearizacija odnosa** između features i target-a
- **Smanjenje uticaja outliers** bez njihovog brisanja

**VAŽNO:** Transformacije se rade **POSLE EDA**, jer EDA pokazuje koje transformacije su potrebne!

---

## Kada Raditi Transformacije?

### EDA Insights → Transformation Decision

| Problem u EDA | Rešenje |
|---------------|---------|
| **Skewed distribution** (skewness > 1 ili < -1) | Log, sqrt, Box-Cox transformation |
| **Heavy-tailed distribution** (veliki outliers) | Log ili Yeo-Johnson transformation |
| **Non-linear relationships** | Polynomial features |
| **Different units/ranges** | Feature Scaling (to ide u sledeći fajl!) |
| **Continuous → Categories needed** | Binning/Discretization |
| **Heteroscedasticity** (različita variansa) | Log ili power transformation |

---

## Tipovi Transformacija

---

## 1. Log Transformation (Logaritamska Transformacija)

**Najčešće korišćena transformacija** za right-skewed (pozitivno asimetrične) distribucije.

### Formula:
```
X' = log(X)     ili    X' = log(X + 1)    [ako ima zeros]
```

### Kada Koristiti?
✅ **Right-skewed data** - Repna distribucija desno (income, prices, population)  
✅ **Eksponencijalni rast** - Kada vrednosti rastu eksponencijalno  
✅ **Multiplicative relationships** - Kada odnosi su multiplikativni, ne aditivni  
✅ **Wide range of values** - Npr. 1 do 1,000,000  
✅ **Outliers desno** - Smanjuje uticaj velikih outliers

❌ **NE koristiti** sa:
- Negativnim vrednostima (log je nedefinisan)
- Podacima sa mnogo zeros (osim log1p)
- Left-skewed data

### Python Implementacija:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Original data (right-skewed)
df['income'].hist(bins=50, alpha=0.7, label='Original')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Original Distribution')
plt.show()

print(f"Original Skewness: {df['income'].skew():.3f}")

# Log transformation
df['income_log'] = np.log(df['income'])

# Log1p - bolje kada ima zeros ili male vrednosti
df['income_log1p'] = np.log1p(df['income'])  # log(1 + x)

# Vizualizacija posle
df['income_log1p'].hist(bins=50, alpha=0.7, label='Log Transformed', color='orange')
plt.xlabel('Log(Income)')
plt.ylabel('Frequency')
plt.title('After Log Transformation')
plt.show()

print(f"Transformed Skewness: {df['income_log1p'].skew():.3f}")

# Inverse transformation (vraćanje u original)
df['income_original'] = np.expm1(df['income_log1p'])  # exp(x) - 1
```

### Efekat na Model:

**Pre transformacije:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train[['income']], y_train)
r2_before = r2_score(y_test, model.predict(X_test[['income']]))
print(f"R² before: {r2_before:.3f}")
```

**Posle transformacije:**
```python
model = LinearRegression()
model.fit(X_train[['income_log']], y_train)
r2_after = r2_score(y_test, model.predict(X_test[['income_log']]))
print(f"R² after: {r2_after:.3f}")  # Obično značajno bolje!
```

---

## 2. Square Root Transformation (Kvadratni Koren)

**Blaža transformacija** od log - koristi se za umereno skewed podatke.

### Formula:
```
X' = √X    ili    X' = √(X + constant)
```

### Kada Koristiti?
✅ **Moderate right-skew** - Skewness između 0.5 i 1.5  
✅ **Count data** - Broj događaja, transakcija (Poisson distribucija)  
✅ **Blaža alternativa log-u** - Kada log previše "sabija" podatke  
✅ **Variance stabilization** - Za Poisson distribucije

### Python Implementacija:
```python
# Provera skewness
print(f"Original Skewness: {df['num_purchases'].skew():.3f}")

# Square root transformation
df['num_purchases_sqrt'] = np.sqrt(df['num_purchases'])

# Ako ima negativne vrednosti, dodaj konstantu prvo
min_val = df['num_purchases'].min()
if min_val < 0:
    df['num_purchases_sqrt'] = np.sqrt(df['num_purchases'] - min_val + 1)

# Vizualizacija
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['num_purchases'], bins=30, color='blue', alpha=0.7)
axes[0].set_title('Original')
axes[1].hist(df['num_purchases_sqrt'], bins=30, color='green', alpha=0.7)
axes[1].set_title('Square Root Transformed')
plt.show()

# Inverse transformation
df['num_purchases_original'] = df['num_purchases_sqrt'] ** 2
```

### Poređenje: Log vs Sqrt
```python
# Za podatke sa različitim nivoima skewness
skewness_original = df['feature'].skew()

if skewness_original > 1.5:
    print("Use LOG transformation")
    df['feature_transformed'] = np.log1p(df['feature'])
elif 0.5 < skewness_original <= 1.5:
    print("Use SQRT transformation")
    df['feature_transformed'] = np.sqrt(df['feature'])
else:
    print("No transformation needed")
    df['feature_transformed'] = df['feature']
```

---

## 3. Box-Cox Transformation

**Najmoćnija transformacija** - automatski nalazi najbolji eksponent λ (lambda) za normalizaciju.

### Formula:
```
         ⎧ (X^λ - 1) / λ    if λ ≠ 0
X' =  ⎨
         ⎩ log(X)            if λ = 0
```

### Kada Koristiti?
✅ **Kada ne znaš koja transformacija** - Box-Cox automatski pronalazi najbolju  
✅ **Strictly positive data** - SAMO pozitivne vrednosti (X > 0)  
✅ **Optimize for normality** - Kada je cilj maksimalna normalnost  
✅ **Statistical modeling** - Linear regression, ANOVA

❌ **NE koristiti** sa:
- Negative vrednostima (mora X > 0)
- Zeros (dodaj malu konstantu ili koristi Yeo-Johnson)

### Python Implementacija:
```python
from scipy.stats import boxcox

# Original data (mora biti > 0!)
if (df['price'] <= 0).any():
    df['price'] = df['price'] + 1  # Dodaj konstantu

# Box-Cox transformation
df['price_boxcox'], lambda_value = boxcox(df['price'])

print(f"Optimal Lambda: {lambda_value:.3f}")
print(f"Original Skewness: {df['price'].skew():.3f}")
print(f"Transformed Skewness: {df['price_boxcox'].skew():.3f}")

# Interpretacija lambda:
# λ = 1   → No transformation
# λ = 0.5 → Square root transformation
# λ = 0   → Log transformation
# λ = -1  → Reciprocal transformation

# Vizualizacija
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['price'], bins=50, color='blue', alpha=0.7)
axes[0].set_title(f'Original (skew={df["price"].skew():.2f})')

axes[1].hist(df['price_boxcox'], bins=50, color='green', alpha=0.7)
axes[1].set_title(f'Box-Cox (λ={lambda_value:.2f}, skew={df["price_boxcox"].skew():.2f})')

# Q-Q plot za normalnost
from scipy import stats
stats.probplot(df['price_boxcox'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot (After Box-Cox)')

plt.tight_layout()
plt.show()

# Inverse transformation
from scipy.special import inv_boxcox
df['price_original'] = inv_boxcox(df['price_boxcox'], lambda_value)
```

### Automatska Primena na Sve Features:
```python
from scipy.stats import boxcox

def apply_boxcox_if_needed(df, columns, skew_threshold=0.75):
    """
    Primenjuje Box-Cox transformaciju na kolone sa visokim skewness.
    """
    transformations = {}
    
    for col in columns:
        if df[col].skew() > skew_threshold:
            # Osiguraj da su sve vrednosti pozitivne
            if (df[col] <= 0).any():
                df[col] = df[col] - df[col].min() + 1
            
            # Box-Cox transformation
            df[f'{col}_boxcox'], lambda_val = boxcox(df[col])
            transformations[col] = lambda_val
            
            print(f"{col}: skew {df[col].skew():.2f} → {df[f'{col}_boxcox'].skew():.2f} (λ={lambda_val:.2f})")
    
    return df, transformations

# Primena
numerical_cols = df.select_dtypes(include=[np.number]).columns
df, lambda_dict = apply_boxcox_if_needed(df, numerical_cols)
```

---

## 4. Yeo-Johnson Transformation

**Box-Cox za sve tipove podataka** - radi i sa negativnim vrednostima!

### Formula:
Kompleksnija formula koja se primenjuje različito za pozitivne i negativne vrednosti.

### Kada Koristiti?
✅ **Sve gde Box-Cox** + dodatno:  
✅ **Negative values** - Ne mora X > 0  
✅ **Mixed data** - Kombinacija pozitivnih i negativnih  
✅ **Zeros** - Može da radi sa zeros

### Python Implementacija:
```python
from sklearn.preprocessing import PowerTransformer

# Yeo-Johnson (radi sa svim vrednostima)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
df['feature_yj'] = pt.fit_transform(df[['feature']])

print(f"Original Skewness: {df['feature'].skew():.3f}")
print(f"Transformed Skewness: {df['feature_yj'].skew():.3f}")

# Lambda vrednost
print(f"Optimal Lambda: {pt.lambdas_[0]:.3f}")

# Vizualizacija
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['feature'], bins=50, alpha=0.7)
axes[0].set_title('Original')
axes[1].hist(df['feature_yj'], bins=50, color='orange', alpha=0.7)
axes[1].set_title('Yeo-Johnson Transformed')
plt.show()

# Inverse transformation
df['feature_original'] = pt.inverse_transform(df[['feature_yj']])
```

### Box-Cox vs Yeo-Johnson:
```python
def choose_power_transform(df, column):
    """
    Automatski bira Box-Cox ili Yeo-Johnson na osnovu podataka.
    """
    if (df[column] <= 0).any():
        print(f"{column}: Has negatives/zeros → Using Yeo-Johnson")
        pt = PowerTransformer(method='yeo-johnson')
    else:
        print(f"{column}: All positive → Using Box-Cox")
        pt = PowerTransformer(method='box-cox')
    
    df[f'{column}_transformed'] = pt.fit_transform(df[[column]])
    return df, pt

# Primena
df, transformer = choose_power_transform(df, 'revenue')
```

---

## 5. Reciprocal Transformation (Recipročna Transformacija)

**Za left-skewed (negativno asimetrične) distribucije.**

### Formula:
```
X' = 1 / X    ili    X' = 1 / (X + constant)
```

### Kada Koristiti?
✅ **Left-skewed data** - Rep distribucije levo  
✅ **Inverse relationships** - Kada je odnos 1/X (npr. brzina vs vreme)  
✅ **Time-to-event** - Konverzija u rate

❌ **NE koristiti** sa zeros ili values blizu nule

### Python Implementacija:
```python
# Left-skewed data
print(f"Original Skewness: {df['time_to_complete'].skew():.3f}")  # Negative skew

# Reciprocal transformation
df['rate'] = 1 / df['time_to_complete']

# Sa safety check za zeros
df['rate_safe'] = 1 / (df['time_to_complete'] + 1e-6)  # Mala konstanta

# Vizualizacija
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['time_to_complete'], bins=50, color='blue', alpha=0.7)
axes[0].set_title(f'Original (left-skew={df["time_to_complete"].skew():.2f})')
axes[1].hist(df['rate'], bins=50, color='purple', alpha=0.7)
axes[1].set_title(f'Reciprocal (skew={df["rate"].skew():.2f})')
plt.show()

# Inverse
df['time_original'] = 1 / df['rate']
```

---

## 6. Binning / Discretization (Diskretizacija)

**Pretvaranje continuous features u categorical/ordinal** features.

### Kada Koristiti?
✅ **Non-linear relationships** - Kada je odnos feature→target kompleksan  
✅ **Interpretability** - Lakše objašnjavanje ("low, medium, high income")  
✅ **Handling outliers** - Ekstremne vrednosti idu u isti bin  
✅ **Categorical algorithms** - Decision Trees, Naive Bayes  
✅ **Reducing noise** - Smoothing kontinualnih podataka

### Tipovi Binning-a:

#### A) **Equal-Width Binning** (Jednaka širina)
```python
# Jednaka širina intervala
df['age_binned'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])

print(df['age_binned'].value_counts().sort_index())

# Custom bin edges
bins = [0, 18, 35, 50, 65, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior']
df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)

# Vizualizacija
df['age_category'].value_counts().plot(kind='bar')
plt.title('Age Distribution After Binning')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.show()
```

#### B) **Equal-Frequency Binning** (Quantile-based)
```python
# Jednaka količina podataka u svakom bin-u
df['income_quartiles'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(df['income_quartiles'].value_counts())  # Približno jednak broj u svakom

# Deciles (10 bins)
df['income_deciles'] = pd.qcut(df['income'], q=10, labels=False)  # 0-9

# Custom quantiles
df['income_custom'] = pd.qcut(df['income'], q=[0, 0.25, 0.5, 0.75, 0.9, 1.0], 
                               labels=['Low', 'Medium', 'High', 'Very High', 'Top 10%'])
```

#### C) **Domain-Based Binning** (Ekspertsko znanje)
```python
# Na osnovu domain knowledge
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['bmi_category'] = df['bmi'].apply(categorize_bmi)

# Ili sa pd.cut
bins = [0, 18.5, 25, 30, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bins, labels=labels)
```

#### D) **KBinsDiscretizer** (Sklearn)
```python
from sklearn.preprocessing import KBinsDiscretizer

# Uniform (equal-width)
kbd_uniform = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df['price_binned_uniform'] = kbd_uniform.fit_transform(df[['price']])

# Quantile (equal-frequency)
kbd_quantile = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['price_binned_quantile'] = kbd_quantile.fit_transform(df[['price']])

# KMeans-based (data-driven clusters)
kbd_kmeans = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
df['price_binned_kmeans'] = kbd_kmeans.fit_transform(df[['price']])

# One-hot encoding output
kbd_onehot = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
price_binned_onehot = kbd_onehot.fit_transform(df[['price']])
```

### Kada Koristiti Koji Tip?

| Tip Binning | Kada Koristiti |
|-------------|----------------|
| **Equal-Width** | Uniformna distribucija, vizualizacija |
| **Equal-Frequency** | Skewed distribucije, osigurava balans |
| **Domain-Based** | Postoje prirodne kategorije (BMI, temperature) |
| **KMeans-Based** | Data-driven, kada nemaš prior knowledge |

---

## 7. Polynomial Features (Polinomijalne Feature-e)

**Kreiranje interakcija i polinoma** za modelovanje non-linear odnosa.

### Formula:
```
Za features [a, b]:
Degree 2: [1, a, b, a², ab, b²]
Degree 3: [1, a, b, a², ab, b², a³, a²b, ab², b³]
```

### Kada Koristiti?
✅ **Non-linear relationships** - EDA pokazuje curve odnos  
✅ **Linear models** - Dodaje non-linearity Linear Regression-u  
✅ **Interaction effects** - Kada features zajedno utiču (price × quantity)  
⚠️ **OPREZ:** Brzo raste broj features → curse of dimensionality

### Python Implementacija:
```python
from sklearn.preprocessing import PolynomialFeatures

# Original features
X = df[['height', 'weight']]
print(f"Original shape: {X.shape}")  # (n, 2)

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Polynomial shape: {X_poly.shape}")  # (n, 5)

# Feature names
print("Feature names:", poly.get_feature_names_out())
# ['height', 'weight', 'height^2', 'height*weight', 'weight^2']

# DataFrame sa novim features
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
print(X_poly_df.head())

# Samo interaction terms (bez kvadrata)
poly_interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interaction = poly_interaction.fit_transform(X)
print("Interaction features:", poly_interaction.get_feature_names_out())
# ['height', 'weight', 'height*weight']  # Nema height^2, weight^2
```

### Efekat na Linear Model:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Bez polynomial features
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
r2_simple = r2_score(y_test, model_simple.predict(X_test))
print(f"R² (Linear): {r2_simple:.3f}")

# SA polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
r2_poly = r2_score(y_test, model_poly.predict(X_test_poly))
print(f"R² (Polynomial): {r2_poly:.3f}")  # Obično značajno bolje!
```

### Oprez - Overfitting!
```python
# Degree 3, 4, 5... → eksplozija features
for degree in [1, 2, 3, 4, 5]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(df[['x1', 'x2', 'x3']])
    print(f"Degree {degree}: {X_poly.shape[1]} features")

# Output:
# Degree 1: 3 features
# Degree 2: 9 features
# Degree 3: 19 features
# Degree 4: 34 features
# Degree 5: 55 features → PREVIŠE!

# Rešenje: Regularization (L1, L2) + Feature Selection
```

---

## Strategija Transformacije - Decision Tree
```
1. Uradi EDA → Identifikuj problem
   │
   ├─→ Right-Skewed (skew > 1.5)?
   │   ├─ DA → Log Transformation
   │   └─ NE → Nastavi
   │
   ├─→ Moderate Skew (0.5 < skew < 1.5)?
   │   ├─ DA → Square Root Transformation
   │   └─ NE → Nastavi
   │
   ├─→ Ne znaš koju transformaciju?
   │   ├─ DA, ima negative → Yeo-Johnson
   │   ├─ DA, sve positive → Box-Cox
   │   └─ NE → Nastavi
   │
   ├─→ Left-Skewed?
   │   ├─ DA → Reciprocal Transformation
   │   └─ NE → Nastavi
   │
   ├─→ Non-linear relationship sa target?
   │   ├─ DA, koristiš Linear model → Polynomial Features
   │   ├─ DA, koristiš Tree model → Binning (opciono)
   │   └─ NE → Nastavi
   │
   └─→ Treba interpretability ili handle outliers?
       ├─ DA → Binning/Discretization
       └─ NE → No transformation needed
```

---

## Automatska Transformacija Pipeline
```python
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox, skew

class AutoTransformer(BaseEstimator, TransformerMixin):
    """
    Automatski primenjuje najbolju transformaciju na svaku feature.
    """
    def __init__(self, skew_threshold=0.75, method='auto'):
        self.skew_threshold = skew_threshold
        self.method = method  # 'auto', 'log', 'sqrt', 'boxcox', 'yeo-johnson'
        self.transformations_ = {}
        self.lambdas_ = {}
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        
        for col in X.columns:
            col_skew = skew(X[col].dropna())
            
            if abs(col_skew) < self.skew_threshold:
                self.transformations_[col] = 'none'
                continue
            
            if self.method == 'auto':
                # Odluči na osnovu skewness i data
                if col_skew > 1.5:
                    if (X[col] <= 0).any():
                        self.transformations_[col] = 'yeo-johnson'
                    else:
                        self.transformations_[col] = 'log'
                elif 0.5 < col_skew <= 1.5:
                    self.transformations_[col] = 'sqrt'
                elif col_skew < -0.5:
                    self.transformations_[col] = 'reciprocal'
                else:
                    self.transformations_[col] = 'none'
            else:
                self.transformations_[col] = self.method
            
            # Sačuvaj lambda za Box-Cox ako potrebno
            if self.transformations_[col] == 'boxcox':
                _, lambda_val = boxcox(X[col] - X[col].min() + 1)
                self.lambdas_[col] = lambda_val
        
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        for col in X.columns:
            transform_type = self.transformations_.get(col, 'none')
            
            if transform_type == 'log':
                X[col] = np.log1p(X[col])
            elif transform_type == 'sqrt':
                X[col] = np.sqrt(X[col])
            elif transform_type == 'reciprocal':
                X[col] = 1 / (X[col] + 1e-6)
            elif transform_type == 'boxcox':
                lambda_val = self.lambdas_[col]
                X[col] = boxcox(X[col], lmbda=lambda_val)
            # yeo-johnson bi zahtevao PowerTransformer
        
        return X

# Korišćenje
transformer = AutoTransformer(skew_threshold=0.75, method='auto')
X_transformed = transformer.fit_transform(df[numerical_cols])

print("Applied transformations:")
for col, trans in transformer.transformations_.items():
    print(f"{col}: {trans}")
```

---

## Best Practices

### ✅ DO:

1. **Uradi EDA prvo** - Ne transformiši naslepo!
```python
# Prvo pogleda distribuciju
df['feature'].hist(bins=50)
print(f"Skewness: {df['feature'].skew():.3f}")

# PA ONDA transformiši
```

2. **Fit na TRAIN, transform na TEST**
```python
# LOŠE - Data leakage!
df_all['price_log'] = np.log1p(df_all['price'])
train, test = train_test_split(df_all)

# DOBRO
train['price_log'] = np.log1p(train['price'])
test['price_log'] = np.log1p(test['price'])  # Ista transformacija, ali odvojeno
```

3. **Sačuvaj parametre za production**
```python
# Sačuvaj lambda iz Box-Cox
_, lambda_val = boxcox(train['price'])
joblib.dump(lambda_val, 'price_lambda.pkl')

# U production
lambda_val = joblib.load('price_lambda.pkl')
test_transformed = boxcox(test['price'], lmbda=lambda_val)
```

4. **Proveri rezultat posle transformacije**
```python
print(f"Before: skew={df['price'].skew():.2f}, kurtosis={df['price'].kurtosis():.2f}")
print(f"After:  skew={df['price_log'].skew():.2f}, kurtosis={df['price_log'].kurtosis():.2f}")

# Q-Q plot za normalnost
from scipy import stats
stats.probplot(df['price_log'], dist="norm", plot=plt)
plt.show()
```

5. **Dokumentuj šta si radio**
```python
transformation_log = {
    'features_transformed': ['price', 'income', 'sales'],
    'method': 'log1p',
    'reason': 'Right-skewed distributions (skew > 1.5)',
    'improvement': 'Skewness reduced from 2.3 to 0.1'
}
```

### ❌ DON'T:

1. **Ne transformiši sve feature-e automatski**
```python
# LOŠE
for col in df.columns:
    df[col] = np.log1p(df[col])  # Šta ako je već normalan?

# DOBRO - samo ako treba
for col in df.columns:
    if df[col].skew() > 1.5:
        df[col] = np.log1p(df[col])
```

2. **Ne koristi Box-Cox na negativnim**
```python
# LOŠE - Error!
boxcox(df['profit'])  # profit može biti negativan

# DOBRO
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['profit_transformed'] = pt.fit_transform(df[['profit']])
```

3. **Ne zaboravi inverse transformation**
```python
# Ako predviđaš na log scale
y_pred_log = model.predict(X_test)

# Moraš vratiti u original scale!
y_pred = np.expm1(y_pred_log)  # Inverse of log1p
```

4. **Ne pretvori target variable bez razloga**
```python
# LOŠE - Komplikuje evaluaciju
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)  # Sad je sve u log space

# DOBRO - Ostavi target ako nije neophodno
# Ili ako transformišeš, budi konzistentan sa evaluacijom
```

---

## Provera - Da li je Transformacija Pomogla?
```python
def evaluate_transformation(original, transformed, feature_name):
    """
    Uporedi original i transformisanu feature.
    """
    from scipy import stats
    
    print(f"\n{'='*50}")
    print(f"Evaluation: {feature_name}")
    print(f"{'='*50}")
    
    # Skewness
    print(f"Skewness: {original.skew():.3f} → {transformed.skew():.3f}")
    
    # Kurtosis
    print(f"Kurtosis: {original.kurtosis():.3f} → {transformed.kurtosis():.3f}")
    
    # Normality test (Shapiro-Wilk)
    _, p_before = stats.shapiro(original.sample(min(5000, len(original))))
    _, p_after = stats.shapiro(transformed.sample(min(5000, len(transformed))))
    print(f"Normality p-value: {p_before:.4f} → {p_after:.4f}")
    print(f"Is normal (p>0.05): {p_before > 0.05} → {p_after > 0.05}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histograms
    axes[0, 0].hist(original, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title(f'Original (skew={original.skew():.2f})')
    axes[0, 1].hist(transformed, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title(f'Transformed (skew={transformed.skew():.2f})')
    
    # Q-Q plots
    stats.probplot(original, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot - Original')
    stats.probplot(transformed, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot - Transformed')
    
    plt.tight_layout()
    plt.show()

# Korišćenje
evaluate_transformation(df['price'], df['price_log'], 'Price')
```

---

## Rezime - Transformation Cheat Sheet

| Scenario | Transformacija | Python |
|----------|----------------|--------|
| Right-skewed (skew > 1.5) | Log | `np.log1p(X)` |
| Moderate skew (0.5-1.5) | Square Root | `np.sqrt(X)` |
| Left-skewed | Reciprocal | `1 / (X + 1e-6)` |
| Automatic (all positive) | Box-Cox | `scipy.stats.boxcox(X)` |
| Automatic (any values) | Yeo-Johnson | `PowerTransformer(method='yeo-johnson')` |
| Non-linear relationship | Polynomial | `PolynomialFeatures(degree=2)` |
| Need categories | Binning | `pd.cut()` ili `pd.qcut()` |
| Count data (Poisson) | Square Root | `np.sqrt(X)` |
| Percentage data | Logit | `np.log(X / (1 - X))` |

---

## Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.linear_model import Ridge

# Full preprocessing pipeline
pipeline = Pipeline([
    ('power_transform', PowerTransformer(method='yeo-johnson')),  # Normalizacija
    ('poly_features', PolynomialFeatures(degree=2)),              # Non-linearity
    ('model', Ridge(alpha=1.0))                                   # Regularized model
])

# Fit i predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Ceo pipeline se sačuva
joblib.dump(pipeline, 'full_pipeline.pkl')
```

---

**Key Takeaway:** Transformacije nisu obavezne - koristi ih samo kada EDA pokaže da su potrebne! Bolje je imati dobar, čist dataset nego lošu transformaciju. 🎯