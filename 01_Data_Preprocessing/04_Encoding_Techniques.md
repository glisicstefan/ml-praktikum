# Encoding Techniques

Encoding je proces **pretvaranja kategoričkih (categorical) promenljivih u numeričke** jer većina ML algoritama može raditi samo sa brojevima. Izbor odgovarajuće encoding tehnike može značajno uticati na performanse modela!

**Zašto je encoding bitan?**
- ML algoritmi rade sa matricama brojeva, ne sa tekstom
- Različite tehnike pogodne su za različite tipove podataka i algoritme
- Loš encoding može dovesti do **data leakage**, lošeg generalizovanja ili gubljenja informacija

**KRITIČNO:** Encoding se radi **POSLE** Data Cleaning i EDA, ali **PRE** Feature Scaling!

---

## Tipovi Kategoričkih Promenljivih

Pre nego što krenemo sa encoding tehnikama, važno je razumeti tipove kategoričkih podataka:

### 1. **Nominal (Neuredjene Kategorije)**
Kategorije **nemaju prirodan redosled**.

**Primeri:**
- Boje: ["Red", "Blue", "Green"]
- Gradovi: ["Belgrade", "Novi Sad", "Niš"]
- Tip proizvoda: ["Electronics", "Clothing", "Food"]
- Gender: ["Male", "Female", "Other"]

**Best Encoding:** One-Hot, Target, Frequency

### 2. **Ordinal (Uredjene Kategorije)**
Kategorije **imaju prirodan redosled**.

**Primeri:**
- Obrazovanje: ["Elementary", "High School", "Bachelor", "Master", "PhD"]
- Veličine: ["Small", "Medium", "Large", "XL"]
- Ocena: ["Poor", "Fair", "Good", "Excellent"]
- Temperature: ["Cold", "Warm", "Hot"]

**Best Encoding:** Ordinal (Label) Encoding sa definisanim redom

### 3. **Binary (Dve Kategorije)**
Samo **dve moguće vrednosti**.

**Primeri:**
- ["Yes", "No"]
- ["True", "False"]
- ["Male", "Female"]

**Best Encoding:** Label Encoding (0/1)

### 4. **High Cardinality (Mnogo Kategorija)**
Kategorije sa **mnogo unique vrednosti** (>50).

**Primeri:**
- ZIP Code (desetine hiljada)
- User ID (milioni)
- Product SKU (hiljade)
- Email domains

**Best Encoding:** Target, Frequency, Hash, Embeddings

---

## 1. Label Encoding (Labeling)

**Najjednostavnija tehnika** - svaka kategorija dobija jedinstveni broj (0, 1, 2, ...).

### Kada Koristiti?
✅ **Ordinal data** - Kada postoji prirodan redosled  
✅ **Binary categorical** - Dve kategorije (Male/Female → 0/1)  
✅ **Tree-based modeli** - Decision Trees, Random Forest, XGBoost (ne smetaju im numerički odnosi)  
✅ **Target variable encoding** - Za classification target (obavezno!)

❌ **NE koristiti za:**
- **Nominal data sa linear modelima** - Model će misliti da postoji redosled!
- **High cardinality nominal** - Nije efikasno

### Python Implementacija:
```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Primer podataka
df = pd.DataFrame({
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade', 'Niš'],
    'education': ['High School', 'Bachelor', 'Master', 'Bachelor', 'PhD']
})

# 1. Sklearn LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

print(df[['city', 'city_encoded']])
#       city  city_encoded
# 0  Belgrade             0
# 1  Novi Sad             1
# 2       Niš             2
# 3  Belgrade             0
# 4       Niš             2

# Mapping
print("Mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
# {'Belgrade': 0, 'Niš': 1, 'Novi Sad': 2}

# Inverse transform (vraćanje u original)
df['city_decoded'] = le.inverse_transform(df['city_encoded'])
print(df['city_decoded'].tolist())
# ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade', 'Niš']

# 2. Pandas .map() metoda (manuelno)
city_mapping = {'Belgrade': 0, 'Novi Sad': 1, 'Niš': 2}
df['city_manual'] = df['city'].map(city_mapping)

# 3. Pandas .factorize() - brže za veliki dataset
df['city_factorized'], unique_cities = pd.factorize(df['city'])
print("Unique:", unique_cities)
```

### Ordinal Encoding (sa definisanim redom):
```python
# Za ordinal data - MORAMO definisati redosled!
education_order = ['Elementary', 'High School', 'Bachelor', 'Master', 'PhD']
education_mapping = {edu: i for i, edu in enumerate(education_order)}

df['education_encoded'] = df['education'].map(education_mapping)

print(df[['education', 'education_encoded']])
#     education  education_encoded
# 0  High School                  1
# 1     Bachelor                  2
# 2       Master                  3
# 3     Bachelor                  2
# 4          PhD                  4

# Ili sa OrdinalEncoder (sklearn)
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[education_order])
df['education_ordinal'] = oe.fit_transform(df[['education']])
```

### ⚠️ OPASNOST - Nominal Data sa Linear Modelima:
```python
# LOŠE za Linear Models!
df = pd.DataFrame({
    'color': ['Red', 'Blue', 'Green', 'Red', 'Green']
})

le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
# Blue=0, Green=1, Red=2

# Problem: Linear model misli da je Red (2) "duplo bolji" od Blue (0)!
# I da je Green (1) "između" Blue i Red - što nema smisla!

# Linear Regression će naći koeficijent za color_encoded, 
# kao da je numerički odnos stvaran!
```

**Rešenje:** Koristi One-Hot Encoding za nominal + linear modele!

---

## 2. One-Hot Encoding (Dummy Encoding)

**Najčešće korišćena tehnika** - svaka kategorija postaje **nova binarna kolona**.

### Kako Radi?
Za N kategorija, kreiraj N binarnih kolona (ili N-1 sa `drop_first`).
```
Original:
  color
0   Red
1  Blue
2 Green

One-Hot:
  color_Red  color_Blue  color_Green
0         1           0            0
1         0           1            0
2         0           0            1
```

### Kada Koristiti?
✅ **Nominal data** - Neuredjene kategorije  
✅ **Linear models** - Logistic Regression, Linear Regression, SVM  
✅ **Neural Networks** - Obično kao input layer  
✅ **Low cardinality** - < 15-20 kategorija

❌ **NE koristiti za:**
- **High cardinality** - Pravi previše kolona (curse of dimensionality)
- **Tree-based modeli sa mnogo kategorija** - Label Encoding je dovoljno dobar
- **Memory constraints** - Sparse matrix zauzeće mnogo RAM-a

### Python Implementacija:
```python
import pandas as pd

df = pd.DataFrame({
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade', 'Niš'],
    'education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master']
})

# 1. Pandas get_dummies() - Najjednostavnije
df_encoded = pd.get_dummies(df, columns=['city'], prefix='city')

print(df_encoded)
#   education  city_Belgrade  city_Niš  city_Novi Sad
# 0  Bachelor              1         0              0
# 1    Master              0         0              1
# 2  Bachelor              0         1              0
# 3       PhD              1         0              0
# 4    Master              0         1              0

# Drop first category (izbegavaj multicollinearity)
df_encoded = pd.get_dummies(df, columns=['city'], prefix='city', drop_first=True)
print(df_encoded)
#   education  city_Niš  city_Novi Sad
# 0  Bachelor         0              0  # Belgrade je referentna kategorija (implicit 0)
# 1    Master         0              1
# 2  Bachelor         1              0

# 2. Sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')  # sparse=False za dense array
city_encoded = ohe.fit_transform(df[['city']])

print("Categories:", ohe.categories_)
print("Encoded shape:", city_encoded.shape)
print(city_encoded)

# Feature names
feature_names = ohe.get_feature_names_out(['city'])
df_ohe = pd.DataFrame(city_encoded, columns=feature_names)
print(df_ohe)

# 3. Sparse matrix (za velike dataset-e)
ohe_sparse = OneHotEncoder(sparse_output=True)  # sparse=True
city_sparse = ohe_sparse.fit_transform(df[['city']])
print("Sparse matrix type:", type(city_sparse))
print("Memory usage:", city_sparse.data.nbytes + city_sparse.indptr.nbytes + city_sparse.indices.nbytes, "bytes")
```

### Handling Unknown Categories:
```python
# Trening data
train = pd.DataFrame({'city': ['Belgrade', 'Novi Sad', 'Niš']})

# Test data SA NOVOM kategorijom
test = pd.DataFrame({'city': ['Belgrade', 'Subotica']})  # Subotica nije u train!

# OneHotEncoder sa handle_unknown='ignore'
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(train[['city']])

train_encoded = ohe.transform(train[['city']])
test_encoded = ohe.transform(test[['city']])

print("Test encoded:")
print(test_encoded)
# [[1. 0. 0.]  # Belgrade - OK
#  [0. 0. 0.]] # Subotica - Sve 0 (unknown)

# Ili 'error' - baci grešku ako vidi unknown
ohe_strict = OneHotEncoder(handle_unknown='error')
ohe_strict.fit(train[['city']])
# ohe_strict.transform(test[['city']])  # ValueError!
```

### Drop First ili Ne?
```python
# SA drop_first=True (N-1 kolona)
# Prednosti: Izbegava multicollinearity (za linear modele)
# Mane: Gubi interpretability (implicit kategorija)

df_drop = pd.get_dummies(df, columns=['city'], drop_first=True)
# city_Niš, city_Novi Sad (Belgrade je implicit 0,0)

# BEZ drop_first (N kolona)
# Prednosti: Sve kategorije eksplicitne, lakše interpretirati
# Mane: Multicollinearity problem za linear modele

df_no_drop = pd.get_dummies(df, columns=['city'], drop_first=False)
# city_Belgrade, city_Niš, city_Novi Sad

# Preporuka:
# - Linear models → drop_first=True (obavezno!)
# - Tree models → drop_first=False (bolja interpretacija)
# - Neural Networks → drop_first=False
```

---

## 3. Target Encoding (Mean Encoding)

**Moćna tehnika** - zamenjuje kategoriju sa **prosečnom vrednošću target-a** za tu kategoriju.

### Kako Radi?
Za svaku kategoriju, izračunaj mean (prosek) target variable.
```
Original:
  city      target
0 Belgrade      1
1 Belgrade      1
2 Niš          0
3 Niš          0
4 Novi Sad     1

Target Encoding:
  city      target  city_encoded
0 Belgrade      1         1.0    # Mean of Belgrade targets = (1+1)/2
1 Belgrade      1         1.0
2 Niš          0         0.0    # Mean of Niš targets = (0+0)/2
3 Niš          0         0.0
4 Novi Sad     1         1.0    # Mean of Novi Sad = 1/1
```

### Kada Koristiti?
✅ **High cardinality nominal** - ZIP codes, user IDs (hiljade kategorija)  
✅ **Strong relationship sa target-om** - Kategorija direktno utiče na target  
✅ **Tree-based modeli** - XGBoost, LightGBM (obožavaju target encoding!)  
✅ **Kaggle competitions** - Često pobednička tehnika

❌ **NE koristiti bez:**
- **Regularization** - Lako vodi u overfitting!
- **Cross-validation** - MORA se raditi properly da bi se izbegao data leakage
- **Smoothing** - Za kategorije sa malo podataka

### Python Implementacija:
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'city': ['Belgrade', 'Belgrade', 'Niš', 'Niš', 'Novi Sad', 'Belgrade', 'Niš'],
    'target': [1, 1, 0, 0, 1, 0, 1]
})

# 1. LOŠE - Direktan mean (DATA LEAKAGE!)
# NE RADI OVO na celom dataset-u!
target_mean = df.groupby('city')['target'].mean()
df['city_encoded_WRONG'] = df['city'].map(target_mean)
# Ovo koristi TARGET iz test data tokom encoding-a!

# 2. DOBRO - Fit na train, transform na test
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=42)

# Računaj mean SAMO na train
train_target_mean = train.groupby('city')['target'].mean()
print("Train target means:")
print(train_target_mean)

# Primeni na train
train['city_encoded'] = train['city'].map(train_target_mean)

# Primeni ISTI mapping na test
test['city_encoded'] = test['city'].map(train_target_mean)

# Handle unknown categories u test-u
global_mean = train['target'].mean()
test['city_encoded'].fillna(global_mean, inplace=True)  # Unknown → global mean

# 3. Target Encoding sa Category Encoders library
from category_encoders import TargetEncoder

te = TargetEncoder(cols=['city'])
train['city_encoded'] = te.fit_transform(train['city'], train['target'])
test['city_encoded'] = te.transform(test['city'])
```

### Smoothing (Regularizacija):
```python
def target_encode_with_smoothing(train_df, test_df, cat_col, target_col, m=10):
    """
    Target encoding sa smoothing-om.
    
    m = smoothing factor (veći m = više regularizacije)
    Formula: (count * mean + m * global_mean) / (count + m)
    """
    # Global mean (fallback)
    global_mean = train_df[target_col].mean()
    
    # Group statistics
    agg = train_df.groupby(cat_col)[target_col].agg(['mean', 'count'])
    
    # Smoothed mean
    smoothed_mean = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
    
    # Apply to train
    train_encoded = train_df[cat_col].map(smoothed_mean)
    
    # Apply to test (sa global mean za unknown)
    test_encoded = test_df[cat_col].map(smoothed_mean).fillna(global_mean)
    
    return train_encoded, test_encoded

# Korišćenje
train['city_smooth'], test['city_smooth'] = target_encode_with_smoothing(
    train, test, 'city', 'target', m=10
)
```

### Cross-Validation Target Encoding (Najbolja Praksa!):
```python
from sklearn.model_selection import KFold

def cv_target_encode(X, y, cat_col, n_splits=5):
    """
    Cross-validated target encoding - NEMA data leakage!
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        # Računaj mean samo na train fold
        train_target_mean = X.iloc[train_idx].groupby(cat_col)[y.iloc[train_idx]].mean()
        
        # Primeni na validation fold
        encoded[val_idx] = X.iloc[val_idx][cat_col].map(train_target_mean)
    
    # Handle NaN (unknown kategorije)
    global_mean = y.mean()
    encoded = pd.Series(encoded).fillna(global_mean).values
    
    return encoded

# Korišćenje
df['city_cv_encoded'] = cv_target_encode(df, df['target'], 'city', n_splits=5)
```

### ⚠️ DATA LEAKAGE OPASNOST!
```python
# 🚨 NIKADA NE RADI OVO:
df['city_encoded'] = df.groupby('city')['target'].transform('mean')
# Koristi CELU target kolonu uključujući test data!

# ✅ UVEK RADI OVO:
# 1. Split prvo
train, test = train_test_split(df, test_size=0.2)

# 2. Fit samo na train
target_mean = train.groupby('city')['target'].mean()

# 3. Transform i train i test sa ISTIM mapping-om
train['city_encoded'] = train['city'].map(target_mean)
test['city_encoded'] = test['city'].map(target_mean).fillna(train['target'].mean())
```

---

## 4. Frequency Encoding (Count Encoding)

Zamenjuje kategoriju sa **frekvencijom njenog pojavljivanja**.

### Kako Radi?
Broj puta koliko se kategorija pojavljuje / ukupan broj redova.
```
Original:
  city
0 Belgrade
1 Belgrade
2 Belgrade
3 Niš
4 Novi Sad

Frequency Encoding:
  city       city_freq  city_count
0 Belgrade       0.6          3
1 Belgrade       0.6          3
2 Belgrade       0.6          3
3 Niš            0.2          1
4 Novi Sad       0.2          1
```

### Kada Koristiti?
✅ **Frekvencija je signal** - Česte kategorije su važnije  
✅ **High cardinality** - Alternativa za target encoding bez leakage rizika  
✅ **Tree-based modeli** - Natural fit  
✅ **Kombinacija sa drugim encodings** - Frequency + One-Hot

### Python Implementacija:
```python
# 1. Frequency encoding (proportion)
freq_map = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq_map)

print(df[['city', 'city_freq']])
#       city  city_freq
# 0  Belgrade        0.6
# 1  Belgrade        0.6
# 2  Belgrade        0.6
# 3       Niš        0.2
# 4  Novi Sad        0.2

# 2. Count encoding (absolute counts)
count_map = df['city'].value_counts()
df['city_count'] = df['city'].map(count_map)

print(df[['city', 'city_count']])
#       city  city_count
# 0  Belgrade           3
# 1  Belgrade           3
# 2  Belgrade           3
# 3       Niš           1
# 4  Novi Sad           1

# 3. Proper train-test split
train, test = train_test_split(df, test_size=0.2)

# Fit na train
train_freq = train['city'].value_counts(normalize=True)

# Transform train i test
train['city_freq'] = train['city'].map(train_freq)
test['city_freq'] = test['city'].map(train_freq).fillna(0)  # Unknown → 0
```

---

## 5. Binary Encoding

**Kompromis** između Label Encoding i One-Hot - konvertuje u **binary digits**.

### Kako Radi?
1. Label Encoding (0, 1, 2, 3, ...)
2. Konvertuj u binary (00, 01, 10, 11, ...)
3. Svaki bit postaje kolona
```
Original:
  color
0   Red      (Label: 0 → Binary: 00)
1  Blue      (Label: 1 → Binary: 01)
2 Green      (Label: 2 → Binary: 10)
3 Yellow     (Label: 3 → Binary: 11)

Binary Encoding:
  color_0  color_1
0        0        0
1        0        1
2        1        0
3        1        1
```

### Kada Koristiti?
✅ **High cardinality** - Mnogo kategorija, ne želiš One-Hot eksploziju  
✅ **Memory efficiency** - log₂(N) kolona umesto N kolona  
✅ **Tree-based modeli** - Mogu split na binary features

**Poređenje:**
- 100 kategorija → One-Hot: 100 kolona, Binary: 7 kolona (2^7 = 128)
- 1000 kategorija → One-Hot: 1000 kolona, Binary: 10 kolona

### Python Implementacija:
```python
from category_encoders import BinaryEncoder

df = pd.DataFrame({
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Subotica', 'Kragujevac', 
             'Belgrade', 'Niš', 'Novi Sad']
})

# Binary encoding
be = BinaryEncoder(cols=['city'])
df_encoded = be.fit_transform(df)

print(df_encoded)
#       city  city_0  city_1  city_2
# 0  Belgrade       0       0       1
# 1  Novi Sad       0       1       0
# 2       Niš       0       1       1
# 3  Subotica       1       0       0
# 4  Kragujevac     1       0       1

# Broj kolona
n_categories = df['city'].nunique()
n_binary_cols = int(np.ceil(np.log2(n_categories)))
print(f"Kategorije: {n_categories}, Binary kolone: {n_binary_cols}")
```

---

## 6. Hashing (Hash Encoding)

**Feature hashing** - koristi hash funkciju za mapiranje kategorija na fiksni broj kolona.

### Kako Radi?
Hash funkcija: category → integer → mod(n_features) → bucket

### Kada Koristiti?
✅ **Extreme high cardinality** - Milioni kategorija (emails, URLs)  
✅ **Online learning** - Nove kategorije mogu biti dodane bez retraining  
✅ **Memory constraints** - Fiksni broj features bez obzira na broj kategorija  
✅ **Text data** - Kombinacija sa TF-IDF

❌ **Mane:**
- **Hash collisions** - Različite kategorije mogu mapirati u isti bucket
- **Loss of interpretability** - Ne možeš znati šta bucket predstavlja

### Python Implementacija:
```python
from sklearn.feature_extraction import FeatureHasher
from category_encoders import HashingEncoder

# 1. Sklearn FeatureHasher
hasher = FeatureHasher(n_features=10, input_type='string')
hashed = hasher.transform(df['city'])
df_hashed = pd.DataFrame(hashed.toarray(), 
                         columns=[f'hash_{i}' for i in range(10)])

print(df_hashed.head())

# 2. Category Encoders HashingEncoder
he = HashingEncoder(cols=['city'], n_components=8)
df_encoded = he.fit_transform(df)
print(df_encoded)

# 3. Manual hash
def hash_encode(value, n_features=10):
    """Simple hash encoding"""
    hash_val = hash(value)
    return hash_val % n_features

df['city_hash'] = df['city'].apply(lambda x: hash_encode(x, n_features=10))
```

---

## 7. Leave-One-Out Encoding

**Variant target encoding-a** - za svaki red, koristi mean od **svih ostalih redova** te kategorije (izuzmi trenutni).

### Kako Radi?
```
Za red i kategorije C:
LOO_mean[i] = (sum(target where category=C) - target[i]) / (count(C) - 1)
```

### Kada Koristiti?
✅ **Smanjuje overfitting** - Više od standardnog target encoding  
✅ **Regularizacija** - Ne koristi sopstvenu target vrednost  
✅ **High signal-to-noise kategorije**

### Python Implementacija:
```python
from category_encoders import LeaveOneOutEncoder

# Leave-One-Out Encoding
loo = LeaveOneOutEncoder(cols=['city'])
train['city_loo'] = loo.fit_transform(train['city'], train['target'])
test['city_loo'] = loo.transform(test['city'])

# Manual implementation
def leave_one_out_encode(df, cat_col, target_col):
    """
    Leave-One-Out encoding manuelno.
    """
    # Group sum i count
    grouped = df.groupby(cat_col)[target_col].agg(['sum', 'count'])
    
    # Za svaki red: (sum - current) / (count - 1)
    def loo_mean(row):
        cat = row[cat_col]
        target = row[target_col]
        total_sum = grouped.loc[cat, 'sum']
        total_count = grouped.loc[cat, 'count']
        return (total_sum - target) / (total_count - 1)
    
    return df.apply(loo_mean, axis=1)

df['city_loo_manual'] = leave_one_out_encode(df, 'city', 'target')
```

---

## 8. Weight of Evidence (WoE) Encoding

**Za binary classification** - meri jačinu prediktivne moći kategorije.

### Formula:
```
WoE = ln(% of positive / % of negative)
```

### Kada Koristiti?
✅ **Binary classification** - Samo za 2-class probleme  
✅ **Logistic Regression** - Natural fit  
✅ **Credit scoring** - Tradicionalno se koristi u finansijama  
✅ **Interpretability** - Pozitivan WoE = više positives u toj kategoriji

### Python Implementacija:
```python
from category_encoders import WOEEncoder

# Weight of Evidence encoding
woe = WOEEncoder(cols=['city'])
train['city_woe'] = woe.fit_transform(train['city'], train['target'])
test['city_woe'] = woe.transform(test['city'])

# Manual WoE calculation
def calculate_woe(df, cat_col, target_col):
    """
    Weight of Evidence encoding za binary target.
    """
    # Total positives i negatives
    total_pos = (df[target_col] == 1).sum()
    total_neg = (df[target_col] == 0).sum()
    
    woe_dict = {}
    for category in df[cat_col].unique():
        cat_data = df[df[cat_col] == category]
        
        # Positives i negatives u ovoj kategoriji
        cat_pos = (cat_data[target_col] == 1).sum()
        cat_neg = (cat_data[target_col] == 0).sum()
        
        # Proporције
        pct_pos = (cat_pos + 0.5) / total_pos  # +0.5 smoothing
        pct_neg = (cat_neg + 0.5) / total_neg
        
        # WoE
        woe = np.log(pct_pos / pct_neg)
        woe_dict[category] = woe
    
    return df[cat_col].map(woe_dict)

df['city_woe_manual'] = calculate_woe(df, 'city', 'target')
```

---

## Kada Koristiti Koju Tehniku? - Decision Tree
```
Koja je priroda kategorije?
│
├─→ ORDINAL (ima redosled)?
│   └─→ Ordinal/Label Encoding sa definisanim redom
│
├─→ BINARY (samo 2 kategorije)?
│   └─→ Label Encoding (0/1)
│
├─→ NOMINAL (nema redosled)?
│   │
│   ├─→ LOW Cardinality (< 15 kategorija)?
│   │   │
│   │   ├─→ Koristiš LINEAR model?
│   │   │   └─→ One-Hot Encoding (drop_first=True)
│   │   │
│   │   └─→ Koristiš TREE model?
│   │       ├─→ One-Hot (bolja interpretacija)
│   │       └─→ ILI Label Encoding (jednostavnije)
│   │
│   └─→ HIGH Cardinality (> 50 kategorija)?
│       │
│       ├─→ Jak signal za target?
│       │   └─→ Target Encoding (sa CV i smoothing!)
│       │
│       ├─→ Frekvencija je važna?
│       │   └─→ Frequency/Count Encoding
│       │
│       ├─→ EXTREME cardinality (>10,000)?
│       │   ├─→ Hash Encoding
│       │   └─→ ILI Binary Encoding
│       │
│       └─→ Redukuj dimensionalnost?
│           └─→ Binary Encoding (log₂(N) kolona)
```

---

## Encoding by Algorithm Type

| Algoritam | LOW Cardinality Nominal | HIGH Cardinality Nominal | Ordinal | Binary |
|-----------|------------------------|--------------------------|---------|--------|
| **Linear/Logistic** | One-Hot (MUST!) | Target, Frequency | Label (ordered) | Label 0/1 |
| **Decision Trees** | One-Hot (preferred), Label (OK) | Target, Label | Label (ordered) | Label 0/1 |
| **Random Forest** | One-Hot (preferred), Label (OK) | Target (best!), Label | Label (ordered) | Label 0/1 |
| **XGBoost/LightGBM** | One-Hot, Label (both OK) | Target (best!), Label | Label (ordered) | Label 0/1 |
| **SVM** | One-Hot (MUST!) | Target, Frequency | One-Hot | Label 0/1 |
| **KNN** | One-Hot (MUST!) | Target, Frequency | Label (ordered) | Label 0/1 |
| **Naive Bayes** | One-Hot, Label | One-Hot, Label | Label (ordered) | Label 0/1 |
| **Neural Networks** | One-Hot, Embeddings | Embeddings, Target | Embeddings, Label | Label 0/1 |

**Legenda:**
- **MUST!** - Druge opcije daju loše rezultate
- **best!** - Optimalna strategija
- **preferred** - Bolja opcija, ali alternativa radi
- **ordered** - Obavezno definiši redosled kategorija

---

## Best Practices - Encoding Checklist

### ✅ DO:

1. **Split PRE encoding**
```python
# UVEK prvo split, PA ONDA encode!
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Fit encoding na train
encoder.fit(train[['category']], train['target'])

# Transform i train i test
train_encoded = encoder.transform(train[['category']])
test_encoded = encoder.transform(test[['category']])
```

2. **Handle unknown categories**
```python
# One-Hot
ohe = OneHotEncoder(handle_unknown='ignore')

# Target Encoding
test_encoded = test['city'].map(train_target_mean).fillna(train['target'].mean())

# Frequency Encoding
test_freq = test['city'].map(train_freq).fillna(0)
```

3. **Dokumentuj odluke**
```python
encoding_strategy = {
    'city': 'target_encoding',  # High cardinality, strong signal
    'education': 'ordinal',      # Natural order
    'gender': 'label',           # Binary
    'color': 'onehot',          # Nominal, low cardinality
    'user_id': 'hash'           # Extreme high cardinality
}
```

4. **Cross-validate target encoding**
```python
# NIKAD direktan mean na celom dataset-u!
# UVEK koristi CV ili proper train-test split
```

5. **Save encoders za production**
```python
import joblib

# Save
joblib.dump(encoder, 'category_encoder.pkl')

# Load u production
encoder = joblib.load('category_encoder.pkl')
new_data_encoded = encoder.transform(new_data)
```

### ❌ DON'T:

1. **Ne encoduj pre split-a**
```python
# 🚨 LOŠE - Data leakage!
df['city_encoded'] = df.groupby('city')['target'].transform('mean')
train, test = train_test_split(df)

# ✅ DOBRO
train, test = train_test_split(df)
train_mean = train.groupby('city')['target'].mean()
train['city_encoded'] = train['city'].map(train_mean)
test['city_encoded'] = test['city'].map(train_mean)
```

2. **Ne koristi Label za nominal + linear modeli**
```python
# 🚨 LOŠE
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])  # Red=0, Blue=1, Green=2
# Linear model misli da postoji redosled!

# ✅ DOBRO
df_encoded = pd.get_dummies(df, columns=['color'])
```

3. **Ne zaboravi One-Hot sa drop_first za linear modele**
```python
# 🚨 Multicollinearity problem
df = pd.get_dummies(df, columns=['city'], drop_first=False)

# ✅ DOBRO za Linear Regression
df = pd.get_dummies(df, columns=['city'], drop_first=True)
```

4. **Ne zanemaruj memory sa high cardinality**
```python
# 🚨 10,000 kategorija → One-Hot → 10,000 kolona → RAM explodes!

# ✅ DOBRO
# Target Encoding, Binary Encoding, ili Hash Encoding
```

---

## Practical Examples - Complete Pipeline

### Primer 1: Mixed Categorical Types
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

# Dataset
df = pd.DataFrame({
    'education': ['High School', 'Bachelor', 'Master', 'Bachelor', 'PhD', 'High School'],
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade', 'Subotica', 'Niš'],
    'color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
    'user_id': ['u1', 'u2', 'u3', 'u4', 'u5', 'u6'],
    'target': [1, 0, 1, 1, 0, 0]
})

# Split
train, test = train_test_split(df, test_size=0.3, random_state=42)

# 1. Ordinal (education)
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
education_map = {edu: i for i, edu in enumerate(education_order)}
train['education_encoded'] = train['education'].map(education_map)
test['education_encoded'] = test['education'].map(education_map)

# 2. Target Encoding (city - high cardinality nominal)
te = TargetEncoder(cols=['city'])
train['city_encoded'] = te.fit_transform(train['city'], train['target'])
test['city_encoded'] = te.transform(test['city'])

# 3. One-Hot (color - low cardinality nominal)
train_ohe = pd.get_dummies(train, columns=['color'], prefix='color', drop_first=True)
test_ohe = pd.get_dummies(test, columns=['color'], prefix='color', drop_first=True)

# Align columns (ako test nema neke kategorije)
test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

# 4. Hash Encoding (user_id - extreme high cardinality)
from category_encoders import HashingEncoder
he = HashingEncoder(cols=['user_id'], n_components=4)
train['user_hash'] = he.fit_transform(train['user_id'])
test['user_hash'] = he.transform(test['user_id'])

print("Train encoded:")
print(train_ohe.head())
```

### Primer 2: Sklearn Pipeline Integration
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier

# Column groups
ordinal_features = ['education']
onehot_features = ['color']
target_features = ['city']
numerical_features = ['age', 'income']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat_onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), onehot_features),
        # Target encoding mora biti u posebnom koraku zbog target dependency
    ],
    remainder='passthrough'
)

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Note: Target encoding ne ide dobro u standard pipeline jer treba target
# Bolje ga uraditi pre pipeline-a ili koristi custom transformer
```

---

## Advanced: Custom Transformer za Target Encoding
```python
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """
    Target Encoder sa cross-validation i smoothing.
    """
    def __init__(self, cols, smoothing=10, cv=5):
        self.cols = cols
        self.smoothing = smoothing
        self.cv = cv
        self.encodings_ = {}
        self.global_means_ = {}
    
    def fit(self, X, y):
        from sklearn.model_selection import KFold
        
        for col in self.cols:
            # Global mean za fallback
            self.global_means_[col] = y.mean()
            
            # CV encoding
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            col_encoded = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                
                # Group stats
                stats = X_train.groupby(col).agg({col: 'count'})
                stats.columns = ['count']
                stats['mean'] = X_train.groupby(col)[y_train.name].mean()
                
                # Smoothed mean
                smoothed = (stats['count'] * stats['mean'] + 
                           self.smoothing * self.global_means_[col]) / (
                           stats['count'] + self.smoothing)
                
                # Encode validation
                col_encoded[val_idx] = X.iloc[val_idx][col].map(smoothed)
            
            # Finalni mapping na celom train
            stats = X.groupby(col).agg({col: 'count'})
            stats.columns = ['count']
            stats['mean'] = X.groupby(col)[y.name].mean()
            
            self.encodings_[col] = (stats['count'] * stats['mean'] + 
                                   self.smoothing * self.global_means_[col]) / (
                                   stats['count'] + self.smoothing)
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            X_copy[col] = X_copy[col].map(self.encodings_[col]).fillna(
                self.global_means_[col]
            )
        return X_copy

# Korišćenje
te_cv = TargetEncoderCV(cols=['city', 'region'], smoothing=10, cv=5)
X_train_encoded = te_cv.fit_transform(X_train, y_train)
X_test_encoded = te_cv.transform(X_test)
```

---

## Encoding Impact - Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

def compare_encodings(df, cat_col, target_col):
    """
    Vizualno poredi različite encoding tehnike.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Original categories
    df[cat_col].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Original Distribution')
    
    # Label Encoding
    le = LabelEncoder()
    label_encoded = le.fit_transform(df[cat_col])
    axes[0, 1].hist(label_encoded, bins=20)
    axes[0, 1].set_title('Label Encoding')
    
    # Target Encoding
    target_mean = df.groupby(cat_col)[target_col].mean()
    target_encoded = df[cat_col].map(target_mean)
    axes[0, 2].hist(target_encoded, bins=20)
    axes[0, 2].set_title('Target Encoding')
    
    # Frequency Encoding
    freq = df[cat_col].value_counts(normalize=True)
    freq_encoded = df[cat_col].map(freq)
    axes[1, 0].hist(freq_encoded, bins=20)
    axes[1, 0].set_title('Frequency Encoding')
    
    # One-Hot (pokazuje broj kolona)
    ohe = pd.get_dummies(df[cat_col])
    axes[1, 1].bar(range(ohe.shape[1]), [1]*ohe.shape[1])
    axes[1, 1].set_title(f'One-Hot: {ohe.shape[1]} columns')
    
    # Target vs Encoded (scatter)
    axes[1, 2].scatter(target_encoded, df[target_col], alpha=0.5)
    axes[1, 2].set_xlabel('Target Encoded Values')
    axes[1, 2].set_ylabel('Actual Target')
    axes[1, 2].set_title('Target Encoding vs Target')
    
    plt.tight_layout()
    plt.show()

# Korišćenje
compare_encodings(df, 'city', 'target')
```

---

## Rezime - Encoding Quick Reference

| Encoding | Cardinality | Nominal/Ordinal | Best For | Avoid For |
|----------|-------------|-----------------|----------|-----------|
| **Label** | Any | Ordinal, Binary | Trees, Ordinal data | Linear + Nominal |
| **One-Hot** | Low (<15) | Nominal | Linear models, NNs | High cardinality |
| **Target** | High (>50) | Nominal | Trees, High card | Without CV (leakage!) |
| **Frequency** | Any | Nominal | When freq matters | When freq is noise |
| **Binary** | High (>50) | Nominal | Memory constraints | Simple problems |
| **Hash** | Extreme (>10k) | Nominal | Online learning | Interpretability needed |
| **WoE** | Any | Nominal | Binary classification | Multi-class |
| **Leave-One-Out** | Medium | Nominal | Reduce overfitting | Small datasets |

---

**Key Takeaway:** Encoding nije "one-size-fits-all"! Prava tehnika zavisi od: tipa podataka, algoritma, cardinalitya, i da li postoji signal u kategorijama. Target Encoding je moćan, ali opasan bez proper CV! 🎯