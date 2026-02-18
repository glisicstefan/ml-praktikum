# Data Cleaning

Data Cleaning je proces identifikovanja i ispravljanja (ili uklanjanja) grešaka, nekonzistentnosti i problema u podacima. Ovo je **KRITIČAN** korak jer kvalitet podataka direktno utiče na performanse modela.

**Pravilo:** "Garbage In, Garbage Out" - Loši podaci = Loš model

Raw (sirovi) podaci su uglavnom **noisy** (šumni) i **incomplete** (nepotpuni), što može drastično da utiče na performanse modela. Nekada 80% vremena u ML projektu ode na čišćenje podataka!

---

## 1. Duplicate Values (Duple Vrednosti)

Potpuno identični redovi koji se ponavljaju u datasetu.

### Zašto su problem?
- **Artificial bias** - Model vidi iste podatke više puta i "misli" da su važniji
- **Overfitting** - Model pamti duplicate umesto da uči opšte obrasce
- **Pogrešne metrike** - Evaluacija nije realna ako test set ima duplikate
- **Veći dataset** - Sporiji trening bez koristi

### Kako detektovati?
```python
# Provera duplikata
print(f"Broj duplikata: {df.duplicated().sum()}")

# Prikaz duplikata
duplicates = df[df.duplicated(keep=False)]
print(duplicates)

# Duplikati na osnovu specifičnih kolona
df.duplicated(subset=['email', 'user_id'])
```

### Kako rešiti?
```python
# Uklanjanje svih duplikata (zadržava prvi)
df_clean = df.drop_duplicates()

# Zadržavanje poslednjeg pojavljivanja
df_clean = df.drop_duplicates(keep='last')

# Provera samo specifičnih kolona
df_clean = df.drop_duplicates(subset=['user_id', 'date'])
```

### Best Practices:
- **Uvek proveri** razlog duplikata - da li su greška ili validni podaci?
- Za **time-series** - možda želiš najnoviji zapis (keep='last')
- Za **transakcije** - duplikati mogu biti validni (ista osoba kupuje više puta)

---

## 2. Irrelevant Columns (Nebitne Kolone)

Kolone koje ne doprinose cilju analize ili predviđanju.

### Šta spada ovde?
- **Identifikatori** - ID kolone, row numbers (ako nisu feature)
- **Constant kolone** - Ista vrednost za sve redove
- **High cardinality** - Previše unique vrednosti (npr. timestamp sa milisekundama)
- **Data leakage** - Kolone koje ne bi bile dostupne u produkciji
- **PII (Personal Identifiable Information)** - Imena, email-ovi (privacy concern)

### Kako identifikovati?
```python
# Konstantne kolone
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Konstantne kolone: {constant_cols}")

# Kolone sa jednom vrednošću
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"{col}: {df[col].unique()}")

# High cardinality check
for col in df.select_dtypes(include='object').columns:
    unique_ratio = df[col].nunique() / len(df)
    if unique_ratio > 0.95:  # 95% unique vrednosti
        print(f"{col}: {unique_ratio:.2%} unique")
```

### Kako rešiti?
```python
# Brisanje specifičnih kolona
df_clean = df.drop(columns=['ID', 'row_number', 'timestamp_ms'])

# Brisanje svih konstantnih kolona
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df_clean = df.drop(columns=constant_cols)

# Brisanje kolona sa previše missing values
threshold = 0.7  # 70% missing
missing_cols = df.columns[df.isnull().mean() > threshold]
df_clean = df.drop(columns=missing_cols)
```

### Best Practices:
- **Domain expertise** - Konsultuj se sa ekspertima da li je kolona zaista nebitna
- **Feature engineering** - Možda možeš izvući korisne informacije (npr. iz timestampa izvuci hour_of_day)
- **PII handling** - Hash ili enkoduj umesto brisanja ako može biti koristan signal

---

## 3. Redundant Columns (Redundantne Kolone)

Kolone koje su gotovo identične ili visoko korelisane sa drugim kolonama.

### Tipovi redundancije:

#### A) **Potpuna redundancija** - Iste informacije različito prezentovane
```python
# Primer:
# temperature_celsius = 25
# temperature_fahrenheit = 77
# (potpuno iste informacije)

# Detekcija
df['temp_f'] = df['temp_c'] * 9/5 + 32
df.corr()  # correlation = 1.0
```

#### B) **Visoka korelacija** - Kolone koje nose istu informaciju
```python
# Računanje correlation matrix
correlation_matrix = df.corr().abs()

# Pronalaženje parova sa visokom korelacijom
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# Kolone sa korelacijom > 0.95
high_corr = [column for column in upper_triangle.columns 
             if any(upper_triangle[column] > 0.95)]
print(f"Visoko korelisane kolone: {high_corr}")
```

#### C) **Multicollinearity** - Problem za linearne modele
```python
# VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                   for i in range(len(df.columns))]
# VIF > 10 = problem
print(vif_data[vif_data["VIF"] > 10])
```

### Kako rešiti?
```python
# 1. Ručno brisanje poznate redundancije
df_clean = df.drop(columns=['temperature_fahrenheit'])

# 2. Automatsko uklanjanje visoko korelisanih
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns 
               if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

df_clean = remove_correlated_features(df, threshold=0.95)
```

### Best Practices:
- **Tree-based modeli** (Random Forest, XGBoost) - Nisu osetljivi na multicollinearity
- **Linear modeli** (Linear Regression, Logistic) - MORAJU se rešiti korelisane feature
- **Feature importance** - Zadrži kolonu koja ima veći uticaj na target
- **Domain knowledge** - Možda obe nose različite nijanse informacija

---

## 4. Data Format Standardization (Standardizacija Formata)

Različiti formati za iste tipove podataka mogu stvoriti probleme.

### Česti problemi:

#### A) **Date/Time formati**
```python
# Problem: Različiti formati
# "2024-01-15", "01/15/2024", "15-Jan-2024", "2024.01.15"

# Rešenje:
df['date'] = pd.to_datetime(df['date'], errors='coerce')
# errors='coerce' pretvara invalide u NaT (Not a Time)

# Standardizovan format
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
```

#### B) **Text inconsistencies**
```python
# Problem: "New York", "new york", "NEW YORK", "ny", "NY"

# Rešenje:
df['city'] = df['city'].str.lower().str.strip()

# Mapping različitih verzija
city_mapping = {
    'ny': 'new york',
    'nyc': 'new york',
    'la': 'los angeles',
    'sf': 'san francisco'
}
df['city'] = df['city'].replace(city_mapping)
```

#### C) **Numerical formats**
```python
# Problem: "1,234.56" vs "1.234,56" (US vs EU format)
# Problem: "$1,234" sa currency simbolima

# Rešenje:
df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
```

#### D) **Boolean inconsistencies**
```python
# Problem: True/False, "yes"/"no", 1/0, "Y"/"N"

# Rešenje:
bool_mapping = {
    'yes': True, 'y': True, '1': True, 'true': True,
    'no': False, 'n': False, '0': False, 'false': False
}
df['active'] = df['active'].str.lower().map(bool_mapping)
```

#### E) **Units of measurement**
```python
# Problem: "5kg", "5 kg", "5000g", "11 lbs"

# Rešenje: Izvuci broj i konvertuj
df['weight_kg'] = df['weight'].str.extract('(\d+\.?\d*)').astype(float)

# Konverzija jedinica
df.loc[df['weight'].str.contains('lbs'), 'weight_kg'] *= 0.453592
df.loc[df['weight'].str.contains('g'), 'weight_kg'] /= 1000
```

### Best Practices:
- **Standardizuj pre bilo koje analize**
- **Lowercase sve text** kolone (osim ako case nije bitan)
- **Strip whitespace** - `.str.strip()`
- **Validacija** - Proveri da li su konverzije uspele

---

## 5. Outliers (Odstupajuće Vrednosti)

Outliers su vrednosti koje drastično odstupaju od ostalih podataka.

### Tipovi outliers:

**1. Natural Outliers** - Validni ali ekstremni
- Primer: Bill Gates u datasetu primanja (legit milijarder)

**2. Error Outliers** - Greške u podacima
- Primer: Starost = 200 godina, Cena = -500$

### Detekcija Outliers:

#### A) **Visualizacija**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='salary')
plt.title('Salary Distribution - Outliers')
plt.show()

# Histogram
df['age'].hist(bins=50)
plt.title('Age Distribution')
plt.show()

# Scatter plot (bivariate outliers)
plt.scatter(df['height'], df['weight'])
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
```

#### B) **IQR Method (Interquartile Range)**
```python
# Najčešća metoda
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

# Definisanje outlier granica
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifikacija outliers
outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]
print(f"Outliers: {len(outliers)} / {len(df)}")
```

#### C) **Z-Score Method**
```python
from scipy import stats

# Z-score računanje
z_scores = np.abs(stats.zscore(df['salary']))

# Vrednosti sa |z| > 3 su outliers
outliers_zscore = df[z_scores > 3]
print(f"Outliers (Z-score): {len(outliers_zscore)}")
```

#### D) **Isolation Forest (ML pristup)**
```python
from sklearn.ensemble import IsolationForest

# Treniranje modela
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(df[['salary', 'age']])

# -1 = outlier, 1 = normal
df['is_outlier'] = outlier_labels
outliers_iso = df[df['is_outlier'] == -1]
```

### Tretman Outliers:

#### **Opcija 1: Remove (Brisanje)**
```python
# IQR metoda
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[
    (df['salary'] >= Q1 - 1.5*IQR) & 
    (df['salary'] <= Q3 + 1.5*IQR)
]
```

**Kada koristiti:**
- Outliers su greške u podacima
- Mali dataset i outlieri utiču drastično
- Manje od 5% podataka

#### **Opcija 2: Capping (Ograničavanje)**
```python
# Winsorization - zamena sa percentile vrednostima
lower = df['salary'].quantile(0.01)  # 1st percentile
upper = df['salary'].quantile(0.99)  # 99th percentile

df['salary_capped'] = df['salary'].clip(lower=lower, upper=upper)
```

**Kada koristiti:**
- Outliers su validni ali ekstremni
- Želiš da zadržiš sve podatke
- Smanjiti uticaj outliera na model

#### **Opcija 3: Transformation (Transformacija)**
```python
# Log transformation - smanjuje uticaj outliera
df['salary_log'] = np.log1p(df['salary'])  # log(1 + x)

# Square root transformation
df['salary_sqrt'] = np.sqrt(df['salary'])

# Box-Cox transformation (samo pozitivne vrednosti)
from scipy.stats import boxcox
df['salary_boxcox'], _ = boxcox(df['salary'] + 1)
```

**Kada koristiti:**
- Skewed distribucije
- Želiš da zadr žiš sve podatke ali smanjiti uticaj

#### **Opcija 4: Keep (Zadržavanje)**
```python
# Outlier indicator kao feature
df['is_high_earner'] = (df['salary'] > upper_bound).astype(int)
```

**Kada koristiti:**
- Outliers nose važne informacije
- Tree-based modeli (robusni na outliers)
- Fraud detection (outliers su cilj!)

### Decision Framework:
```
Da li je outlier greška? 
    ├─ DA → Remove
    └─ NE → Da li je mali dataset?
            ├─ DA → Capping ili Transformation
            └─ NE → Da li koristiš tree-based model?
                    ├─ DA → Keep (možda kao feature)
                    └─ NE → Transformation ili Capping
```

---

## 6. Missing Values (Nedostajuće Vrednosti)

Jedan od najčešćih problema u realnim podacima.

### Tipovi Missing Values:

**1. MCAR (Missing Completely At Random)**
- Nema pattern, potpuno nasumično
- Primer: Uređaj se pokvario i izgubili smo neke podatke

**2. MAR (Missing At Random)**
- Missing zavisi od drugih kolona
- Primer: Mladi ljudi ređe odgovaraju na pitanje o primanjima

**3. MNAR (Missing Not At Random)**
- Missing zavisi od same missing vrednosti
- Primer: Ljudi sa visokim primanjima ne žele da dele informacije

### Detekcija:
```python
# Ukupno missing po koloni
print(df.isnull().sum())

# Procenat missing
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.sort_values(ascending=False))

# Vizualizacija missing patterns
import missingno as msno
msno.matrix(df)
msno.heatmap(df)  # Korelacija između missing vrednosti
```

### Strategije za Rešavanje:

#### **Opcija 1: Deletion (Brisanje)**

**Listwise Deletion** - Briši cele redove
```python
# Briši redove sa bilo kojom missing vrednosti
df_clean = df.dropna()

# Briši redove gde su specifične kolone missing
df_clean = df.dropna(subset=['age', 'salary'])

# Briši ako ima više od threshold missing vrednosti
threshold = 3
df_clean = df.dropna(thresh=len(df.columns) - threshold)
```

**Columnwise Deletion** - Briši cele kolone
```python
# Briši kolone sa > 50% missing
threshold = 0.5
df_clean = df.dropna(axis=1, thresh=len(df) * threshold)
```

**Kada koristiti:**
- ✅ Manje od 5% missing vrednosti
- ✅ MCAR (potpuno random)
- ✅ Veliki dataset
- ❌ Ako gubimo previše podataka

#### **Opcija 2: Imputation (Popunjavanje)**

**A) Simple Imputation**
```python
# Mean imputation (numeričke kolone)
df['age'].fillna(df['age'].mean(), inplace=True)

# Median (bolje za outliers)
df['salary'].fillna(df['salary'].median(), inplace=True)

# Mode (kategoričke kolone)
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Konstanta
df['country'].fillna('Unknown', inplace=True)
df['score'].fillna(0, inplace=True)
```

**Kada koristiti:**
- Mean/Median: Približno normalna distribucija
- Mode: Kategoričke promenljive
- Konstanta: Kada "missing" sama nosi informaciju

**B) Forward/Backward Fill** (Time Series)
```python
# Forward fill - koristi prethodnu vrednost
df['temperature'].fillna(method='ffill', inplace=True)

# Backward fill - koristi sledeću vrednost
df['temperature'].fillna(method='bfill', inplace=True)

# Kombinovano
df['temperature'].fillna(method='ffill').fillna(method='bfill')
```

**C) Interpolation**
```python
# Linearna interpolacija
df['temperature'].interpolate(method='linear', inplace=True)

# Polynomial
df['sales'].interpolate(method='polynomial', order=2)

# Time-based
df['price'].interpolate(method='time')
```

**D) KNN Imputation**
```python
from sklearn.impute import KNNImputer

# Koristi K najbližih suseda za popunjavanje
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

**E) Model-Based Imputation**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Trenira model da predvidi missing vrednosti
imputer = IterativeImputer(random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

**F) Multiple Imputation** (MICE)
```python
# Kreira više verzija imputed dataset-a
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    sample_posterior=True  # Različite impute vrednosti
)
```

#### **Opcija 3: Indicator Variable**
```python
# Kreiraj binary kolonu koja pokazuje da li je bilo missing
df['age_was_missing'] = df['age'].isnull().astype(int)

# Zatim popuni missing vrednosti
df['age'].fillna(df['age'].median(), inplace=True)
```

**Kada koristiti:**
- Kada "missing" sama može nositi informaciju
- MNAR scenario - missing nije random

### Decision Framework:
```
Koliko je % missing?
├─ < 5% → Deletion (dropna)
├─ 5-20% → Simple Imputation (mean/median/mode)
├─ 20-40% → Advanced Imputation (KNN, IterativeImputer)
└─ > 40% → Consider dropping column ili domain-specific approach

Da li je Time Series?
└─ DA → Forward/Backward Fill ili Interpolation

Da li je kategorička?
└─ DA → Mode ili "Unknown" category

Da li postoji pattern u missing?
└─ DA → Indicator variable + Imputation
```

---

## Best Practices - Celokupno Data Cleaning

### 1. **Redosled operacija je bitan:**
```
1. Remove duplicates prvo
2. Handle irrelevant columns
3. Standardize formats
4. Handle outliers
5. Handle missing values (zadnje!)
```

### 2. **Dokumentuj sve promene:**
```python
# Log sve transformacije
print(f"Original shape: {df_original.shape}")
print(f"After duplicates: {df.shape}")
print(f"After removing columns: {df.shape}")
print(f"After outliers: {df.shape}")
print(f"After missing: {df.shape}")
```

### 3. **Čuvaj original podatke:**
```python
df_original = df.copy()  # Backup
# ... cleaning operations ...
# Uvek možeš da se vratiš
```

### 4. **Validacija posle čišćenja:**
```python
# Provera da li ima missing
assert df.isnull().sum().sum() == 0, "Still has missing values!"

# Provera data types
print(df.dtypes)

# Provera da nema duplikata
assert df.duplicated().sum() == 0, "Still has duplicates!"

# Summary statistics
print(df.describe())
```

### 5. **Visualization pre i posle:**
```python
# Pre
df_original['salary'].hist(bins=50, alpha=0.5, label='Before')
# Posle
df_clean['salary'].hist(bins=50, alpha=0.5, label='After')
plt.legend()
plt.show()
```

### 6. **Automated Data Quality Report:**
```python
def data_quality_report(df):
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicates': df.duplicated().sum(),
        'missing_cells': df.isnull().sum().sum(),
        'missing_percent': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'constant_columns': [col for col in df.columns if df[col].nunique() == 1],
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return pd.Series(report)

print(data_quality_report(df))
```

---

## Common Pitfalls (Česte Greške)

❌ **Impute missing na train+test zajedno** → Data leakage!
```python
# LOŠE
df_all = pd.concat([train, test])
df_all['age'].fillna(df_all['age'].mean())  # LEAKAGE!

# DOBRO
train['age'].fillna(train['age'].mean())
test['age'].fillna(train['age'].mean())  # Koristi mean iz train!
```

❌ **Brisanje outliers bez razumevanja**
```python
# LOŠE - možda brišeš važne podatke
df = df[df['salary'] < 1000000]  # Bill Gates out!
```

❌ **Impute pre razumevanja missing pattern**
```python
# LOŠE - možda missing nije random!
df['income'].fillna(df['income'].mean())  # Bogati ljudi ne dele info!
```

❌ **Zaboraviti da sacuvaš cleaning pipeline**
```python
# DOBRO - save logic za production
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

cleaning_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    # ... other steps
])
```

---

## Rezime

Data Cleaning je **iterativan proces**:

1. **Exploruj** → Razumi podatke
2. **Identifikuj** → Pronađi probleme
3. **Odluči** → Izaberi strategiju
4. **Primeni** → Očisti podatke
5. **Validuj** → Proveri rezultate
6. **Dokumentuj** → Zapiši šta si radio

**Kvalitet podataka > Kvantitet podataka!** 

Bolje imati 1000 čistih redova nego 10,000 šumnih. 🧹✨