# Exploratory Data Analysis (EDA)

EDA (Exploratory Data Analysis) je proces vizualizacije i analize podataka da bismo razumeli njihovu strukturu, otkrili obrasce, identifikovali anomalije i pronašli veze između promenljivih. Ovo je **KRITIČAN korak** koji ti daje insights šta da radiš u narednim fazama.

**EDA je kao "upoznavanje" sa podacima pre nego što počneš da ih modeluješ.** 🔍

---

## Zašto je EDA Bitan?

### 1. **Razumevanje Dataseta**
- Koliko podataka imamo? (broj redova i kolona)
- Koji tipovi podataka? (numerical, categorical, datetime)
- Kako su podaci distribuirani? (normal, skewed, uniform)
- Ima li balansa u klasama? (za classification probleme)

### 2. **Identifikacija Problema**
- **Missing values** - Koliko, gde, i da li postoji pattern?
- **Outliers** - Koje vrednosti drastično odstupaju?
- **Data quality issues** - Duplikati, nekonzistentnosti, greške
- **Class imbalance** - Da li jedna klasa dominira?

### 3. **Insights za Dalje Korake**
EDA direktno utiče na odluke koje donosiš:
- **Feature Engineering** - Koje feature-e kreirati na osnovu obrazaca?
- **Feature Selection** - Koje feature-e su najbitnije za target?
- **Data Transformation** - Da li treba log transform, scaling, encoding?
- **Model Selection** - Koji algoritmi bi mogli dobro da rade?

**Primer:** Ako EDA pokaže da je odnos između features i target nelinearan → Decision Trees ili Neural Networks > Linear Regression

---

## Tipovi EDA

### 1. Univariate Analysis (Analiza Jedne Promenljive)

Fokus na **jednu promenljivu** da razumemo njene karakteristike.

#### Za Numeričke Promenljive:

**Deskriptivna Statistika:**
```python
# Summary statistics
print(df['age'].describe())
# count, mean, std, min, 25%, 50%, 75%, max

# Dodatne metrike
print(f"Skewness: {df['age'].skew()}")      # Mera asimetrije
print(f"Kurtosis: {df['age'].kurtosis()}")  # Mera "težine repova"
```

**Vizualizacije:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. HISTOGRAM - Distribucija
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# 2. BOX PLOT - Outliers i quartiles
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['salary'])
plt.title('Salary - Outlier Detection')
plt.show()

# 3. DENSITY PLOT - Smoothed distribution
df['age'].plot(kind='density', figsize=(10, 6))
plt.title('Age Density Plot')
plt.show()

# 4. Q-Q PLOT - Provera normalnosti
from scipy import stats
stats.probplot(df['age'], dist="norm", plot=plt)
plt.title('Q-Q Plot - Age')
plt.show()
```

**Šta tražimo:**
- **Distribucija** - Normalna, skewed (left/right), bimodal, uniform?
- **Centralna tendencija** - Mean, median, mode
- **Spread** - Standard deviation, range, IQR
- **Outliers** - Vrednosti izvan IQR × 1.5 ili Z-score > 3
- **Skewness** - Da li treba transformacija?

#### Za Kategoričke Promenljive:
```python
# Value counts
print(df['category'].value_counts())
print(df['category'].value_counts(normalize=True))  # Proporciје

# Bar chart
df['category'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

# Pie chart
df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Category Proportions')
plt.show()
```

**Šta tražimo:**
- **Frekvencija** - Koja kategorija dominira?
- **Class imbalance** - 90% jedna klasa, 10% druga?
- **Rare categories** - Kategorije sa < 1% podataka
- **Cardinality** - Koliko unique vrednosti?

---

### 2. Bivariate Analysis (Analiza Dve Promenljive)

Fokus na **vezu između dve promenljive** - korelaciju, zavisnost ili nezavisnost.

#### Numerička vs Numerička:
```python
# 1. SCATTER PLOT - Vizualizacija odnosa
plt.figure(figsize=(10, 6))
plt.scatter(df['height'], df['weight'], alpha=0.5)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight Relationship')
plt.show()

# 2. CORRELATION COEFFICIENT - Jačina linearne veze
correlation = df['height'].corr(df['weight'])
print(f"Correlation: {correlation:.3f}")
# -1 = perfect negative, 0 = no correlation, +1 = perfect positive

# 3. CORRELATION MATRIX - Sve numeričke kolone
corr_matrix = df.corr()
print(corr_matrix)

# Heatmap vizualizacija
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# 4. REGRESSION LINE
sns.regplot(x='height', y='weight', data=df)
plt.title('Height vs Weight with Regression Line')
plt.show()
```

**Šta tražimo:**
- **Pozitivna korelacija** - Obe rastu zajedno (height ↑, weight ↑)
- **Negativna korelacija** - Jedna raste, druga pada (price ↑, demand ↓)
- **Jačina veze** - |r| > 0.7 = jaka, 0.3-0.7 = umerena, < 0.3 = slaba
- **Linearna vs nelinearna** - Scatter plot pokazuje obrazac
- **Multicollinearity** - Visoka korelacija između features (problem za linear modele)

#### Kategorička vs Numerička:
```python
# 1. BOX PLOT - Distribucija numerical po kategorijama
plt.figure(figsize=(12, 6))
sns.boxplot(x='education_level', y='salary', data=df)
plt.title('Salary by Education Level')
plt.xticks(rotation=45)
plt.show()

# 2. VIOLIN PLOT - Kombinacija box + density
sns.violinplot(x='department', y='performance_score', data=df)
plt.title('Performance Score by Department')
plt.show()

# 3. STATISTICAL TESTS
from scipy.stats import f_oneway

# ANOVA - Da li postoji značajna razlika između grupa?
groups = [group['salary'].values for name, group in df.groupby('education_level')]
f_stat, p_value = f_oneway(*groups)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
# p < 0.05 → Postoji značajna razlika između grupa
```

**Šta tražimo:**
- **Razlike između grupa** - Da li PhD ima veće plate od Bachelor?
- **Overlap** - Da li se distribucije preklapaju?
- **Outliers po grupi** - Jedna grupa ima više outliera?

#### Kategorička vs Kategorička:
```python
# 1. CROSS TABULATION (Contingency Table)
ct = pd.crosstab(df['gender'], df['department'])
print(ct)

# Sa procentima
ct_pct = pd.crosstab(df['gender'], df['department'], normalize='all') * 100
print(ct_pct)

# 2. HEATMAP
plt.figure(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Gender vs Department Cross-tabulation')
plt.show()

# 3. STACKED BAR CHART
ct.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Department Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Department')
plt.show()

# 4. CHI-SQUARE TEST - Nezavisnost
from scipy.stats import chi2_contingency

chi2, p_value, dof, expected = chi2_contingency(ct)
print(f"Chi-square: {chi2:.3f}, p-value: {p_value:.4f}")
# p < 0.05 → Promenljive NISU nezavisne (postoji veza)
```

**Šta tražimo:**
- **Zavisnost** - Da li gender utiče na department choice?
- **Patterns** - Određene kombinacije se češće pojavljuju?

#### Time Series (Vremenske Serije):
```python
# LINE GRAPH - Trend kroz vreme
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

plt.figure(figsize=(14, 6))
df['sales'].plot()
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Multiple variables
df[['revenue', 'profit']].plot(figsize=(14, 6))
plt.title('Revenue vs Profit Over Time')
plt.show()
```

---

### 3. Multivariate Analysis (Analiza Više Promenljivih)

Identifikuje veze između **tri ili više promenljivih** istovremeno.

#### Pair Plot (Pairwise Relationships):
```python
# Scatter plot matrica svih numeričkih kolona
sns.pairplot(df[['age', 'salary', 'experience', 'performance']], 
             diag_kind='kde')
plt.suptitle('Pairplot - All Numerical Features', y=1.02)
plt.show()

# Sa color po kategoriji
sns.pairplot(df, hue='department', 
             vars=['age', 'salary', 'experience'])
plt.show()
```

**Šta tražimo:**
- **Clusters** - Prirodno grupisanje podataka
- **Relationships** - Kompleksni obrasci između više features
- **Separation** - Da li klase mogu biti razdvojene?

#### 3D Scatter Plot:
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'], df['salary'], df['experience'], 
           c=df['performance'], cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Salary')
ax.set_zlabel('Experience')
plt.title('3D Scatter: Age vs Salary vs Experience')
plt.colorbar(ax.scatter(df['age'], df['salary'], df['experience'], 
                        c=df['performance'], cmap='viridis'))
plt.show()
```

#### Principal Component Analysis (PCA) - Dimensionality Reduction:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_cols])

# PCA - redukcija na 2D za vizualizaciju
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Vizualizacija
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], 
            c=df['target'], cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA - 2D Projection')
plt.colorbar(label='Target')
plt.show()

# Explained variance
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

**Šta tražimo:**
- **Separabilnost** - Da li klase mogu biti razdvojene u lower dimensions?
- **Feature importance** - Koje features doprinose najviše variance?
- **Redundancy** - Mnogo features objašnjava malo variance?

---

## Koraci u EDA

### 1. **Problem and Data Understanding**
```python
# Šta pokušavamo da rešimo?
# - Classification? Regression? Clustering?
# - Koji je target variable?
# - Koje su business metrics?
```

### 2. **Initial Data Profiling**
```python
# Basic info
print(df.shape)           # (rows, columns)
print(df.info())          # Data types, non-null counts
print(df.head(10))        # Prvih 10 redova
print(df.tail(10))        # Poslednjih 10 redova
print(df.sample(10))      # Random 10 redova

# Statistical summary
print(df.describe())                    # Numerical columns
print(df.describe(include='object'))    # Categorical columns
```

### 3. **Univariate Analysis**
```python
# Za svaku promenljivu pojedinačno
# - Distribucija (histogram, density plot)
# - Summary statistics (mean, median, std)
# - Outlier detection (box plot, Z-score)
# - Missing values (df.isnull().sum())
```

### 4. **Bivariate Analysis**
```python
# Odnosi između parova promenljivih
# - Feature vs Target (najvažnije!)
# - Feature vs Feature (korelacije)
# - Scatter plots, correlation matrix
# - Group comparisons (box plots po kategorijama)
```

### 5. **Multivariate Analysis**
```python
# Kompleksne veze između više features
# - Pair plots
# - 3D visualizations
# - PCA za dimenzionalnost
# - Heatmaps za sve kombinacije
```

### 6. **Feature Relationships with Target**
```python
# KLJUČNO: Koje features imaju najjači uticaj na target?

# Za numerical target (regression)
for col in numerical_features:
    correlation = df[col].corr(df['target'])
    print(f"{col}: {correlation:.3f}")

# Za categorical target (classification)
for col in numerical_features:
    plt.figure()
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f'{col} by Target Class')
    plt.show()
```

### 7. **Data Quality Assessment**
```python
# Missing values pattern
import missingno as msno
msno.matrix(df)
msno.heatmap(df)

# Outliers summary
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# Duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Class imbalance (za classification)
print(df['target'].value_counts(normalize=True))
```

### 8. **Insights and Decision Making**
```python
# Na osnovu EDA, dokumentuj:
# 1. Koje features su najvažnije?
# 2. Da li treba transformacija? (log, sqrt za skewed data)
# 3. Kako tretirati outliers? (remove, cap, transform)
# 4. Kako tretirati missing values? (impute, drop)
# 5. Da li postoji class imbalance? (SMOTE, class weights)
# 6. Koji algoritmi bi mogli dobro raditi?
#    - Linear relationships → Linear models
#    - Non-linear → Trees, Neural Networks
# 7. Da li treba feature engineering?
#    - Interaction features (price × quantity)
#    - Polynomial features
#    - Date features extraction
```

---

## Ključni Alati za EDA

### Python Biblioteke:
```python
import pandas as pd                 # Data manipulation
import numpy as np                  # Numerical operations
import matplotlib.pyplot as plt     # Basic plotting
import seaborn as sns              # Statistical visualization
import plotly.express as px        # Interactive plots
import missingno as msno           # Missing data visualization

# Statistical tests
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

# Advanced
from pandas_profiling import ProfileReport  # Automated EDA report
```

### Automated EDA Tools:
```python
# 1. Pandas Profiling - Comprehensive report
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")

# 2. Sweetviz - Comparison reports
import sweetviz as sv

report = sv.analyze(df)
report.show_html("sweetviz_report.html")

# 3. AutoViz - Automatic visualizations
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
df_viz = AV.AutoViz("data.csv")
```

---

## Best Practices

### ✅ DO:
- **Počni sa jednostavnim** - Univariate → Bivariate → Multivariate
- **Visualizuj SVE** - "Jedna slika vredi 1000 reči"
- **Dokumentuj insights** - Zapiši šta si otkrio
- **Testiraj pretpostavke** - Statistical tests za validaciju
- **Poređaj features po važnosti** - Feature vs Target correlation
- **Proveri data quality** - Missing, duplicates, outliers

### ❌ DON'T:
- **Ne radi EDA nakon treninga** - EDA je **pre** modelovanja!
- **Ne ignoriši outliers** - Mogu biti greške ILI važni signali
- **Ne fokusiraj se samo na mean** - Pogledaj i median, mode, std
- **Ne zaboravi domain knowledge** - Statistika + kontekst = insights
- **Ne preskači EDA** - "Neću gubiti vreme" → gubi se vreme kasnije

---

## Šta EDA Otkriva → Šta Radiš Dalje

| EDA Insight | Akcija |
|-------------|--------|
| **Skewed distribution** | Log/sqrt transformation (Data Transformation) |
| **High correlation (>0.9)** | Remove redundant feature (Feature Selection) |
| **Missing values** | Imputation strategija (Data Cleaning) |
| **Outliers** | Cap, remove, ili transform (Data Cleaning) |
| **Class imbalance (90/10)** | SMOTE, class weights (Handling Imbalanced Data) |
| **Non-linear relationships** | Polynomial features ili tree-based models |
| **Low variance feature** | Remove feature (Feature Selection) |
| **Categorical high cardinality** | Target encoding ili binning |
| **Different scales** | Standardization/Normalization (Feature Scaling) |
| **Interaction patterns** | Create interaction features (Feature Engineering) |

---

## Primer: Kompletna EDA Skripta
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# ==================== 1. INITIAL PROFILING ====================
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# ==================== 2. UNIVARIATE ANALYSIS ====================
print("\n" + "="*50)
print("UNIVARIATE ANALYSIS")
print("="*50)

# Numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    print(f"\n{col.upper()}")
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.3f}")
    print(f"Kurtosis: {df[col].kurtosis():.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[col].dropna(), bins=30, edgecolor='black')
    axes[0].set_title(f'{col} - Histogram')
    axes[1].boxplot(df[col].dropna())
    axes[1].set_title(f'{col} - Box Plot')
    plt.tight_layout()
    plt.show()

# Categorical features
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\n{col.upper()}")
    print(df[col].value_counts())
    
    # Visualize
    df[col].value_counts().plot(kind='bar', figsize=(10, 6))
    plt.title(f'{col} - Distribution')
    plt.xticks(rotation=45)
    plt.show()

# ==================== 3. BIVARIATE ANALYSIS ====================
print("\n" + "="*50)
print("BIVARIATE ANALYSIS")
print("="*50)

# Correlation matrix
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Feature vs Target (assuming 'target' column)
if 'target' in df.columns:
    for col in numerical_cols:
        if col != 'target':
            correlation = df[col].corr(df['target'])
            print(f"{col} vs target correlation: {correlation:.3f}")
            
            plt.figure(figsize=(10, 6))
            plt.scatter(df[col], df['target'], alpha=0.5)
            plt.xlabel(col)
            plt.ylabel('target')
            plt.title(f'{col} vs Target')
            plt.show()

# ==================== 4. MULTIVARIATE ANALYSIS ====================
print("\n" + "="*50)
print("MULTIVARIATE ANALYSIS")
print("="*50)

# Pair plot
sns.pairplot(df[numerical_cols], diag_kind='kde')
plt.suptitle('Pairplot - All Features', y=1.02)
plt.show()

# ==================== 5. INSIGHTS ====================
print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)

# High correlation pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((corr_matrix.columns[i], 
                                   corr_matrix.columns[j], 
                                   corr_matrix.iloc[i, j]))

print("\nHighly Correlated Features (>0.7):")
for pair in high_corr_pairs:
    print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

# Outlier summary
print("\nOutlier Summary:")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    pct = len(outliers) / len(df) * 100
    print(f"{col}: {len(outliers)} outliers ({pct:.1f}%)")

print("\n" + "="*50)
print("EDA COMPLETE!")
print("="*50)
```

---

## Rezime

**EDA nije opcioni korak - EDA je OBAVEZAN!** 

Bez EDA:
- ❌ Ne znaš koje feature-e su bitne
- ❌ Ne znaš koje transformacije trebaš
- ❌ Ne znaš kakav model da koristiš
- ❌ Pravićeš greške u preprocessing-u

Sa EDA:
- ✅ Razumeš podatke dubinski
- ✅ Donosiš informisane odluke
- ✅ Identifikuješ probleme rano
- ✅ Poboljšavaš accuracy modela

**"Spend 80% of your time understanding the data, 20% building models."** 📊🔍