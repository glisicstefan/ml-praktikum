# ML Pipeline

ML Pipeline je **organizovana sekvenca transformacija i modelovanja** koja automatizuje ceo ML workflow od raw podataka do predikcija. Pipeline objedinjuje sve preprocessing korake (scaling, encoding, feature engineering) i model u jedan reusable, reproducible objekat.

**Zašto je Pipeline kritičan?**
- **Spreči data leakage** - Garantuje da preprocessing fit-uje samo na train data
- **Reproducibility** - Iste transformacije se primenjuju uvek isto
- **Čist kod** - Sve transformacije u jednom objektu
- **Production-ready** - Jedna linija koda za trening i deployment
- **Cross-validation friendly** - Preprocessing se dešava UNUTAR svakog fold-a
- **Easy to save/load** - Jedan objekat sadrži SVE

**VAŽNO:** Pipeline je ESENCIJALAN za pravilan ML workflow i produkciju!

---

## Problem Bez Pipeline-a

### Manual Preprocessing (LOŠE! ❌):
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 40000, 50000, 60000, 70000],
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade', 'Niš'],
    'target': [0, 1, 0, 1, 1]
})

X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Manual preprocessing - MNOGO KORAKA! ❌
# 1. Encoding
X_train_encoded = pd.get_dummies(X_train, columns=['city'])
X_test_encoded = pd.get_dummies(X_test, columns=['city'])

# Problem 1: Test može imati različite kolone!
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# 2. Scaling
scaler = StandardScaler()
numerical_cols = ['age', 'income']
X_train_encoded[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
X_test_encoded[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])

# 3. Train model
model = LogisticRegression()
model.fit(X_train_encoded, y_train)

# 4. Predict
y_pred = model.predict(X_test_encoded)

# Problem 2: Za novi data moraš ponoviti SVE korake ručno!
# Problem 3: Lako napraviti grešku (fit na test, zaboraviti korak, itd.)
# Problem 4: Ne možeš koristiti cross_val_score direktno
```

### Sa Pipeline-om (DOBRO! ✅):
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Sve u JEDAN objekat!
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['city'])
    ])),
    ('classifier', LogisticRegression())
])

# Train (sve preprocessing automatski!)
pipeline.fit(X_train, y_train)

# Predict (sve preprocessing automatski primenjeno!)
y_pred = pipeline.predict(X_test)

# Cross-validation (preprocessing INSIDE svakog fold-a!)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Save/Load (SVE u jednom fajlu!)
import joblib
joblib.dump(pipeline, 'model_pipeline.pkl')

# U production - samo load i predict!
pipeline_loaded = joblib.load('model_pipeline.pkl')
predictions = pipeline_loaded.predict(new_data)
```

**Pipeline prednosti:**
- ✅ Jedan objekat sadrži SVE
- ✅ Automatski fit na train, transform na test
- ✅ Nema data leakage
- ✅ Cross-validation radi ispravno
- ✅ Production deployment je trivijalan
- ✅ Čist, maintainable kod

---

## 1. Basic Pipeline (sklearn)

**Pipeline** je niz transformera sa finalnim estimatorom (model).

### Struktura:
```
Pipeline([
    ('step1_name', Transformer1()),
    ('step2_name', Transformer2()),
    ('step3_name', Transformer3()),
    ('final_step', Estimator())
])
```

**Pravila:**
- Svi koraci osim poslednjeg MORAJU biti **transformers** (imaju fit i transform)
- Poslednji korak može biti **estimator** (model koji ima fit i predict)
- Svaki korak ima **ime** (string) i **objekat**

### Osnovni Primer:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Step 1: Scale
    ('pca', PCA(n_components=10)),          # Step 2: Dimensionality reduction
    ('classifier', LogisticRegression())    # Step 3: Model
])

# Fit (svi koraci fit + transform + final fit)
pipeline.fit(X_train, y_train)

# Predict (svi koraci transform + final predict)
y_pred = pipeline.predict(X_test)

# Score
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

### Šta se Dešava Interno:
```python
# pipeline.fit(X_train, y_train) radi:
# 1. scaler.fit(X_train)
# 2. X_scaled = scaler.transform(X_train)
# 3. pca.fit(X_scaled)
# 4. X_pca = pca.transform(X_scaled)
# 5. classifier.fit(X_pca, y_train)

# pipeline.predict(X_test) radi:
# 1. X_scaled = scaler.transform(X_test)  # NE fit!
# 2. X_pca = pca.transform(X_scaled)      # NE fit!
# 3. y_pred = classifier.predict(X_pca)
```

### Pristup Koracima:
```python
# Pristup step-ovima
print(pipeline.named_steps)
# {'scaler': StandardScaler(), 'pca': PCA(), 'classifier': LogisticRegression()}

# Pristup specifičnom step-u
scaler = pipeline.named_steps['scaler']
print(f"Scaler mean: {scaler.mean_}")

# Pristup preko indeksa
print(pipeline.steps[0])  # ('scaler', StandardScaler())

# Pristup preko [-1] za final estimator
final_model = pipeline[-1]
print(f"Model coefficients: {final_model.coef_}")

# Set params
pipeline.set_params(classifier__C=0.5)  # Double underscore za nested params!
```

### Pipeline sa make_pipeline (Bez Imena):
```python
from sklearn.pipeline import make_pipeline

# Automatski generiše imena (lowercaseclassname)
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    LogisticRegression()
)

# Imena: standardscaler, pca, logisticregression
print(pipeline.named_steps.keys())
```

---

## 2. ColumnTransformer (Različit Preprocessing po Kolonama)

**ColumnTransformer** omogućava **različite transformacije za različite kolone** - ESENCIJALAN za mixed data!

### Struktura:
```python
ColumnTransformer([
    ('name1', Transformer1(), column_selector1),
    ('name2', Transformer2(), column_selector2),
    ...
])
```

### Column Selectors:
```python
# 1. Lista imena kolona
['age', 'income', 'experience']

# 2. Boolean mask
[True, False, True, False]

# 3. Callable (funkcija)
lambda X: [col for col in X.columns if X[col].dtype == 'float64']

# 4. Index pozicije
[0, 1, 2]

# 5. Slice
slice(0, 3)  # Kolone 0, 1, 2
```

### Osnovni Primer:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [30000, 40000, 50000, 60000],
    'city': ['Belgrade', 'Novi Sad', 'Niš', 'Belgrade'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor'],
    'target': [0, 1, 0, 1]
})

X = df.drop('target', axis=1)
y = df['target']

# Define column groups
numerical_features = ['age', 'income']
categorical_features = ['city', 'education']

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Šta sa ostalim kolonama? (passthrough = ostavi kao što jesu)
)

# Transform
X_transformed = preprocessor.fit_transform(X)

print(f"Original shape: {X.shape}")          # (4, 4)
print(f"Transformed shape: {X_transformed.shape}")  # (4, 7) - scaled age/income + one-hot city/education
```

### remainder Parametar:
```python
# remainder='drop' (default) - Briši kolone koje nisu specificirane
ct_drop = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income'])
], remainder='drop')  # 'city' i 'education' će biti izbačeni!

# remainder='passthrough' - Zadrži kolone nepromenjene
ct_pass = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income'])
], remainder='passthrough')  # 'city' i 'education' ostaju kako jesu

# remainder=transformer - Primeni transformer na sve ostale
from sklearn.preprocessing import MinMaxScaler
ct_custom = ColumnTransformer([
    ('num', StandardScaler(), ['age'])
], remainder=MinMaxScaler())  # Sve ostale se MinMax skaliraju
```

### Kompletan Pipeline sa ColumnTransformer:
```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Score
print(f"Accuracy: {pipeline.score(X_test, y_test):.3f}")

# Feature names after transformation
preprocessor = pipeline.named_steps['preprocessor']
feature_names = (
    numerical_features + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)
print(f"Feature names: {feature_names}")
```

---

## 3. imblearn Pipeline (Sa Resampling)

**sklearn Pipeline NE podržava resampling** (SMOTE, undersampling) jer resampling menja broj samples. **imblearn Pipeline** omogućava resampling u pipeline-u!

### Osnovni Primer:
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Imbalanced data
X_imb, y_imb = make_classification(
    n_samples=1000, 
    n_features=20,
    weights=[0.9, 0.1],  # 90/10 imbalance
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42)

# imblearn Pipeline sa SMOTE
pipeline_imb = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),     # Resampling step!
    ('classifier', LogisticRegression())
])

# Train (SMOTE se primenjuje SAMO na train unutar fit!)
pipeline_imb.fit(X_train, y_train)

# Predict (SMOTE se NE primenjuje na test!)
y_pred = pipeline_imb.predict(X_test)

print("Classification Report:")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### imblearn + ColumnTransformer:
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Mixed data (numerical + categorical) + imbalanced
df_imb = pd.DataFrame({
    'age': np.random.randint(20, 60, 1000),
    'income': np.random.randint(20000, 100000, 1000),
    'city': np.random.choice(['Belgrade', 'Novi Sad', 'Niš'], 1000),
    'target': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
})

X = df_imb.drop('target', axis=1)
y = df_imb['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define columns
numerical_features = ['age', 'income']
categorical_features = ['city']

# Full pipeline: Preprocessing + SMOTE + Model
pipeline_full = ImbPipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Train
pipeline_full.fit(X_train, y_train)

# Evaluate
y_pred = pipeline_full.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Cross-Validation sa imblearn Pipeline:
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Cross-validation - SMOTE se dešava UNUTAR svakog fold-a!
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline_full, X_train, y_train, cv=skf, scoring='f1')

print(f"CV F1-Score: {scores.mean():.3f} ± {scores.std():.3f}")

# Ovo je PRAVILAN način - SMOTE se fit-uje SAMO na train fold!
# Bez pipeline-a ovo bi bilo VEOMA komplikovano!
```

---

## 4. Custom Transformers

**Kreiranje sopstvenih transformera** za specifične preprocessing korake.

### BaseEstimator i TransformerMixin:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Template za custom transformer.
    """
    def __init__(self, param1=None):
        self.param1 = param1
    
    def fit(self, X, y=None):
        # Nauči parametre iz X (ako treba)
        # y je optional (za supervised transformacije)
        return self
    
    def transform(self, X):
        # Primeni transformaciju na X
        X_transformed = X.copy()
        # ... transformacija ...
        return X_transformed
    
    def fit_transform(self, X, y=None):
        # Ovo se automatski nasledjuje od TransformerMixin
        # Ne moraš definisati ako koristiš fit + transform
        return self.fit(X, y).transform(X)
```

### Primer 1: Log Transformer:
```python
class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Primenjuje log1p transformaciju na specifične kolone.
    """
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        # Nema parametara za učiti, samo vrati self
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.columns is None:
            # Ako nisu specificirane kolone, transformiši sve numerical
            cols_to_transform = X_copy.select_dtypes(include=[np.number]).columns
        else:
            cols_to_transform = self.columns
        
        for col in cols_to_transform:
            X_copy[col] = np.log1p(X_copy[col])
        
        return X_copy

# Korišćenje u pipeline
pipeline_log = Pipeline([
    ('log_transform', LogTransformer(columns=['income', 'price'])),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```

### Primer 2: Feature Creator:
```python
class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Kreira nove features iz postojećih.
    """
    def __init__(self, create_interactions=True):
        self.create_interactions = create_interactions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.create_interactions and 'age' in X_copy.columns and 'income' in X_copy.columns:
            # Kreiranje interaction features
            X_copy['age_x_income'] = X_copy['age'] * X_copy['income']
            X_copy['income_per_age'] = X_copy['income'] / (X_copy['age'] + 1)
        
        return X_copy

# Pipeline sa feature creation
pipeline_features = Pipeline([
    ('feature_creator', FeatureCreator(create_interactions=True)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])
```

### Primer 3: Outlier Clipper:
```python
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clipuje outliere na percentile granice.
    """
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
    
    def fit(self, X, y=None):
        # Nauči granice iz train data
        for col in X.select_dtypes(include=[np.number]).columns:
            self.lower_bounds_[col] = np.percentile(X[col], self.lower_percentile)
            self.upper_bounds_[col] = np.percentile(X[col], self.upper_percentile)
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in self.lower_bounds_.keys():
            X_copy[col] = X_copy[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )
        
        return X_copy

# Pipeline sa outlier clipping
pipeline_clip = Pipeline([
    ('outlier_clipper', OutlierClipper(lower_percentile=1, upper_percentile=99)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

### Primer 4: Date Feature Extractor:
```python
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Ekstraktuje date features (month, day_of_week, itd).
    """
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.date_column in X_copy.columns:
            # Convert to datetime ako već nije
            X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column])
            
            # Extract features
            X_copy['year'] = X_copy[self.date_column].dt.year
            X_copy['month'] = X_copy[self.date_column].dt.month
            X_copy['day'] = X_copy[self.date_column].dt.day
            X_copy['day_of_week'] = X_copy[self.date_column].dt.dayofweek
            X_copy['is_weekend'] = X_copy['day_of_week'].isin([5, 6]).astype(int)
            
            # Drop original date column
            X_copy = X_copy.drop(columns=[self.date_column])
        
        return X_copy

# Pipeline sa date extraction
pipeline_date = Pipeline([
    ('date_extractor', DateFeatureExtractor(date_column='transaction_date')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])
```

### Primer 5: Missing Indicator + Imputer:
```python
class MissingIndicatorImputer(BaseEstimator, TransformerMixin):
    """
    Kreira missing indicators PA ONDA impute-uje.
    """
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.fill_values_ = {}
    
    def fit(self, X, y=None):
        # Nauči fill values
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.strategy == 'mean':
                self.fill_values_[col] = X[col].mean()
            elif self.strategy == 'median':
                self.fill_values_[col] = X[col].median()
            elif self.strategy == 'mode':
                self.fill_values_[col] = X[col].mode()[0]
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Kreiranje missing indicators
        for col in self.fill_values_.keys():
            X_copy[f'{col}_was_missing'] = X_copy[col].isnull().astype(int)
        
        # Impute
        for col, fill_value in self.fill_values_.items():
            X_copy[col].fillna(fill_value, inplace=True)
        
        return X_copy

# Pipeline
pipeline_missing = Pipeline([
    ('missing_handler', MissingIndicatorImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

---

## 5. Feature Union (Paralelne Transformacije)

**FeatureUnion** kombinuje OUTPUT više transformera - korisno za kreiranje različitih feature tipova.

### Kako Radi:
```
Input X
    ├─→ Transformer1 → Features A
    ├─→ Transformer2 → Features B
    └─→ Transformer3 → Features C

Output: [Features A | Features B | Features C] (horizontally concatenated)
```

### Primer:
```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# FeatureUnion - kombinuj PCA i SelectKBest features
feature_union = FeatureUnion([
    ('pca', PCA(n_components=5)),           # 5 PCA components
    ('kbest', SelectKBest(f_classif, k=10)) # 10 best statistical features
])

# Pipeline sa FeatureUnion
pipeline_union = Pipeline([
    ('scaler', StandardScaler()),
    ('features', feature_union),  # Kombinuj 5 PCA + 10 best = 15 features
    ('classifier', LogisticRegression())
])

pipeline_union.fit(X_train, y_train)
print(f"Accuracy: {pipeline_union.score(X_test, y_test):.3f}")
```

### Kompleksniji Primer - Text + Numerical Features:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset sa text i numerical features
df_mixed = pd.DataFrame({
    'text': ['Great product!', 'Terrible quality', 'Love it', 'Not good'],
    'price': [100, 50, 150, 75],
    'rating': [5, 1, 5, 2],
    'target': [1, 0, 1, 0]
})

# Selector za text kolonu
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column]

# Selector za numerical kolone
class NumericalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

# FeatureUnion - kombinuj text i numerical features
feature_union_mixed = FeatureUnion([
    ('text_features', Pipeline([
        ('selector', TextSelector(column='text')),
        ('tfidf', TfidfVectorizer(max_features=50))
    ])),
    ('numerical_features', Pipeline([
        ('selector', NumericalSelector(columns=['price', 'rating'])),
        ('scaler', StandardScaler())
    ]))
])

# Full pipeline
pipeline_text_num = Pipeline([
    ('features', feature_union_mixed),
    ('classifier', LogisticRegression())
])

X = df_mixed.drop('target', axis=1)
y = df_mixed['target']

pipeline_text_num.fit(X, y)
```

---

## 6. Nested Pipelines

**Pipeline unutar Pipeline-a** - modularna struktura za kompleksne preprocessing-e.

### Primer - Preprocessing Pipeline + Model Pipeline:
```python
# Sub-pipeline za numerical features
numerical_pipeline = Pipeline([
    ('outlier_clipper', OutlierClipper(lower_percentile=1, upper_percentile=99)),
    ('log_transform', LogTransformer(columns=['income'])),
    ('scaler', StandardScaler())
])

# Sub-pipeline za categorical features
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Main preprocessor sa nested pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Final pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
final_pipeline.fit(X_train, y_train)

# Predict
y_pred = final_pipeline.predict(X_test)
```

### Kompletan Enterprise Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Define column types
numerical_features = ['age', 'income', 'experience']
categorical_nominal = ['city', 'department']
categorical_ordinal = ['education']  # Has order: Elementary < High School < Bachelor < Master < PhD

# Education order
education_order = [['Elementary', 'High School', 'Bachelor', 'Master', 'PhD']]

# Numerical pipeline - Handle missing, scale, select
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Nominal categorical pipeline
nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Ordinal categorical pipeline
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ordinal', OrdinalEncoder(categories=education_order, handle_unknown='use_encoded_value', unknown_value=-1))
])

# Full preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat_nom', nominal_pipeline, categorical_nominal),
    ('cat_ord', ordinal_pipeline, categorical_ordinal)
], remainder='drop')

# Full enterprise pipeline
enterprise_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    ))
])

# Train
enterprise_pipeline.fit(X_train, y_train)

# Evaluate
print(f"Train Accuracy: {enterprise_pipeline.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {enterprise_pipeline.score(X_test, y_test):.3f}")
```

---

## 7. Hyperparameter Tuning sa Pipeline

**GridSearchCV i RandomizedSearchCV** rade sa pipeline-ovima!

### Parametar Sintaksa:
```python
# Parametri se referišu sa: 'step_name__parameter_name'

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Set params
pipeline.set_params(classifier__C=0.5)  # Double underscore!
```

### GridSearchCV Primer:
```python
from sklearn.model_selection import GridSearchCV

# Pipeline
pipeline_grid = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid - koristit double underscore!
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()]  # Test različite scalers!
}

# GridSearchCV
grid_search = GridSearchCV(
    pipeline_grid,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Search
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Best model
best_pipeline = grid_search.best_estimator_

# Test
print(f"Test score: {best_pipeline.score(X_test, y_test):.3f}")
```

### RandomizedSearchCV Primer:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Parameter distributions
param_distributions = {
    'classifier__n_estimators': randint(50, 300),
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__max_features': ['sqrt', 'log2', None],
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline_grid,
    param_distributions,
    n_iter=50,  # 50 random combinations
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
```

---

## 8. Saving and Loading Pipelines

**Serialization** - save pipeline za production deployment.

### joblib (Preporučeno):
```python
import joblib

# Save pipeline
joblib.dump(pipeline, 'model_pipeline.pkl')

# Load pipeline
pipeline_loaded = joblib.load('model_pipeline.pkl')

# Use loaded pipeline
predictions = pipeline_loaded.predict(new_data)
```

### pickle:
```python
import pickle

# Save
with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Load
with open('model_pipeline.pkl', 'rb') as f:
    pipeline_loaded = pickle.load(f)
```

### Compression (Za Velike Modele):
```python
# Save sa compression
joblib.dump(pipeline, 'model_pipeline.pkl.gz', compress=3)  # compression level 0-9

# Load compressed
pipeline_loaded = joblib.load('model_pipeline.pkl.gz')
```

### Versioning:
```python
import joblib
from datetime import datetime

# Save sa timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'model_pipeline_{timestamp}.pkl'
joblib.dump(pipeline, filename)

print(f"Saved: {filename}")

# Load najnoviji
import glob
import os

# Find najnoviji fajl
list_of_files = glob.glob('model_pipeline_*.pkl')
latest_file = max(list_of_files, key=os.path.getctime)

pipeline_loaded = joblib.load(latest_file)
print(f"Loaded: {latest_file}")
```

### Production DeploymentРимер:
```python
# train.py - Training script
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, 'production_model.pkl')
print("Model saved for production!")

# -----------------------------------------------------------

# predict.py - Production prediction script
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('production_model.pkl')

# New data (from API, database, etc.)
new_data = pd.DataFrame({
    'age': [35],
    'income': [50000],
    'city': ['Belgrade']
})

# Predict (sve preprocessing automatski!)
prediction = pipeline.predict(new_data)
probability = pipeline.predict_proba(new_data)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

---

## 9. Best Practices

### ✅ DO:

**1. Uvek Koristi Pipeline za Production**
```python
# ✅ DOBRO - Pipeline je production-ready
pipeline = Pipeline([...])
joblib.dump(pipeline, 'model.pkl')

# ❌ LOŠE - Manual preprocessing u production je error-prone
# preprocessing.py + model.pkl = Može biti inconsistent!
```

**2. Imenuj Korake Jasno**
```python
# ✅ DOBRO - Jasna imena
Pipeline([
    ('remove_outliers', OutlierClipper()),
    ('handle_missing', MissingIndicatorImputer()),
    ('scale_features', StandardScaler()),
    ('select_features', SelectKBest(k=10)),
    ('train_model', RandomForestClassifier())
])

# ❌ LOŠE - Nejasna imena
Pipeline([
    ('step1', OutlierClipper()),
    ('step2', MissingIndicatorImputer()),
    ('step3', StandardScaler())
])
```

**3. Koristi ColumnTransformer za Mixed Data**
```python
# ✅ DOBRO
ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# ❌ LOŠE - Manual kolona handling
```

**4. Validate Pipeline sa Cross-Validation**
```python
# ✅ DOBRO - CV testira ceo pipeline
scores = cross_val_score(pipeline, X, y, cv=5)

# Preprocessing se dešava UNUTAR svakog fold-a!
```

**5. Version Control Pipelines**
```python
# ✅ DOBRO - Trackuj verzije
pipeline_v1_0 = Pipeline([...])
joblib.dump(pipeline_v1_0, 'model_v1.0.pkl')

# ❌ LOŠE - Prepišeš stari model bez verzije
joblib.dump(pipeline, 'model.pkl')  # Gubi staru verziju!
```

**6. Dokumentuj Pipeline Strukturu**
```python
# ✅ DOBRO
"""
Pipeline Structure:
1. Preprocessor (ColumnTransformer)
   - Numerical: OutlierClipper → StandardScaler
   - Categorical: OneHotEncoder (drop='first')
2. Feature Selection: SelectKBest (k=15)
3. Classifier: RandomForestClassifier (n_estimators=100)

Input: Raw DataFrame sa ['age', 'income', 'city']
Output: Binary predictions (0/1)
"""
```

### ❌ DON'T:

**1. Ne Fit Pipeline na Test Data**
```python
# ❌ LOŠE
pipeline.fit(X_test, y_test)  # NIKAD!

# ✅ DOBRO
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**2. Ne Mešaj sklearn i imblearn Pipeline Nepravilno**
```python
# ❌ LOŠE - sklearn Pipeline sa resampling
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipeline = Pipeline([  # sklearn Pipeline!
    ('smote', SMOTE()),  # ERROR - resampling u sklearn Pipeline!
    ('model', LogisticRegression())
])

# ✅ DOBRO - imblearn Pipeline za resampling
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('model', LogisticRegression())
])
```

**3. Ne Zaboravi handle_unknown za OneHotEncoder**
```python
# ❌ LOŠE - Novi test data može imati unknown categories!
OneHotEncoder(drop='first')

# ✅ DOBRO
OneHotEncoder(drop='first', handle_unknown='ignore')
```

**4. Ne Hardcode Column Names**
```python
# ❌ LOŠE
ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income'])  # Hardcoded!
])

# ✅ DOBRO - Define outside
numerical_features = ['age', 'income']
categorical_features = ['city']

ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])
```

**5. Ne Ignoriši Preprocessing u Test Metrics**
```python
# ❌ LOŠE - Manual transform test
X_test_transformed = scaler.transform(X_test)  # Zaboravljaš neki korak?
y_pred = model.predict(X_test_transformed)

# ✅ DOBRO - Pipeline radi SVE
y_pred = pipeline.predict(X_test)  # Garantovano svi koraci!
```

---

## 10. Complete End-to-End Example

### Real-World Scenario: Customer Churn Prediction
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ==================== 1. LOAD DATA ====================
df = pd.DataFrame({
    'customer_id': range(1000),
    'age': np.random.randint(18, 70, 1000),
    'tenure_months': np.random.randint(1, 120, 1000),
    'monthly_charges': np.random.uniform(20, 150, 1000),
    'total_charges': np.random.uniform(100, 10000, 1000),
    'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], 1000),
    'payment_method': np.random.choice(['Electronic', 'Mailed Check', 'Bank Transfer', 'Credit Card'], 1000),
    'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], 1000),
    'customer_service_calls': np.random.randint(0, 10, 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # 80/20 imbalance
})

# Introduce some missing values
df.loc[df.sample(frac=0.05).index, 'tenure_months'] = np.nan
df.loc[df.sample(frac=0.03).index, 'total_charges'] = np.nan

print("Dataset Info:")
print(df.info())
print(f"\nChurn distribution:\n{df['churn'].value_counts(normalize=True)}")

# ==================== 2. DEFINE FEATURES ====================
# Drop customer_id (not a feature)
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']

# Define column types
numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'customer_service_calls']
categorical_nominal = ['payment_method', 'internet_service']
categorical_ordinal = ['contract_type']

# Ordinal order
contract_order = [['Month-to-Month', 'One Year', 'Two Year']]

print(f"\nFeatures:")
print(f"Numerical: {numerical_features}")
print(f"Categorical (Nominal): {categorical_nominal}")
print(f"Categorical (Ordinal): {categorical_ordinal}")

# ==================== 3. TRAIN-TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# ==================== 4. BUILD PIPELINE ====================

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Nominal categorical pipeline
nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Ordinal categorical pipeline
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Month-to-Month')),
    ('ordinal', OrdinalEncoder(categories=contract_order, handle_unknown='use_encoded_value', unknown_value=-1))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat_nom', nominal_pipeline, categorical_nominal),
    ('cat_ord', ordinal_pipeline, categorical_ordinal)
], remainder='drop')

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

print("\nPipeline Structure:")
print(pipeline)

# ==================== 5. HYPERPARAMETER TUNING ====================

param_grid = {
    'feature_selection__k': [8, 10, 12],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("\nStarting GridSearchCV...")
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.3f}")

# Best pipeline
best_pipeline = grid_search.best_estimator_

# ==================== 6. EVALUATE ====================

# Train predictions
y_train_pred = best_pipeline.predict(X_train)
y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]

# Test predictions
y_test_pred = best_pipeline.predict(X_test)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

print("\nTrain Metrics:")
print(f"F1-Score: {f1_score(y_train, y_train_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_train, y_train_proba):.3f}")

print("\nTest Metrics:")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.3f}")

print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

# ==================== 7. SAVE PIPELINE ====================

# Save best pipeline
model_filename = 'churn_prediction_pipeline.pkl'
joblib.dump(best_pipeline, model_filename)
print(f"\nPipeline saved: {model_filename}")

# Save metadata
metadata = {
    'model_name': 'Customer Churn Prediction',
    'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'best_params': grid_search.best_params_,
    'test_f1': f1_score(y_test, y_test_pred),
    'test_roc_auc': roc_auc_score(y_test, y_test_proba),
    'feature_columns': X.columns.tolist()
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Metadata saved: model_metadata.json")

# ==================== 8. LOAD AND USE (Production Simulation) ====================

# Load pipeline
loaded_pipeline = joblib.load(model_filename)

# New customer data
new_customer = pd.DataFrame({
    'age': [35],
    'tenure_months': [24],
    'monthly_charges': [75.5],
    'total_charges': [1812.0],
    'contract_type': ['One Year'],
    'payment_method': ['Electronic'],
    'internet_service': ['Fiber Optic'],
    'customer_service_calls': [3]
})

# Predict
churn_prediction = loaded_pipeline.predict(new_customer)[0]
churn_probability = loaded_pipeline.predict_proba(new_customer)[0, 1]

print("\n" + "="*60)
print("PRODUCTION PREDICTION EXAMPLE")
print("="*60)
print(f"New Customer: {new_customer.to_dict('records')[0]}")
print(f"Churn Prediction: {'YES' if churn_prediction == 1 else 'NO'}")
print(f"Churn Probability: {churn_probability:.1%}")
```

---

## Rezime - ML Pipeline

### Zašto Pipeline?

| Problem Bez Pipeline | Rešenje Sa Pipeline |
|----------------------|---------------------|
| Manual preprocessing za svaki novi data | Automatski preprocessing |
| Data leakage (fit na test) | Garantovano fit samo na train |
| Inconsistent preprocessing u production | Konzistentno sve vreme |
| Teško za cross-validation | CV radi out-of-the-box |
| Više fajlova za save (scaler.pkl, model.pkl) | Jedan fajl za sve |
| Bug-prone kod | Testiran, robustan kod |

### Pipeline Components:

| Component | Kada Koristiti |
|-----------|---------------|
| **Pipeline** | Sekvenca transformera + model |
| **ColumnTransformer** | Različit preprocessing za različite kolone |
| **imblearn Pipeline** | Kada treba resampling (SMOTE) |
| **FeatureUnion** | Kombinovanje različitih feature extraction metoda |
| **Custom Transformer** | Specifični preprocessing koji nema u sklearn |

### Best Practices Checklist:
```
✅ Uvek koristi Pipeline za production
✅ ColumnTransformer za mixed data (numerical + categorical)
✅ handle_unknown='ignore' u OneHotEncoder
✅ imblearn.Pipeline ako treba resampling
✅ Imenuj korake jasno i deskriptivno
✅ Cross-validation sa celim pipeline-om
✅ GridSearchCV sa pipeline parametrima
✅ Save pipeline + metadata
✅ Version control za pipelines
✅ Dokumentuj strukturu pipeline-a
```

**Key Takeaway:** Pipeline je **ne-negocijabilni** deo production ML! Sve što si naučio o preprocessing-u (cleaning, transformation, encoding, scaling, feature engineering) sada ide u Pipeline. Jedan objekat, jedan save, jedna load - to je put ka robusnim, reproducible ML sistemima! 🎯