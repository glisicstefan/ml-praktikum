# Machine Learning - Proces Rada

Proces rada u mašinskom učenju je iterativan i sistematičan pristup koji se sastoji od nekoliko ključnih koraka. Svaki korak je kritičan za uspeh finalnog modela.

## 1. Problem Definition (Definisanje Problema)

Prva i najvažnija faza - jasno definisanje problema koji želimo da rešimo.

**Ključna pitanja:**
- Koji je poslovni cilj? Šta želimo da postignemo?
- Da li je ovo **classification**, **regression**, **clustering** ili neki drugi tip problema?
- Koji su **success metrics**? Kako ćemo meriti uspeh?
- Koji su constraints (ograničenja)? - Vreme, resursi, accuracy requirements
- Šta su **input features** i šta je **target variable**?

**Primeri:**
- "Predvideti da li će kupac otkazati pretplatu" → Binary Classification
- "Proceniti cenu nekretnine" → Regression
- "Segmentirati korisnike u grupe" → Clustering
- "Preporučiti proizvode korisniku" → Recommendation System

**Best Practices:**
- Razgovaraj sa stakeholder-ima i domain ekspertima
- Definiši jasne i merljive ciljeve
- Proceni da li je ML pravi pristup (ponekad jednostavna rule-based logika radi bolje)

## 2. Data Collection (Prikupljanje Podataka)

Prikupljanje kvalitetnih i relevantnih podataka je osnova svakog ML projekta.

**Izvori podataka:**
- **Databases** - SQL/NoSQL baze podataka kompanije
- **APIs** - Eksterni servisi (Twitter API, Google Maps API, finansijski podaci)
- **Web Scraping** - Automatsko prikupljanje sa web sajtova (BeautifulSoup, Scrapy)
- **Files** - CSV, Excel, JSON, XML fajlovi
- **IoT Sensors** - Real-time podaci sa senzora
- **Public Datasets** - Kaggle, UCI ML Repository, Government Open Data
- **Surveys** - Upitnici i ankete
- **Manual Labeling** - Ručno označavanje podataka (za supervised learning)

**Važna razmatranja:**
- **Količina podataka** - "Više je skoro uvek bolje", ali zavisi od problema
- **Kvalitet > Kvantitet** - 1000 kvalitetnih primera bolje od 10,000 šumnih
- **Reprezentativnost** - Podaci moraju pokrivati sve moguće scenarije
- **Privacy & Legal** - GDPR, consent, anonimizacija
- **Cost** - Prikupljanje i labeling mogu biti skupi

**Minimum podaci po problemu:**
- Simple models: 1,000 - 10,000+ primera
- Deep learning: 100,000+ primera
- Computer vision: 1,000+ slika po klasi

## 3. Data Cleaning and Preprocessing

Sirovi podaci su retko spremni za direktnu upotrebu - potrebno ih je očistiti i transformisati.

### Data Cleaning (Čišćenje Podataka)

Identifikacija i ispravljanje problema u podacima.

#### Missing Values (Nedostajuće Vrednosti)
**Strategije:**
- **Deletion** - Brisanje redova/kolona (ako je mali procenat missing)
  - `df.dropna()` - briše redove sa missing values
  - `df.dropna(axis=1)` - briše kolone
- **Imputation** - Popunjavanje vrednosti:
  - Mean/Median/Mode - Za numeričke vrednosti
  - Forward/Backward Fill - Za time series
  - KNN Imputation - Koristi slične redove
  - Model-based - Predviđanje missing values
  - Konstanta - npr. "Unknown" za kategoričke

**Kod primer:**
```python
# Mean imputation
df['age'].fillna(df['age'].mean(), inplace=True)

# Mode za kategoričke
df['category'].fillna(df['category'].mode()[0], inplace=True)
```

#### Outliers (Odstupajuće Vrednosti)
**Detekcija:**
- **IQR Method** - Vrednosti izvan Q1 - 1.5×IQR i Q3 + 1.5×IQR
- **Z-score** - Vrednosti sa |z| > 3
- **Isolation Forest** - ML pristup
- **Domain knowledge** - "Starost 200 godina je nemoguća"

**Tretman:**
- **Remove** - Obriši outliers (ako su greška)
- **Cap** - Ograniči na određeni percentile (95th, 99th)
- **Transform** - Log transform za smanjenje uticaja
- **Keep** - Ako su validni i važni

#### Inconsistencies (Nekonzistentnosti)
- **Duplicates** - Uklanjanje duplikata (`df.drop_duplicates()`)
- **Format issues** - "01/02/2024" vs "2024-02-01"
- **Typos** - "New York" vs "new york" vs "NY"
- **Invalid values** - Negativne starosti, buduće datume rođenja
- **Data types** - Konverzija u odgovarajuće tipove

### Data Preprocessing (Priprema Podataka)

Transformacija podataka u format pogodan za ML algoritme.

#### Standardization and Normalization (Skaliranje)
Većina ML algoritama bolje radi kada su feature-i na sličnoj skali.

**Min-Max Scaling (Normalization)**
- Skalira vrednosti na opseg [0, 1]
- Formula: X' = (X - Xmin) / (Xmax - Xmin)
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

**Standardization (Z-score normalization)**
- Centira podatke oko 0 sa std=1
- Formula: X' = (X - μ) / σ
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

**Kada koristiti:**
- **Min-Max**: Neural networks, image processing, bounded features
- **Standardization**: Linear models, SVMs, PCA, features sa outliers

#### Encoding Categorical Variables

ML algoritmi rade samo sa numeričkim vrednostima.

**Label Encoding**
- Pretvara kategorije u brojeve: ["Red", "Green", "Blue"] → [0, 1, 2]
- **Koristi za**: Ordinal data (Low, Medium, High) ili tree-based modeli
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
```

**One-Hot Encoding**
- Kreira binary kolone za svaku kategoriju
- "Red" → [1, 0, 0], "Green" → [0, 1, 0], "Blue" → [0, 0, 1]
- **Koristi za**: Nominal data (boje, gradovi) i linear modele
```python
df = pd.get_dummies(df, columns=['color'], drop_first=True)
```

**Target Encoding**
- Zamenjuje kategorije sa prosečnom vrednošću target-a
- **Oprez**: Može dovesti do data leakage! Uvek na train skupu!
```python
target_mean = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_mean)
```

**Frequency Encoding**
- Zamenjuje sa frekvencijom pojavljivanja kategorije

#### Text Data Processing
- **Tokenization** - Deljenje teksta na reči
- **Lowercasing** - Pretvaranje u mala slova
- **Removing stopwords** - Uklanjanje "a", "the", "is"
- **Stemming/Lemmatization** - "running" → "run"
- **Vectorization** - TF-IDF, Word2Vec, BERT embeddings

#### Date/Time Features
- **Extract components**: year, month, day, hour, day_of_week
- **Cyclical encoding**: sin/cos transform za časove/mesece
- **Time differences**: "days since event"

## 4. Exploratory Data Analysis (EDA)

Koristimo statističke i vizuelne alate da istražimo paterne, trendove i odnose u podacima.

### Deskriptivna Statistika
```python
df.describe()  # Mean, std, min, max, quartiles
df.info()      # Data types, non-null counts
df.corr()      # Correlation matrix
```

### Vizualizacija

**Univariate Analysis (Jedna promenljiva):**
- **Histogram** - Distribucija numeričkih vrednosti
- **Box plot** - Identifikacija outliers, quartiles
- **Bar chart** - Frekvencija kategoričkih vrednosti
- **Density plot** - Smoothed distribucija

**Bivariate Analysis (Dve promenljive):**
- **Scatter plot** - Relacija između dve numeričke promenljive
- **Correlation heatmap** - Korelacija između svih feature-a
- **Box plot by category** - Distribucija numeričke po kategoriji
- **Count plot** - Kategoričke vs kategoričke

**Multivariate Analysis:**
- **Pair plot** - Odnosi između svih parova feature-a
- **3D scatter** - Tri dimenzije odjednom
- **Parallel coordinates** - Više dimenzija

### Insights iz EDA

**Šta tražimo:**
- **Distribucija** - Da li je normalna, skewed, bimodal?
- **Outliers** - Ekstremne vrednosti koje mogu uticati na model
- **Patterns** - Trendi, sezonalnost, cyclical behavior
- **Correlations** - Koje feature-i su povezani sa target-om?
- **Class imbalance** - Da li neke klase dominiraju?
- **Missing patterns** - Da li missing values prate neki obrazac?

**Donosimo odluke:**
- Koje feature-e zadržati/izbaciti
- Koje transformacije primeniti
- Koji algoritmi bi mogli raditi dobro
- Da li nam treba više podataka
- Kako će tretirati outliers i missing values

## 5. Feature Engineering

Kreiranje, transformacija i selekcija feature-a da bi poboljšali performanse modela.

### Feature Creation (Kreiranje Novih Feature-a)

**Interakcije:**
```python
df['price_per_sqft'] = df['price'] / df['square_feet']
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['total_spending'] = df['online_spending'] + df['offline_spending']
```

**Polynomial Features:**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# Kreira x1², x2², x1×x2, itd.
```

**Binning (Bucketing):**
```python
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                          labels=['Child', 'Young', 'Adult', 'Senior'])
```

**Aggregations:**
- Za transakcijske podatke: sum, mean, count, std po korisniku/vremenu
```python
user_stats = transactions.groupby('user_id').agg({
    'amount': ['sum', 'mean', 'count'],
    'date': lambda x: (x.max() - x.min()).days  # recency
})
```

**Domain-Specific:**
- Finance: Moving averages, RSI, volatility
- E-commerce: Days since last purchase, lifetime value
- Healthcare: Symptom combinations, risk scores

### Feature Selection (Selekcija Feature-a)

Biramo najvažnije feature-e da smanjimo kompleksnost i poboljšamo generalizaciju.

**Filter Methods (Pre treninga):**
- **Correlation** - Uklanjanje visoko korelisanih feature-a
- **Variance Threshold** - Uklanjanje feature-a sa malom varijansom
- **Statistical Tests** - Chi-square, ANOVA, mutual information

**Wrapper Methods (Koriste model):**
- **Forward Selection** - Postepeno dodavanje najboljih feature-a
- **Backward Elimination** - Postepeno uklanjanje najgorih
- **Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE
selector = RFE(estimator=model, n_features_to_select=10)
X_selected = selector.fit_transform(X, y)
```

**Embedded Methods (Unutar modela):**
- **L1 Regularization (Lasso)** - Automatski smanjuje koeficijente na 0
- **Tree-based Feature Importance** - Random Forest, XGBoost
```python
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

### Domain Expertise

Korišćenje opšteg znanja o domenu za bolje feature-e:
- **Medical**: Kombinovanje simptoma za risk score
- **Finance**: Ratio analysis (P/E ratio, debt-to-equity)
- **Marketing**: Customer lifetime value, churn probability
- **Real Estate**: Location scores, school district quality

### Optimization (Balansiranje)

**Zašto ne koristiti sve feature-e?**
- **Curse of dimensionality** - Više dimenzija = više potrebnih podataka
- **Overfitting** - Model pamti šum umesto obrazaca
- **Computational cost** - Sporiji trening i inference
- **Interpretability** - Teže je razumeti model

**Pravilo:** Koristi najmanje feature-a koji daju najbolje performanse.

## 6. Model Selection (Izbor Modela)

Izbor odgovarajućeg algoritma na osnovu problema, podataka i zahteva.

**Faktori koji utiču:**
- **Tip problema** - Classification, regression, clustering
- **Veličina dataseta** - Neki modeli zahtevaju više podataka
- **Feature karakteristike** - Linearne vs nelinearne relacije
- **Interpretability** - Da li moramo razumeti odluke?
- **Performance requirements** - Speed vs accuracy trade-off
- **Resources** - Computational power, memory

**Poređenje popularnih modela:**

| Model | Pros | Cons | Kada koristiti |
|-------|------|------|----------------|
| Linear/Logistic Regression | Brz, interpretativan, jednostavan | Samo linearne relacije | Baseline, interpretability važna |
| Decision Trees | Interpretativan, radi sa mixed data | Overfitting, nestabilan | Kada treba razumeti odluke |
| Random Forest | Robustan, feature importance, manje overfitting | Sporiji, teže za deploy | Generalno dobar izbor |
| XGBoost/LightGBM | State-of-the-art accuracy | Teže za tune, spor na velikom broju feature-a | Kaggle, production |
| SVM | Efikasan u high-dimensional | Spor na velikim datasetima | Text classification, image recognition |
| Neural Networks | Modeluje kompleksne relacije | Potrebno mnogo podataka, "black box" | Images, text, veliki dataseti |
| KNN | Jednostavan, no training | Spor prediction, osetljiv na scale | Mala data, baseline |

**Strategija:**
1. Počni sa **baseline** modelom (logistic regression, decision tree)
2. Testiraj nekoliko različitih pristupa
3. Koristi **ensemble methods** za poboljšanje
4. Optimizuj najbolje performujući model

## 7. Model Training (Treniranje Modela)

Faza gde model uči obrasce iz training podataka.

### Train-Test Split

**Hold-out metoda:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- **Train**: 70-80% - Za učenje
- **Test**: 20-30% - Za finalnu evaluaciju
- **Stratify**: Održava proporciju klasa

**Train-Validation-Test Split:**
```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```
- **Train**: 60-70% - Učenje
- **Validation**: 15-20% - Hyperparameter tuning
- **Test**: 15-20% - Finalna evaluacija

### Cross-Validation

Robusniji pristup evaluaciji modela.

**K-Fold Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```
- Deli data na K delova (foldova)
- K iteracija: svaki fold jednom test, ostali train
- Prosek rezultata daje bolju procenu

**Stratified K-Fold:**
- Održava proporciju klasa u svakom foldu
- Najbolji za imbalanced datasets

**Time Series Split:**
- Za vremenske serije (ne meša redosled)
- Train na prošlosti, test na budućnosti

### Fitting the Model
```python
# Sklearn standardni workflow
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Best Practices:**
- Uvek **fit na train**, **predict na test**
- Nikada ne koristiti test podatke tokom treninga
- Sacuvaj **random_state** za reproducibility
- Prati training time i resource usage

## 8. Model Evaluation and Tuning

Procena performansi modela i optimizacija hyperparameters.

### Evaluation Metrics

**Classification Metrics:**

**Confusion Matrix:**
```
                Predicted
                 0    1
Actual   0     TN   FP
         1     FN   TP
```

**Accuracy** = (TP + TN) / Total
- Ukupan procenat tačnih predikcija
- **Kada NE koristiti**: Imbalanced datasets (99% class 0, model uvek predviđa 0 = 99% accuracy!)

**Precision** = TP / (TP + FP)
- Od predviđenih pozitivnih, koliko je stvarno pozitivno
- **Kada koristiti**: Kada su False Positives skupi (spam detection - ne želimo da oznčimo važan email kao spam)

**Recall (Sensitivity)** = TP / (TP + FN)
- Od stvarno pozitivnih, koliko smo detektovali
- **Kada koristiti**: Kada su False Negatives skupi (cancer detection - ne želimo propustiti bolest)

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonijska sredina Precision i Recall
- **Kada koristiti**: Balans između precision i recall, imbalanced data

**ROC-AUC**
- Area Under ROC Curve (TPR vs FPR)
- **0.5** = Random guessing, **1.0** = Perfect
- **Kada koristiti**: Binary classification, uporediti modele

**Regression Metrics:**

**MAE (Mean Absolute Error)** = Average |predicted - actual|
- Prosečna apsolutna greška
- Jednostavna interpretacija
- **Kada koristiti**: Outliers ne smeju previše uticati

**MSE (Mean Squared Error)** = Average (predicted - actual)²
- Penalizuje veće greške više
- **Kada koristiti**: Velike greške su neprihvatljive

**RMSE (Root Mean Squared Error)** = √MSE
- Iste jedinice kao target variable
- **Najkoriščeniji** regression metric

**R² Score (Coefficient of Determination)**
- Koliko dobro model objašnjava varijansu (0-1)
- **1.0** = Perfect fit, **0** = Model ne objašnjava ništa
- Može biti negativan (model gori od proseka)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Classification
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")

# Regression
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### Strengths and Weaknesses Analysis

**Analiza grešaka:**
- Gde model pravi greške? (confusion matrix analysis)
- Da li postoji pattern u pogrešnim predikcijama?
- Koje feature-e model koristi najviše?
- Da li postoji bias prema određenim grupama?

**Learning curves:**
- Train vs Validation accuracy kroz vreme
- **High bias (underfitting)**: Train i val accuracy obe niske
- **High variance (overfitting)**: Train accuracy visoka, val niska

### Hyperparameter Tuning

Optimizacija parametara modela koji se ne uče iz podataka.

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```
- **Pros**: Garantovano pronalazi najbolju kombinaciju u grid-u
- **Cons**: Sporo za veliki broj parametara

**Random Search:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10, 15]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=20,  # broj random kombinacija
    cv=5,
    random_state=42
)
```
- **Pros**: Brže, dobro za veliki search space
- **Cons**: Može propustiti najbolju kombinaciju

**Bayesian Optimization:**
- Pametniji pristup (Optuna, Hyperopt)
- Uči iz prethodnih iteracija
- Efikasniji od random search

### Model Robustness (Robusnost Modela)

**Testiranje:**
- **Noise injection** - Dodavanje šuma u podatke
- **Different data splits** - Stabilnost kroz različite splitove
- **Adversarial examples** - Edge cases
- **Cross-validation variance** - Mala variance = robusniji model

**Regularization:**
- **L1 (Lasso)** - Feature selection
- **L2 (Ridge)** - Smanjuje magnitude koeficijenata
- **Dropout** - Neural networks
- **Early stopping** - Zaustavi trening pre overfitting-a

### Iterative Improvements

ML je iterativan proces:
1. **Baseline model** → Evaluacija
2. **Feature engineering** → Re-train → Evaluacija
3. **Try different algorithms** → Evaluacija
4. **Hyperparameter tuning** → Evaluacija
5. **Ensemble methods** → Evaluacija
6. Ponavljaj dok ne dostigneš target performanse

## 9. Model Deployment (Puštanje u Produkciju)

Implementacija modela u produkcijsko okruženje za real-world upotrebu.

### Deployment Strategies

**Batch Prediction:**
- Periodični predictions (hourly, daily, weekly)
- Proces: Load data → Predict → Save results
- **Primena**: Email marketing campaigns, fraud detection reviews

**Real-time Prediction:**
- Instant predictions putem API-ja
- Latency < 100ms obično poželjno
- **Primena**: Recommendation systems, fraud detection, chatbots

**Edge Deployment:**
- Model radi na uređaju (mobile, IoT)
- **Prednosti**: Brži, privacy, offline capability
- **Primena**: Mobile apps, autonomous vehicles

### Model Serialization

Čuvanje istreniranog modela:
```python
import joblib
import pickle

# Joblib (preporučeno za sklearn)
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')

# Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# TensorFlow/Keras
model.save('model.h5')

# ONNX (cross-platform)
# Koristi se za interoperability između framework-a
```

### API Development

**Flask/FastAPI primer:**
```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
```

**REST API endpoints:**
- **POST /predict** - Dobija features, vraća prediction
- **GET /health** - Status provera
- **GET /model-info** - Metadata o modelu

### Scalability (Skalabilnost)

**Horizontal Scaling:**
- Više instanci API-ja iza load balancer-a
- Auto-scaling na osnovu traffic-a

**Vertical Scaling:**
- Jači serveri (više CPU/RAM)

**Optimizacije:**
- **Model compression** - Quantization, pruning
- **Caching** - Keširaj česte predictions
- **Batch processing** - Grupisi requests
- **GPU acceleration** - Za deep learning modele

### Security (Sigurnost)

**Best Practices:**
- **Authentication** - API keys, OAuth tokens
- **Rate limiting** - Spread requests limitovanja
- **Input validation** - Proveri format i opseg
- **Encryption** - HTTPS za sve komunikacije
- **Logging** - Trackuj sve requests (ali ne sensitive data)
- **Model versioning** - Mogućnost rollback-a

**Privacy:**
- **Data anonymization** - Ukloni PII (Personally Identifiable Information)
- **Differential privacy** - Dodaj noise za zaštitu
- **Federated learning** - Trening bez centralizovanog data

### Containerization & Orchestration

**Docker:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes:**
- Orkestracija kontejnera
- Auto-scaling, load balancing, health checks

**Cloud Platforms:**
- **AWS SageMaker** - End-to-end ML platform
- **Google AI Platform** - Managed ML services
- **Azure ML** - Microsoft cloud ML
- **Heroku, Railway** - Jednostavniji deploy

## 10. Model Monitoring and Maintenance

Kontinuirano praćenje performansi modela i održavanje kvaliteta.

### Performance Tracking

**Key Metrics:**
- **Prediction accuracy** - Stvarni accuracy u produkciji
- **Latency** - Response time (p50, p95, p99 percentiles)
- **Throughput** - Requests per second
- **Error rate** - Neuspeli predictions
- **Resource usage** - CPU, memory, disk

**Monitoring Tools:**
- **Prometheus + Grafana** - Metrics collection i vizualizacija
- **ELK Stack** - Elasticsearch, Logstash, Kibana za logs
- **CloudWatch, Datadog, New Relic** - Cloud monitoring

**Custom Dashboards:**
```python
# Logging predictions i ground truth
import logging

logger.info({
    'prediction': pred,
    'actual': actual,
    'features': features,
    'timestamp': datetime.now()
})
```

### Data Drift Detection

**Šta je drift?**
- Promene u distribuciji input podataka tokom vremena
- Model je treniran na jednoj distribuciji, ali u produkciji dobija drugu

**Tipovi:**
- **Covariate Drift** - Promena u X (features)
- **Prior Probability Drift** - Promena u Y (target distribucija)
- **Concept Drift** - Promena u relaciji X → Y

**Detekcija:**
```python
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(train_feature, production_feature)
if p_value < 0.05:
    print("⚠️ Drift detected!")
```

**Monitoring strategije:**
- **Statistical tests** - KS test, Chi-square
- **Distribucija features** - Histogram comparison
- **Population Stability Index (PSI)**
- **Model performance metrics** - Accuracy degradation

### Concept Drift

Promena u vezi između features i target-a.

**Primeri:**
- **Retail**: Kupovine navike se menjaju sezonski
- **Finance**: Market conditions se drastično menjaju
- **Healthcare**: Novi tretmani menjaju outcomes

**Handling:**
- **Sliding window** - Treniranje na nedavnim podacima
- **Adaptive learning** - Online learning algorithms
- **Ensemble models** - Weighted kombinacija starih i novih modela

### Model Retraining

**Kada retrenirati?**
- **Scheduled** - Periodično (weekly, monthly)
- **Performance-based** - Kada accuracy padne ispod threshold-a
- **Drift detection** - Automatski kada se detektuje drift
- **New data availability** - Kada dođe značajna količina novih podataka

**Retraining Pipeline:**
1. Collect new labeled data
2. Merge sa starim data (ili samo novi)
3. Re-run preprocessing pipeline
4. Train new model version
5. Validate (A/B test vs stari model)
6. Deploy ako je bolji
7. Monitor new model performance

**Automated ML Pipelines:**
```python
# Airflow DAG primer
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('model_retraining', schedule_interval='@weekly')

fetch_data = PythonOperator(task_id='fetch', python_callable=fetch_new_data, dag=dag)
preprocess = PythonOperator(task_id='preprocess', python_callable=preprocess_data, dag=dag)
train = PythonOperator(task_id='train', python_callable=train_model, dag=dag)
deploy = PythonOperator(task_id='deploy', python_callable=deploy_model, dag=dag)

fetch_data >> preprocess >> train >> deploy
```

### Logging and Alerts

**Šta logovati:**
- **Predictions** - Input, output, timestamp
- **Model version** - Koja verzija je napravila prediction
- **Errors** - Exception traces
- **Performance metrics** - Accuracy, latency
- **Data stats** - Feature distributions

**Alerting triggers:**
- Accuracy drops below threshold
- Latency exceeds SLA
- Error rate spike
- Data drift detected
- Resource exhaustion (memory, disk)

**Alert channels:**
- Email, Slack, PagerDuty
- Automated incident creation
```python
# Alert primer
if current_accuracy < target_accuracy * 0.95:
    send_alert(
        severity='HIGH',
        message=f'Model accuracy dropped to {current_accuracy}',
        channel='#ml-alerts'
    )
```

### A/B Testing

Postepeno rollout novog modela uz monitoring:

**Strategy:**
- **Control group (A)**: Stari model - 90% traffic
- **Treatment group (B)**: Novi model - 10% traffic
- Prati performance metrics
- Postepeno povećavaj B (10% → 50% → 100%)
- Rollback ako B underperforms

**Metrics to compare:**
- Business metrics (conversion, revenue, engagement)
- Technical metrics (latency, accuracy)
- User satisfaction (surveys, feedback)

### Model Versioning

**Tracking:**
- **Model artifacts** - .pkl, .h5 fajlovi
- **Code version** - Git commit hash
- **Data version** - Dataset snapshot ili checksum
- **Hyperparameters** - Config fajlovi
- **Dependencies** - requirements.txt, environment

**Tools:**
- **MLflow** - Experiment tracking, model registry
- **DVC** - Data version control
- **Weights & Biases** - Experiment tracking
- **Neptune.ai** - ML metadata store
```python
import mlflow

mlflow.start_run()
mlflow.log_params({"n_estimators": 100, "max_depth": 10})
mlflow.log_metrics({"accuracy": 0.95, "f1": 0.93})
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

### Documentation

**Maintain:**
- **Model cards** - Purpose, performance, limitations, biases
- **API documentation** - Endpoints, request/response formats
- **Deployment guide** - How to deploy, configure
- **Monitoring runbooks** - How to respond to alerts
- **Data schemas** - Expected input format

## Best Practices - Ceo Pipeline

1. **Start simple** - Baseline model prvo
2. **Version everything** - Code, data, models
3. **Automate** - CI/CD pipelines za ML
4. **Monitor continuously** - Ne puštaj i zaboravi
5. **Document thoroughly** - Budući ti će zahvaliti
6. **Test extensively** - Unit tests, integration tests
7. **Iterate based on feedback** - Production je learning opportunity
8. **Consider ethics** - Bias, fairness, privacy
9. **Plan for failure** - Fallback strategies, graceful degradation
10. **Keep learning** - ML field se brzo menja

---

**Ovaj proces nije linearan - često se vraćaš nazad kroz korake dok ne dostigneš optimalne performanse!** 🔄