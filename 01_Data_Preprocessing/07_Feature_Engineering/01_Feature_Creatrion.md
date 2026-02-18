# Feature Creation

Feature Creation je proces **kreiranja novih promenljivih iz postojećih** da bi se poboljšala sposobnost modela da nauči obrasce u podacima. Ovo je **najkreativniji i najmoćniji** deo feature engineering-a!

**Zašto je feature creation bitan?**
- **Otkriva skrivene obrasce** - Kombinacije features mogu biti informatvnije od pojedinačnih
- **Poboljšava model performanse** - Često značajno povećava accuracy
- **Enkoduje domain znanje** - Pretvara ekspertsko znanje u features
- **Linearizuje odnose** - Pomaže linear modelima da uhvate non-linear obrasce
- **Reduces model complexity** - Jedan dobar feature može zameniti više loših

**VAŽNO:** Feature creation se radi **POSLE** data cleaning i EDA, a **PRE** scaling-a!

---

## Kada Raditi Feature Creation?

### EDA Insights → Feature Creation Ideas

| Problem u EDA | Feature Creation Rešenje |
|---------------|--------------------------|
| **Non-linear relationship** | Polynomial features, interactions |
| **Cyclical patterns** (vreme) | Sin/Cos transformacije |
| **Group-based differences** | Aggregations po grupama |
| **Temporal patterns** | Lag features, rolling windows |
| **Text data** | Length, word count, sentiment |
| **Missing patterns** | Binary "was_missing" flag |
| **Kategorije sa signal-om** | Target encoding, frequency encoding |

---

## 1. Interaction Features (Interakcije)

**Kombinovanje dva ili više features** da uhvatiš njihov zajednički uticaj.

### Zašto su Bitne?
```python
# Problem: Dva features pojedinačno nisu dovoljno

# Primer: Predviđanje prodaje
# price = 10 → Sales?
# quantity = 100 → Sales?

# ALI: price × quantity = Total Revenue! 💡
# Ova interakcija je direktno prediktivna za Sales!
```

### Tipovi Interakcija:

#### A) **Multiplication (Množenje)**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'price': [10, 20, 15, 25],
    'quantity': [100, 50, 80, 40],
    'discount': [0.1, 0.2, 0.15, 0.25]
})

# 1. Simple multiplication
df['total_revenue'] = df['price'] * df['quantity']

# 2. With discount
df['revenue_after_discount'] = df['price'] * df['quantity'] * (1 - df['discount'])

# 3. Multiple interactions
df['price_quantity_discount'] = df['price'] * df['quantity'] * df['discount']

print(df)
#    price  quantity  discount  total_revenue  revenue_after_discount
# 0     10       100      0.10           1000                     900
# 1     20        50      0.20           1000                     800
# 2     15        80      0.15           1200                    1020
# 3     25        40      0.25           1000                     750
```

**Kada koristiti:**
- **E-commerce**: price × quantity = revenue
- **Real estate**: area × rooms = spaciousness_score
- **Finance**: principal × interest_rate = expected_return
- **Health**: weight × height = BMI-related feature

#### B) **Addition/Subtraction (Sabiranje/Oduzimanje)**
```python
# Addition - Total counts
df['total_interactions'] = df['likes'] + df['comments'] + df['shares']

# Subtraction - Differences
df['profit'] = df['revenue'] - df['cost']
df['age_difference'] = df['user_age'] - df['avg_age']

# Change over time
df['price_change'] = df['current_price'] - df['previous_price']
df['price_change_pct'] = (df['current_price'] - df['previous_price']) / df['previous_price']
```

**Kada koristiti:**
- **Social media**: Ukupne interakcije
- **Finance**: Profit, margins, changes
- **Inventory**: Net change (in - out)
- **Temporal**: Deltas, differences

#### C) **Division (Deljenje) - Ratios**
```python
# Ratios are VERY informative!
df['price_per_sqft'] = df['price'] / df['square_feet']
df['revenue_per_employee'] = df['revenue'] / df['num_employees']
df['click_through_rate'] = df['clicks'] / df['impressions']
df['conversion_rate'] = df['purchases'] / df['visits']
df['engagement_rate'] = df['interactions'] / df['followers']

# Avoid division by zero!
df['price_per_sqft'] = df['price'] / (df['square_feet'] + 1e-6)

# Or use np.where
df['price_per_sqft'] = np.where(
    df['square_feet'] > 0,
    df['price'] / df['square_feet'],
    0
)
```

**Kada koristiti:**
- **Real estate**: Cena po kvadratu
- **Business**: Efficiency ratios (revenue per employee)
- **Marketing**: Conversion rates, CTR
- **Finance**: P/E ratio, debt-to-equity

#### D) **Pairwise Interactions (Sve kombinacije)**
```python
from itertools import combinations

# Kreiranje svih interakcija između numerical features
numerical_cols = ['age', 'income', 'credit_score']

# Multiplication interactions
for col1, col2 in combinations(numerical_cols, 2):
    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

# age_x_income, age_x_credit_score, income_x_credit_score

# Division interactions
for col1, col2 in combinations(numerical_cols, 2):
    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)

# Automatic function
def create_interactions(df, cols, operations=['multiply', 'divide']):
    """
    Kreiranje interakcija između kolona.
    """
    df_new = df.copy()
    
    for col1, col2 in combinations(cols, 2):
        if 'multiply' in operations:
            df_new[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        if 'divide' in operations:
            df_new[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
            df_new[f'{col2}_div_{col1}'] = df[col2] / (df[col1] + 1e-6)
        
        if 'add' in operations:
            df_new[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        
        if 'subtract' in operations:
            df_new[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
    return df_new

# Korišćenje
df_with_interactions = create_interactions(
    df, 
    ['age', 'income', 'credit_score'],
    operations=['multiply', 'divide']
)
```

**⚠️ OPREZ:** Broj features eksplodira! n features → n(n-1)/2 interakcija

---

## 2. Polynomial Features

**Kreiranje polinoma** (kvadrata, kubova) za modelovanje non-linear odnosa.

### Python Implementacija:
```python
from sklearn.preprocessing import PolynomialFeatures

df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 40000, 50000, 60000, 70000]
})

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df)

# Feature names
feature_names = poly.get_feature_names_out(df.columns)
df_poly = pd.DataFrame(poly_features, columns=feature_names)

print(df_poly.columns)
# ['age', 'income', 'age^2', 'age income', 'income^2']

print(df_poly.head())
#    age  income   age^2  age income  income^2
# 0   25   30000     625     750000   9.0e+08
# 1   30   40000     900    1200000   1.6e+09
# 2   35   50000    1225    1750000   2.5e+09
```

### Kontrola nad Features:
```python
# Samo kvadrati (bez interakcija)
poly_no_interact = PolynomialFeatures(
    degree=2, 
    interaction_only=False,  # Uključuje i kvadrate
    include_bias=False
)

# Samo interakcije (bez kvadrata)
poly_interact_only = PolynomialFeatures(
    degree=2, 
    interaction_only=True,   # Samo age×income, ne age² i income²
    include_bias=False
)

# Degree 3 (kubovi)
poly_cubic = PolynomialFeatures(degree=3, include_bias=False)
# age, income, age², age×income, income², age³, age²×income, age×income², income³
```

**Kada koristiti:**
- **Linear models** sa non-linear data
- **EDA pokazuje curve relationship**
- **Regression problemi** sa jasnim non-linearity

**⚠️ OPREZ:**
- Degree 3+ → MNOGO features → Overfitting risk
- UVEK koristi regularization (Ridge, Lasso) sa polynomial features!

---

## 3. Aggregation Features (Agregacije)

**Grupisanje i računanje statistika** - sum, mean, count, std po kategorijama ili grupama.

### Tipovi Agregacija:
```python
# E-commerce dataset
df_orders = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
    'order_date': pd.date_range('2024-01-01', periods=9, freq='D'),
    'order_amount': [100, 150, 200, 300, 250, 50, 75, 100, 80],
    'product_category': ['Electronics', 'Clothing', 'Electronics', 
                         'Electronics', 'Electronics', 'Clothing', 
                         'Clothing', 'Food', 'Food']
})

# 1. Per-User Aggregations
user_aggs = df_orders.groupby('user_id').agg({
    'order_amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
    'product_category': lambda x: x.nunique()  # Broj unique kategorija
}).reset_index()

user_aggs.columns = ['user_id', 'total_spent', 'avg_order', 'std_order', 
                     'min_order', 'max_order', 'num_orders', 'num_categories']

print(user_aggs)
#    user_id  total_spent  avg_order  std_order  min_order  max_order  num_orders  num_categories
# 0        1          450      150.0       50.0        100        200           3               2
# 1        2          550      275.0       35.4        250        300           2               1
# 2        3          305       76.25      21.5         50        100           4               2

# Merge sa original dataframe
df_orders = df_orders.merge(user_aggs, on='user_id', how='left')
```

### Recency, Frequency, Monetary (RFM) Features:
```python
# RFM Analysis - Marketing classic!
from datetime import datetime

current_date = df_orders['order_date'].max()

rfm = df_orders.groupby('user_id').agg({
    'order_date': lambda x: (current_date - x.max()).days,  # Recency
    'user_id': 'count',                                      # Frequency
    'order_amount': 'sum'                                    # Monetary
}).reset_index()

rfm.columns = ['user_id', 'recency_days', 'frequency', 'monetary_value']

print(rfm)
#    user_id  recency_days  frequency  monetary_value
# 0        1             6          3             450
# 1        2             4          2             550
# 2        3             0          4             305

# Additional RFM features
rfm['avg_order_value'] = rfm['monetary_value'] / rfm['frequency']
rfm['days_between_orders'] = rfm['recency_days'] / (rfm['frequency'] - 1)
```

### Rolling/Moving Aggregations:
```python
# Time series features - rolling window
df_ts = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'sales': np.random.randn(100).cumsum() + 100
})

# Rolling statistics (trailing windows)
df_ts['sales_rolling_7d_mean'] = df_ts['sales'].rolling(window=7).mean()
df_ts['sales_rolling_7d_std'] = df_ts['sales'].rolling(window=7).std()
df_ts['sales_rolling_30d_mean'] = df_ts['sales'].rolling(window=30).mean()

# Exponential weighted moving average
df_ts['sales_ewm_7d'] = df_ts['sales'].ewm(span=7).mean()

print(df_ts.tail())
#          date      sales  sales_rolling_7d_mean  sales_rolling_7d_std
# 95 2024-04-05  102.345              101.234              2.456
# 96 2024-04-06  103.456              101.567              2.123
```

**Kada koristiti:**
- **Per-user/customer features** - E-commerce, banking
- **Category-based features** - Retail, classification
- **Time-based aggregations** - Forecasting, time series
- **RFM features** - Customer segmentation, churn prediction

---

## 4. Date/Time Feature Extraction

**Ekstrakcija korisnih informacija iz datetime kolona.**

### Osnovne Komponente:
```python
df_dates = pd.DataFrame({
    'transaction_date': pd.date_range('2024-01-01', periods=365, freq='D')
})

# Convert to datetime (ako već nije)
df_dates['transaction_date'] = pd.to_datetime(df_dates['transaction_date'])

# 1. Temporal components
df_dates['year'] = df_dates['transaction_date'].dt.year
df_dates['month'] = df_dates['transaction_date'].dt.month
df_dates['day'] = df_dates['transaction_date'].dt.day
df_dates['day_of_week'] = df_dates['transaction_date'].dt.dayofweek  # 0=Monday, 6=Sunday
df_dates['day_of_year'] = df_dates['transaction_date'].dt.dayofyear
df_dates['week_of_year'] = df_dates['transaction_date'].dt.isocalendar().week
df_dates['quarter'] = df_dates['transaction_date'].dt.quarter

# 2. Time components (ako ima vreme)
df_times = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01 00:00:00', periods=24, freq='H')
})
df_times['hour'] = df_times['timestamp'].dt.hour
df_times['minute'] = df_times['timestamp'].dt.minute
df_times['second'] = df_times['timestamp'].dt.second

# 3. Boolean features
df_dates['is_weekend'] = df_dates['day_of_week'].isin([5, 6]).astype(int)
df_dates['is_month_start'] = df_dates['transaction_date'].dt.is_month_start.astype(int)
df_dates['is_month_end'] = df_dates['transaction_date'].dt.is_month_end.astype(int)
df_dates['is_quarter_start'] = df_dates['transaction_date'].dt.is_quarter_start.astype(int)

# 4. Name features
df_dates['day_name'] = df_dates['transaction_date'].dt.day_name()  # 'Monday', 'Tuesday', ...
df_dates['month_name'] = df_dates['transaction_date'].dt.month_name()  # 'January', 'February', ...

print(df_dates.head())
#    transaction_date  year  month  day  day_of_week  is_weekend  day_name
# 0        2024-01-01  2024      1    1            0           0    Monday
# 1        2024-01-02  2024      1    2            1           0   Tuesday
# 2        2024-01-03  2024      1    3            2           0 Wednesday
```

### Cyclical Encoding (Sin/Cos):
```python
# Problem: month=12 i month=1 su blizu, ali numerički daleko!
# Rešenje: Sin/Cos transformacija

def encode_cyclical_feature(df, col, max_val):
    """
    Enkoduje ciklične feature sa sin i cos.
    """
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

# Month (1-12)
df_dates = encode_cyclical_feature(df_dates, 'month', 12)

# Day of week (0-6)
df_dates = encode_cyclical_feature(df_dates, 'day_of_week', 7)

# Hour (0-23)
df_times = encode_cyclical_feature(df_times, 'hour', 24)

print(df_dates[['month', 'month_sin', 'month_cos']].head())
#    month  month_sin  month_cos
# 0      1      0.500      0.866
# 1      1      0.500      0.866
# 2      1      0.500      0.866

# December (12) i January (1) su sad bliski!
# month=12: sin≈0, cos≈1
# month=1:  sin≈0.5, cos≈0.866
```

### Time Since/Until Features:
```python
# Reference date
reference_date = pd.Timestamp('2024-06-15')

# Days since
df_dates['days_since_jan1'] = (df_dates['transaction_date'] - pd.Timestamp('2024-01-01')).dt.days
df_dates['days_since_reference'] = (df_dates['transaction_date'] - reference_date).dt.days

# Days until next event
next_holiday = pd.Timestamp('2024-12-25')
df_dates['days_until_christmas'] = (next_holiday - df_dates['transaction_date']).dt.days

print(df_dates[['transaction_date', 'days_since_jan1', 'days_until_christmas']].head())
```

### Age Calculation:
```python
# Starost od datuma rođenja
df_users = pd.DataFrame({
    'birth_date': ['1990-05-15', '1985-08-20', '2000-12-01']
})

df_users['birth_date'] = pd.to_datetime(df_users['birth_date'])
current_date = pd.Timestamp('2024-02-16')

df_users['age'] = (current_date - df_users['birth_date']).dt.days // 365

# Ili preciznije
from dateutil.relativedelta import relativedelta

df_users['age_precise'] = df_users['birth_date'].apply(
    lambda x: relativedelta(current_date, x).years
)

print(df_users)
#    birth_date  age  age_precise
# 0  1990-05-15   33           33
# 1  1985-08-20   38           38
# 2  2000-12-01   23           23
```

**Kada koristiti:**
- **Temporal patterns** - Sales forecasting, demand prediction
- **Seasonality** - Retail, tourism, energy consumption
- **User behavior** - Time of purchase, visit patterns
- **Age-based analysis** - Demographics, insurance, healthcare

---

## 5. Lag Features (Za Time Series)

**Prethodne vrednosti kao features** - koristi prošlost za predviđanje budućnosti.

### Python Implementacija:
```python
# Time series sales data
df_sales = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'sales': np.random.randn(100).cumsum() + 100
})

# Lag features (prethodne vrednosti)
df_sales['sales_lag_1'] = df_sales['sales'].shift(1)   # Jučerašnja prodaja
df_sales['sales_lag_7'] = df_sales['sales'].shift(7)   # Pre 7 dana
df_sales['sales_lag_30'] = df_sales['sales'].shift(30) # Pre 30 dana

# Lead features (buduće vrednosti - SAMO za feature engineering, NE za target!)
df_sales['sales_lead_1'] = df_sales['sales'].shift(-1)  # Sutrašnja prodaja

# Difference features (delta)
df_sales['sales_diff_1'] = df_sales['sales'].diff(1)    # Change od juče
df_sales['sales_diff_7'] = df_sales['sales'].diff(7)    # Change od pre 7 dana

# Percentage change
df_sales['sales_pct_change_1'] = df_sales['sales'].pct_change(1)

print(df_sales.head(10))
#         date      sales  sales_lag_1  sales_lag_7  sales_diff_1
# 0 2024-01-01  100.123          NaN          NaN           NaN
# 1 2024-01-02  101.456      100.123          NaN         1.333
# 2 2024-01-03   99.789      101.456          NaN        -1.667
# ...
```

### Rolling Features sa Lag:
```python
# Kombinacija rolling i lag
df_sales['rolling_mean_7d_lag_1'] = df_sales['sales'].shift(1).rolling(window=7).mean()

# Ovo je: "Prosek prodaje poslednjih 7 dana, ali BEZ današnje"
# Koristi se za predviđanje današnje prodaje!

# Multiple lags
for lag in [1, 2, 3, 7, 14, 30]:
    df_sales[f'sales_lag_{lag}'] = df_sales['sales'].shift(lag)
```

### ⚠️ KRITIČNO - Data Leakage!
```python
# ❌ LOŠE - Koristi buduće podatke!
df_sales['future_sales_avg'] = df_sales['sales'].rolling(window=7).mean()
# Rolling gleda i današnju i buduće prodaje!

# ✅ DOBRO - Samo prošlost
df_sales['past_sales_avg'] = df_sales['sales'].shift(1).rolling(window=7).mean()
# Shift prvo, PA ONDA rolling!
```

**Kada koristiti:**
- **Time series forecasting** - Stock prices, sales, demand
- **Sequential data** - Sensor readings, weather data
- **Autoregressive models** - ARIMA-style features za ML

---

## 6. Domain-Specific Features

**Features specifične za domen problema** - ovde ekspertsko znanje dolazi do izražaja!

### Real Estate Example:
```python
df_houses = pd.DataFrame({
    'price': [300000, 450000, 600000],
    'square_feet': [1500, 2000, 2500],
    'num_bedrooms': [3, 4, 5],
    'num_bathrooms': [2, 2.5, 3],
    'lot_size': [5000, 7000, 10000],
    'year_built': [1990, 2005, 2015],
    'garage_spaces': [2, 2, 3]
})

current_year = 2024

# Domain features
df_houses['price_per_sqft'] = df_houses['price'] / df_houses['square_feet']
df_houses['house_age'] = current_year - df_houses['year_built']
df_houses['bedrooms_per_sqft'] = df_houses['num_bedrooms'] / df_houses['square_feet']
df_houses['bathrooms_per_bedroom'] = df_houses['num_bathrooms'] / df_houses['num_bedrooms']
df_houses['living_space_ratio'] = df_houses['square_feet'] / df_houses['lot_size']
df_houses['total_rooms'] = df_houses['num_bedrooms'] + df_houses['num_bathrooms']
df_houses['has_multiple_bathrooms_per_bedroom'] = (df_houses['bathrooms_per_bedroom'] > 1).astype(int)

print(df_houses[['price', 'price_per_sqft', 'house_age', 'living_space_ratio']])
```

### E-Commerce Example:
```python
df_ecommerce = pd.DataFrame({
    'user_id': [1, 2, 3],
    'total_orders': [10, 5, 20],
    'total_spent': [1000, 500, 2500],
    'days_since_signup': [365, 180, 730],
    'avg_days_between_orders': [36, 36, 36],
    'returns': [1, 0, 3],
    'reviews_written': [5, 2, 15]
})

# E-commerce domain features
df_ecommerce['avg_order_value'] = df_ecommerce['total_spent'] / df_ecommerce['total_orders']
df_ecommerce['orders_per_month'] = df_ecommerce['total_orders'] / (df_ecommerce['days_since_signup'] / 30)
df_ecommerce['return_rate'] = df_ecommerce['returns'] / df_ecommerce['total_orders']
df_ecommerce['review_rate'] = df_ecommerce['reviews_written'] / df_ecommerce['total_orders']
df_ecommerce['customer_lifetime_value'] = df_ecommerce['total_spent'] / (df_ecommerce['days_since_signup'] / 365)
df_ecommerce['engagement_score'] = (
    df_ecommerce['orders_per_month'] * 0.4 + 
    df_ecommerce['review_rate'] * 0.3 + 
    (1 - df_ecommerce['return_rate']) * 0.3
)
```

### Healthcare Example:
```python
df_health = pd.DataFrame({
    'weight_kg': [70, 85, 95],
    'height_cm': [170, 180, 175],
    'age': [30, 45, 50],
    'systolic_bp': [120, 140, 150],
    'diastolic_bp': [80, 90, 95],
    'glucose': [90, 110, 130],
    'cholesterol': [180, 220, 250]
})

# Healthcare features
df_health['bmi'] = df_health['weight_kg'] / ((df_health['height_cm'] / 100) ** 2)
df_health['bmi_category'] = pd.cut(df_health['bmi'], 
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df_health['pulse_pressure'] = df_health['systolic_bp'] - df_health['diastolic_bp']
df_health['mean_arterial_pressure'] = df_health['diastolic_bp'] + (df_health['pulse_pressure'] / 3)
df_health['is_hypertensive'] = ((df_health['systolic_bp'] >= 140) | 
                                 (df_health['diastolic_bp'] >= 90)).astype(int)
df_health['metabolic_risk_score'] = (
    (df_health['bmi'] > 25).astype(int) +
    (df_health['glucose'] > 100).astype(int) +
    (df_health['cholesterol'] > 200).astype(int) +
    df_health['is_hypertensive']
)
```

**Ključ za domain features:**
1. **Konsultuj eksperte** - Lekari, inženjeri, analitičari
2. **Istraži literaturu** - Naučni radovi, industry reports
3. **Analiziraj konkurenciju** - Šta drugi koriste?
4. **Testiraj i validiraj** - Da li feature dodaje value?

---

## 7. Text Features

**Ekstrakcija numeričkih features iz teksta.**

### Osnovne Text Features:
```python
df_text = pd.DataFrame({
    'review': [
        'This product is amazing!',
        'Worst purchase ever. Do not buy.',
        'It is okay, nothing special.',
        'Absolutely love it! Highly recommend.'
    ]
})

# 1. Length features
df_text['char_count'] = df_text['review'].str.len()
df_text['word_count'] = df_text['review'].str.split().str.len()
df_text['avg_word_length'] = df_text['char_count'] / df_text['word_count']

# 2. Uppercase/punctuation
df_text['uppercase_count'] = df_text['review'].str.count(r'[A-Z]')
df_text['punctuation_count'] = df_text['review'].str.count(r'[!?.,]')
df_text['exclamation_count'] = df_text['review'].str.count(r'!')

# 3. Specific words
df_text['has_not'] = df_text['review'].str.contains('not', case=False).astype(int)
df_text['positive_words'] = df_text['review'].str.count(r'\b(love|amazing|great|excellent)\b', case=False)
df_text['negative_words'] = df_text['review'].str.count(r'\b(worst|terrible|bad|awful)\b', case=False)

# 4. Sentiment polarity (simple)
df_text['sentiment_score'] = df_text['positive_words'] - df_text['negative_words']

print(df_text[['review', 'word_count', 'sentiment_score']])
#                                review  word_count  sentiment_score
# 0          This product is amazing!           4                1
# 1  Worst purchase ever. Do not buy.           6               -1
# 2        It is okay, nothing special.          5                0
# 3  Absolutely love it! Highly recommend.      5                1
```

### Advanced Text Features (NLP):
```python
# TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=50)  # Top 50 words
tfidf_matrix = tfidf.fit_transform(df_text['review'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf.get_feature_names_out()
)

# Sentiment analysis (sa library)
# !pip install textblob
from textblob import TextBlob

df_text['polarity'] = df_text['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df_text['subjectivity'] = df_text['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

print(df_text[['review', 'polarity', 'subjectivity']])
```

**Kada koristiti:**
- **Product reviews** - Sentiment, rating prediction
- **Customer support** - Ticket classification, priority
- **Social media** - Engagement prediction, toxicity detection
- **Document classification** - Spam detection, topic modeling

---

## 8. Missing Value Indicators

**Binary feature koji pokazuje da li je vrednost bila missing.**

### Zašto je Bitan?
```python
# Ponekad MISSING VALUE sam sadrži informaciju!

# Primer: Ako income je missing → Možda je osoba nezaposlena?
# Primer: Ako medical_history je missing → Možda je nova u sistemu?
```

### Python Implementacija:
```python
df_missing = pd.DataFrame({
    'age': [25, np.nan, 35, 40],
    'income': [30000, 40000, np.nan, 60000],
    'credit_score': [650, 700, np.nan, 800]
})

# 1. Binary missing indicators
for col in ['age', 'income', 'credit_score']:
    df_missing[f'{col}_was_missing'] = df_missing[col].isnull().astype(int)

# 2. Total missing count per row
df_missing['total_missing'] = df_missing[['age', 'income', 'credit_score']].isnull().sum(axis=1)

print(df_missing)
#      age  income  credit_score  age_was_missing  income_was_missing  total_missing
# 0   25.0   30000         650.0                0                   0              0
# 1    NaN   40000         700.0                1                   0              1
# 2   35.0     NaN           NaN                0                   1              2
# 3   40.0   60000         800.0                0                   0              0

# PA ONDA impute missing values
df_missing['age'].fillna(df_missing['age'].median(), inplace=True)
df_missing['income'].fillna(df_missing['income'].median(), inplace=True)
df_missing['credit_score'].fillna(df_missing['credit_score'].median(), inplace=True)

# Sad imaš I original values (imputed) I informaciju da li je bilo missing!
```

---

## Feature Creation Strategy - Decision Framework
```
EDA Pokazuje...
│
├─→ Non-linear relationship?
│   └─→ Polynomial features, Interactions
│
├─→ Group differences?
│   └─→ Aggregations (mean, sum, count po grupama)
│
├─→ Temporal patterns?
│   └─→ Date/Time extraction, Lag features, Rolling
│
├─→ Ratio makes sense? (X per Y)
│   └─→ Division interactions (price/sqft, revenue/employee)
│
├─→ Cyclical data? (seasons, hours)
│   └─→ Sin/Cos encoding
│
├─→ Domain knowledge suggests features?
│   └─→ Custom domain-specific features
│
├─→ Text data?
│   └─→ Length, word count, sentiment
│
└─→ Missing has pattern?
    └─→ Missing indicators
```

---

## Best Practices

### ✅ DO:

**1. Start sa Domain Knowledge**
```python
# Najbolje features dolaze iz razumevanja problema!
# Konsultuj eksperte, istraži literaturu
```

**2. Check Feature Importance**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10))
# Kreiraj više features kao top 10!
```

**3. Validate New Features**
```python
# Pre nego dodaš feature:
# 1. Correlation sa target-om
correlation = df[['new_feature', 'target']].corr().iloc[0, 1]
print(f"Correlation with target: {correlation:.3f}")

# 2. Cross-validation improvement
from sklearn.model_selection import cross_val_score

# Bez feature
score_without = cross_val_score(model, X_without, y, cv=5).mean()

# Sa feature
score_with = cross_val_score(model, X_with, y, cv=5).mean()

print(f"Improvement: {score_with - score_without:.3f}")
```

**4. Dokumentuj Feature Logic**
```python
feature_definitions = {
    'price_per_sqft': 'price / square_feet',
    'days_since_last_order': '(current_date - last_order_date).days',
    'engagement_rate': 'interactions / followers',
    'bmi': 'weight / (height ** 2)'
}

# Sačuvaj u JSON ili dokumentaciju
import json
with open('feature_definitions.json', 'w') as f:
    json.dump(feature_definitions, f, indent=2)
```

**5. Iterative Process**
```python
# 1. Kreiraj batch features
# 2. Train model
# 3. Check feature importance
# 4. Remove low-importance features
# 5. Kreiraj nove features na osnovu insights
# 6. Repeat
```

### ❌ DON'T:

**1. Ne Koristi Target u Feature Creation (Data Leakage!)**
```python
# ❌ LOŠE - Target leakage!
df['avg_price_by_category'] = df.groupby('category')['price'].transform('mean')
# 'price' JE target! Ne sme biti u feature!

# ✅ DOBRO
df['avg_sqft_by_category'] = df.groupby('category')['square_feet'].transform('mean')
```

**2. Ne Kreiraj Redundant Features**
```python
# ❌ LOŠE - Duplo ista informacija
df['total_revenue'] = df['price'] * df['quantity']
df['revenue_total'] = df['price'] * df['quantity']  # Isti feature!

# Check correlation
corr_matrix = df.corr()
# Remove features sa correlation > 0.95
```

**3. Ne Kreiraj Features sa Svim NaN**
```python
# ❌ LOŠE
df['useless_feature'] = df['col1'] / df['col2']
# Ako je col2 uvek 0 → sve NaN!

# ✅ DOBRO - Check prvo
if (df['col2'] == 0).all():
    print("Cannot create feature - division by zero!")
else:
    df['new_feature'] = df['col1'] / (df['col2'] + 1e-6)
```

**4. Ne Zaboravi Handle Future Data (Time Series)**
```python
# ❌ LOŠE - Koristi buduće podatke!
df['rolling_mean'] = df['sales'].rolling(window=7).mean()

# ✅ DOBRO - Samo prošlost
df['rolling_mean'] = df['sales'].shift(1).rolling(window=7).mean()
```

---

## Automatski Feature Creation (Tools)

### Featuretools - Automated Feature Engineering:
```python
# !pip install featuretools

import featuretools as ft

# Sample data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'signup_date': ['2020-01-01', '2020-06-15', '2021-03-20']
})

orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4, 5],
    'customer_id': [1, 1, 2, 3, 3],
    'order_date': ['2021-01-10', '2021-02-15', '2021-03-01', '2021-04-05', '2021-05-10'],
    'order_amount': [100, 150, 200, 300, 250]
})

# Create EntitySet
es = ft.EntitySet(id='customers_orders')
es = es.add_dataframe(dataframe_name='customers', dataframe=customers, index='customer_id')
es = es.add_dataframe(dataframe_name='orders', dataframe=orders, index='order_id', time_index='order_date')

# Add relationship
es = es.add_relationship('customers', 'customer_id', 'orders', 'customer_id')

# Automated feature generation
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2
)

print(feature_matrix.columns)
# Automatski kreirane features: COUNT(orders), SUM(orders.order_amount), MEAN(orders.order_amount), itd.
```

---

## Rezime - Feature Creation Checklist

### Tipovi Features:

| Tip | Primeri | Kada |
|-----|---------|------|
| **Interactions** | price × quantity, age × income | Kombinovani uticaj |
| **Polynomial** | age², income³ | Non-linear relationships |
| **Aggregations** | SUM, MEAN, COUNT po grupama | Group differences |
| **Date/Time** | month, day_of_week, hour | Temporal patterns |
| **Lag** | sales_lag_7, rolling_mean | Time series |
| **Domain** | BMI, price_per_sqft | Specifično za problem |
| **Text** | word_count, sentiment | Text data |
| **Missing** | was_missing flags | Missing pattern |

### Process:
```
1. EDA → Identify patterns
2. Domain Knowledge → Brainstorm features
3. Create Features → Implement
4. Validate → Check importance & correlation
5. Iterate → Refine i kreiraj nove
```

**Key Takeaway:** Feature Creation je najkreativniji deo ML! Dobar feature može više pomoći od složenog modela. "Data > Algorithm" - bolji podaci (features) daju bolje rezultate! 🚀