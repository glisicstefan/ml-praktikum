# Regression Metrics

Regression metrics su **metode za evaluaciju performansi regression modela** - modela koji predviđaju kontinuirane numeričke vrednosti (cene, temperature, prodaje, itd). Različite metrike mere različite aspekte grešaka i izbor zavisi od prirode problema.

**Zašto je izbor metrike kritičan?**
- **Različite metrike penalizuju greške različito** - MSE penalizuje velike greške više od MAE
- **Scale podataka** - RMSE zavisi od scale, MAPE ne zavisi
- **Outliers** - MSE je osetljiv, MAE otporan
- **Interpretability** - RMSE je u istim jedinicama kao target, R² je percentualni

**VAŽNO:** Kao i kod classification, nikad ne koristi samo jednu metriku! Kombinuj više metrika + residual analysis.

---

## Osnove - Residuals (Reziduali)

**Residual** je razlika između stvarne vrednosti i predviđene vrednosti.
```
Residual = y_actual - y_predicted
         = y_true - y_pred

Pozitivan residual → Model PODPREDVIĐA (underpredicts)
Negativan residual → Model NADPREDVIĐA (overpredicts)
Residual = 0       → Perfektna predikcija
```

### Python:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 140, 190, 260, 295])

# Residuals
residuals = y_true - y_pred
print("Residuals:", residuals)
# [-10  10  10 -10   5]

# Interpretation:
# Sample 0: Predicted 110, actual 100 → Overpredicted by 10
# Sample 1: Predicted 140, actual 150 → Underpredicted by 10
```

---

## 1. MAE (Mean Absolute Error)

**Prosečna apsolutna greška** - prosek apsolutnih vrednosti residuala.

### Formula:
```
MAE = (1/n) × Σ|y_true - y_pred|
    = Prosek apsolutnih grešaka
```

### Karakteristike:
- **Jednako tretira sve greške** - greška od 10 uvek košta 10, bez obzira gde je
- **Otporan na outliere** - Velika greška ne dominira metriku
- **U istim jedinicama kao target** - Lako interpretirati
- **Ne penalizuje velike greške** - Za razliku od MSE

### Python:
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")

# Manual calculation
mae_manual = np.mean(np.abs(y_true - y_pred))
print(f"MAE (manual): {mae_manual:.2f}")
```

### Interpretacija:
```python
# MAE = 15.5 za house price prediction (u hiljadama $)

# Znači: U proseku, model greši za $15,500
# Može biti +$15,500 (underprediction) ili -$15,500 (overprediction)
```

### Kada Koristiti MAE:

✅ **DOBRO za:**
- **Outliers su validni** - Ne želiš da ekstremne vrednosti dominiraju
- **Jednaki cost svih grešaka** - Greška od 10 nije "gora" od greške od 5
- **Interpretability** - Lako objasniti stakeholderima
- **Robust evaluation** - Ne varira previše sa outlierima

❌ **LOŠE za:**
- **Velike greške su MNOGO skuplje** - Bolje koristi MSE/RMSE
- **Matematička optimizacija** - Nije derivable u 0 (manje pogodno za gradient descent)

### Primer - House Price Prediction:
```python
# House prices (u hiljadama $)
actual_prices = np.array([250, 320, 180, 450, 290])
predicted_prices = np.array([260, 310, 190, 420, 285])

mae_houses = mean_absolute_error(actual_prices, predicted_prices)
print(f"MAE: ${mae_houses:.1f}K")
# MAE: $12.0K

# Interpretacija: U proseku, greška u predikciji cene je $12,000
```

---

## 2. MSE (Mean Squared Error)

**Prosečna kvadratna greška** - prosek kvadrata residuala.

### Formula:
```
MSE = (1/n) × Σ(y_true - y_pred)²
```

### Karakteristike:
- **Penalizuje velike greške** - Kvadrat grešaka → velika greška = MNOGO veća kazna
- **Osetljiv na outliere** - Jedan outlier može drastično povećati MSE
- **NE u istim jedinicama** - Kvadratne jedinice (teže za interpretaciju)
- **Matematički zgodan** - Derivable svuda (dobar za optimizaciju)

### Python:
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")

# Manual
mse_manual = np.mean((y_true - y_pred) ** 2)
print(f"MSE (manual): {mse_manual:.2f}")
```

### Zašto Kvadrat?
```python
# Bez kvadrata - pozitivni i negativni residuali se canceluju!
residuals = y_true - y_pred  # [-10, 10, 10, -10, 5]
mean_residual = np.mean(residuals)
print(f"Mean residual: {mean_residual:.2f}")  # Blizu 0, ali greške postoje!

# Sa kvadratom - sve greške postaju pozitivne
squared_residuals = (y_true - y_pred) ** 2
mean_squared = np.mean(squared_residuals)
print(f"MSE: {mean_squared:.2f}")  # Odražava magnitude grešaka
```

### Outlier Impact:
```python
# Normal predictions
y_true_normal = np.array([100, 150, 200, 250, 300])
y_pred_normal = np.array([105, 145, 195, 255, 295])

mae_normal = mean_absolute_error(y_true_normal, y_pred_normal)
mse_normal = mean_squared_error(y_true_normal, y_pred_normal)

print(f"Normal - MAE: {mae_normal:.2f}, MSE: {mse_normal:.2f}")

# With one outlier
y_true_outlier = np.array([100, 150, 200, 250, 300])
y_pred_outlier = np.array([105, 145, 195, 255, 500])  # Huge error!

mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)
mse_outlier = mean_squared_error(y_true_outlier, y_pred_outlier)

print(f"Outlier - MAE: {mae_outlier:.2f}, MSE: {mse_outlier:.2f}")

# MSE eksplodira zbog outlier-a, MAE raste linearno!
```

### Kada Koristiti MSE:

✅ **DOBRO za:**
- **Velike greške su MNOGO skupije** - Penalizacija kvadratom
- **Model optimization** - Loss function za gradient descent
- **Teorijska analiza** - Matematički poželjan

❌ **LOŠE za:**
- **Interpretability** - Kvadratne jedinice (teško objasniti)
- **Outliers mogu biti validni** - MSE ih previše penalizuje
- **Reporting stakeholderima** - Koristi RMSE umesto

---

## 3. RMSE (Root Mean Squared Error)

**Koren prosečne kvadratne greške** - jednostavno, kvadratni koren MSE.

### Formula:
```
RMSE = √MSE
     = √[(1/n) × Σ(y_true - y_pred)²]
```

### Karakteristike:
- **U istim jedinicama kao target** - Direktno interpretabilno! ✅
- **Penalizuje velike greške** - Kao MSE
- **Osetljiv na outliere** - Kao MSE
- **Najpopularnija metrika** - Za regression probleme

### Python:
```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_true, y_pred, squared=False)  # squared=False → RMSE
print(f"RMSE: {rmse:.2f}")

# Manual
rmse_manual = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE (manual): {rmse_manual:.2f}")

# Ili
rmse_manual2 = np.sqrt(np.mean((y_true - y_pred) ** 2))
print(f"RMSE (manual2): {rmse_manual2:.2f}")
```

### Interpretacija:
```python
# RMSE = 18.5 za temperature prediction (u °C)

# Znači: U proseku, predikcija temperature greši za ±18.5°C
# (sa težim penalizovanjem velikih grešaka)
```

### MAE vs RMSE Comparison:
```python
# Create dataset sa različitim error patterns
errors_uniform = np.random.uniform(-10, 10, 1000)  # Uniform errors
errors_normal = np.random.normal(0, 10, 1000)     # Normal errors
errors_outliers = np.concatenate([
    np.random.normal(0, 5, 950),
    np.random.uniform(-100, 100, 50)  # 5% outliers
])

for name, errors in [('Uniform', errors_uniform), 
                      ('Normal', errors_normal),
                      ('With Outliers', errors_outliers)]:
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    print(f"{name:15s} - MAE: {mae:6.2f}, RMSE: {rmse:6.2f}, RMSE/MAE: {rmse/mae:.2f}")

# Uniform         - MAE:   5.00, RMSE:   5.77, RMSE/MAE: 1.15
# Normal          - MAE:   7.98, RMSE:  10.02, RMSE/MAE: 1.26
# With Outliers   - MAE:   7.89, RMSE:  19.45, RMSE/MAE: 2.47 ← Velika razlika!

# RMSE/MAE ratio pokazuje koliko su outlieri prisutni!
# Ratio > 1.5 → Outliers postoje
```

### Kada Koristiti RMSE:

✅ **DOBRO za:**
- **Default regression metric** - Najčešće korišćena
- **Velike greške su skupije** - Kvadratna penalizacija
- **Interpretability + penalizacija** - Najbolje od oba sveta
- **Reporting** - Lako objasniti stakeholderima

❌ **LOŠE za:**
- **Outliers dominiraju** - Mogu iskriviti metriku
- **Robust evaluation** - Koristi MAE ako outlieri ne smeju dominirati

---

## 4. R² (R-squared / Coefficient of Determination)

**Procenat varijanse u target-u koji je objašnjen modelom.**

### Formula:
```
R² = 1 - (SS_res / SS_tot)

SS_res (Residual Sum of Squares) = Σ(y_true - y_pred)²
SS_tot (Total Sum of Squares)    = Σ(y_true - y_mean)²

R² pokazuje koliko je model bolji od "dumb" modela koji UVEK predviđa mean!
```

### Karakteristike:
- **Scale-independent** - Uvek između -∞ i 1
- **Interpretabilan** - "Model objašnjava X% varijanse"
- **Ne zavisi od jedinica** - Može porediti modele različitih problema
- **Može biti negativan!** - Ako model je gori od mean baseline

### Python:
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Manual calculation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (manual): {r2_manual:.3f}")
```

### Interpretacija R²:
```
R² = 1.0     - Perfektna predikcija (sve residuali = 0)
R² = 0.9     - Model objašnjava 90% varijanse (odličan!)
R² = 0.7     - Model objašnjava 70% varijanse (dobar)
R² = 0.5     - Model objašnjava 50% varijanse (OK)
R² = 0.0     - Model je jednako dobar kao predviđanje mean-a (beskoristan)
R² < 0       - Model je GORI od predviđanja mean-a! (katastrofa)
```

### Baseline Comparison:
```python
# "Dumb" model - uvek predviđa mean
y_mean_pred = np.full_like(y_true, y_true.mean(), dtype=float)

# R² za mean baseline
r2_baseline = r2_score(y_true, y_mean_pred)
print(f"R² (mean baseline): {r2_baseline:.3f}")  # 0.0

# Tvoj model
r2_model = r2_score(y_true, y_pred)
print(f"R² (model): {r2_model:.3f}")

# Ako R² > 0 → Model je bolji od mean
# Ako R² < 0 → Model je GORI od mean! (nešto ne valja)
```

### Visualizing R²:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Generate data
np.random.seed(42)
X_simple = np.linspace(0, 10, 100)
y_simple = 2 * X_simple + 1 + np.random.randn(100) * 2

# Fit simple linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_simple.reshape(-1, 1), y_simple)
y_pred_simple = model.predict(X_simple.reshape(-1, 1))

r2 = r2_score(y_simple, y_pred_simple)

# Plot
plt.figure(figsize=(12, 5))

# Left: Predictions vs Actual
plt.subplot(1, 2, 1)
plt.scatter(X_simple, y_simple, alpha=0.5, label='Actual')
plt.plot(X_simple, y_pred_simple, 'r-', linewidth=2, label='Predicted')
plt.axhline(y_simple.mean(), color='green', linestyle='--', label='Mean Baseline')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Predictions (R² = {r2:.3f})')
plt.legend()
plt.grid(True)

# Right: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_simple, y_pred_simple, alpha=0.5)
plt.plot([y_simple.min(), y_simple.max()], 
         [y_simple.min(), y_simple.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Negativan R² - Primer:
```python
# Model koji je gori od mean
y_true_bad = np.array([10, 20, 30, 40, 50])
y_pred_bad = np.array([50, 10, 40, 20, 30])  # Nasumično loš

r2_bad = r2_score(y_true_bad, y_pred_bad)
print(f"R² (bad model): {r2_bad:.3f}")  # Negativno!

# Mean baseline
r2_mean = r2_score(y_true_bad, np.full_like(y_true_bad, y_true_bad.mean(), dtype=float))
print(f"R² (mean baseline): {r2_mean:.3f}")  # 0.0

# R² < 0 znači: Model je GORI od prostog mean-a!
```

### Kada Koristiti R²:

✅ **DOBRO za:**
- **Poređenje modela** - Scale-independent
- **Explaining variance** - "Model captures X% of variation"
- **Quick assessment** - Jedan broj pokazuje overall fit
- **Communication** - Stakeholderi razumeju procente

❌ **LOŠE za:**
- **R² alone nije dovoljno** - Može biti visok, ali model loš (outlieri, overfitting)
- **Comparing datasets različitih scale-a** - Može biti misleading
- **Model selection alone** - Kombinuj sa RMSE, residual plots

---

## 5. Adjusted R²

**R² adjusted for broj features** - penalizuje dodavanje nepotrebnih features.

### Formula:
```
Adjusted R² = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]

n = broj samples
p = broj features (predictors)
```

### Zašto Adjusted R²?
```python
# Problem sa R²: UVEK raste (ili ostaje isti) kada dodaš feature!
# Čak i ako je feature potpuno nekoristan (random noise)

# Primer:
from sklearn.datasets import make_regression

# 5 features
X_5, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
model = LinearRegression()
model.fit(X_5, y)
r2_5 = model.score(X_5, y)
print(f"R² (5 features): {r2_5:.3f}")

# 50 features (45 su RANDOM!)
X_50 = np.column_stack([X_5, np.random.randn(100, 45)])
model.fit(X_50, y)
r2_50 = model.score(X_50, y)
print(f"R² (50 features): {r2_50:.3f}")  # Veći R², ali 45 features su noise!

# R² UVEK raste! Ovo je problem.
```

### Python:
```python
def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Calculate Adjusted R².
    
    Parameters:
    - y_true: actual values
    - y_pred: predicted values
    - n_features: number of features (p)
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    return adjusted_r2

# Example
n_features = 50
adj_r2 = adjusted_r2_score(y, model.predict(X_50), n_features)

print(f"R²:          {r2_50:.3f}")
print(f"Adjusted R²: {adj_r2:.3f}")  # Niži nego R² (penalizuje 50 features)
```

### R² vs Adjusted R²:
```python
# Test sa različitim brojem features
r2_scores = []
adj_r2_scores = []
feature_counts = range(5, 55, 5)

X_base, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

for n_feat in feature_counts:
    # Add random features
    if n_feat > 5:
        X_test = np.column_stack([X_base, np.random.randn(100, n_feat - 5)])
    else:
        X_test = X_base
    
    model = LinearRegression()
    model.fit(X_test, y)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y, y_pred)
    adj_r2 = adjusted_r2_score(y, y_pred, n_feat)
    
    r2_scores.append(r2)
    adj_r2_scores.append(adj_r2)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(feature_counts, r2_scores, 'o-', label='R²', linewidth=2)
plt.plot(feature_counts, adj_r2_scores, 's-', label='Adjusted R²', linewidth=2)
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('R² vs Adjusted R² (Adding Random Features)')
plt.legend()
plt.grid(True)
plt.show()

# R² raste, Adjusted R² pada! (penalizuje nepotrebne features)
```

### Kada Koristiti Adjusted R²:

✅ **DOBRO za:**
- **Model sa mnogo features** - Penalizuje feature bloat
- **Feature selection** - Poređenje modela različitog broja features
- **Avoiding overfitting** - Ne nagrađuje dodavanje noise features

---

## 6. MAPE (Mean Absolute Percentage Error)

**Prosečna apsolutna procentualna greška** - greška kao procenat actual value.

### Formula:
```
MAPE = (100/n) × Σ|y_true - y_pred| / |y_true|
     = Prosek (|greška| / |actual|) u procentima
```

### Karakteristike:
- **Scale-independent** - Procenat, ne zavisi od jedinica
- **Lako interpretirati** - "Model greši u proseku za X%"
- **Problem sa y_true = 0** - Division by zero!
- **Asymmetric** - Penalizuje overpredictions više od underpredictions

### Python:
```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.2%}")  # U procentima

# Manual
mape_manual = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"MAPE (manual): {mape_manual:.2f}%")
```

### Interpretacija:
```python
# MAPE = 8.5%

# Znači: U proseku, predikcija greši za 8.5% od actual value
# Ako actual = 100, greška je ~8.5
# Ako actual = 1000, greška je ~85
```

### Problem sa Zeros:
```python
# Data sa zeros
y_true_zeros = np.array([0, 10, 20, 30, 40])
y_pred_zeros = np.array([5, 12, 18, 32, 38])

try:
    mape_zeros = mean_absolute_percentage_error(y_true_zeros, y_pred_zeros)
except:
    print("Error: Cannot compute MAPE with y_true = 0!")

# Workaround: Add small epsilon
epsilon = 1e-10
y_true_adjusted = y_true_zeros + epsilon
mape_adjusted = mean_absolute_percentage_error(y_true_adjusted, y_pred_zeros)
print(f"MAPE (with epsilon): {mape_adjusted:.2%}")
```

### Asymmetry Problem:
```python
# MAPE je asymmetric!
actual = 100

# Case 1: Underprediction
pred_under = 90
error_under = abs((actual - pred_under) / actual) * 100
print(f"Underprediction (-10): MAPE = {error_under:.1f}%")  # 10%

# Case 2: Overprediction (same magnitude)
pred_over = 110
error_over = abs((actual - pred_over) / actual) * 100
print(f"Overprediction (+10):  MAPE = {error_over:.1f}%")   # 10%

# Ali ako gledaš sa druge strane:
# Actual = 90, Predicted = 100 → MAPE = 11.1% (više!)
# Actual = 110, Predicted = 100 → MAPE = 9.1% (manje!)

# MAPE penalizuje overpredictions više!
```

### Symmetric MAPE (SMAPE):
```python
def smape(y_true, y_pred):
    """
    Symmetric MAPE - tretira over i under predictions jednako.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

smape_value = smape(y_true, y_pred)
print(f"SMAPE: {smape_value:.2f}%")
```

### Kada Koristiti MAPE:

✅ **DOBRO za:**
- **Scale-independent comparison** - Uporedi različite probleme (sales, temperature, stock prices)
- **Interpretability** - Stakeholderi razumeju procente
- **Relative errors bitni** - Ne apsolutne greške

❌ **LOŠE za:**
- **Data sa zeros ili blizu nule** - Division by zero
- **Asymmetric costs** - Over i under predictions imaju različit cost
- **Low values dominiraju** - Greška od 1 na actual=2 je 50%, greška od 10 na actual=1000 je 1%

---

## 7. MSLE (Mean Squared Log Error)

**Prosečna kvadratna logaritamska greška** - MSE na log scale.

### Formula:
```
MSLE = (1/n) × Σ(log(y_true + 1) - log(y_pred + 1))²
```

### Karakteristike:
- **Penalizuje underpredictions** - Više od overpredictions
- **Handling exponential growth** - Dobar za data koji raste eksponencijalno
- **Relative errors** - Fokus na magnitude, ne apsolutne vrednosti
- **Samo za pozitivne vrednosti** - log zahteva y > 0

### Python:
```python
from sklearn.metrics import mean_squared_log_error

# Data mora biti pozitivna!
y_true_pos = np.array([100, 200, 300, 400, 500])
y_pred_pos = np.array([110, 190, 320, 380, 510])

msle = mean_squared_log_error(y_true_pos, y_pred_pos)
print(f"MSLE: {msle:.4f}")

# Manual
msle_manual = np.mean((np.log1p(y_true_pos) - np.log1p(y_pred_pos)) ** 2)
print(f"MSLE (manual): {msle_manual:.4f}")

# RMSLE (Root MSLE)
rmsle = np.sqrt(msle)
print(f"RMSLE: {rmsle:.4f}")
```

### Zašto Log Scale?
```python
# Log scale tretira relative greške jednako

# Case 1: Small scale
actual_small = 10
pred_small = 20
error_abs_small = abs(actual_small - pred_small)  # 10
error_log_small = abs(np.log1p(actual_small) - np.log1p(pred_small))  # ~0.64

# Case 2: Large scale (ista relativna greška - duplo veća vrednost)
actual_large = 100
pred_large = 200
error_abs_large = abs(actual_large - pred_large)  # 100 (10× veća!)
error_log_large = abs(np.log1p(actual_large) - np.log1p(pred_large))  # ~0.64 (ista!)

print(f"Absolute errors: {error_abs_small} vs {error_abs_large}")
print(f"Log errors: {error_log_small:.3f} vs {error_log_large:.3f}")

# Log scale tretira obe greške jednako jer su relativno iste (2×)!
```

### Under vs Over Prediction Asymmetry:
```python
# MSLE penalizuje underpredictions više!

actual = 100

# Underprediction (-50)
pred_under = 50
msle_under = (np.log1p(actual) - np.log1p(pred_under)) ** 2
print(f"MSLE (underprediction): {msle_under:.4f}")

# Overprediction (+50)
pred_over = 150
msle_over = (np.log1p(actual) - np.log1p(pred_over)) ** 2
print(f"MSLE (overprediction):  {msle_over:.4f}")

# Underprediction ima VEĆU MSLE!
```

### Kada Koristiti MSLE:

✅ **DOBRO za:**
- **Exponential growth data** - Sales, stock prices, population
- **Relative errors bitni** - Ne apsolutne
- **Penalizuj underpredictions** - Underpredicting sales je skuplje
- **Wide range of values** - Log scale normalizuje

❌ **LOŠE za:**
- **Negative values** - Log ne radi
- **Overpredictions su skupije** - MSLE favorizuje overpredictions
- **Interpretability** - Teže objasniti nego MAE/RMSE

---

## 8. Median Absolute Error

**Median apsolutnih grešaka** - robustan na outliere.

### Formula:
```
MedAE = median(|y_true - y_pred|)
```

### Python:
```python
from sklearn.metrics import median_absolute_error

medae = median_absolute_error(y_true, y_pred)
print(f"Median Absolute Error: {medae:.2f}")

# Manual
medae_manual = np.median(np.abs(y_true - y_pred))
print(f"MedAE (manual): {medae_manual:.2f}")
```

### MAE vs MedAE sa Outlierima:
```python
# Normal predictions
y_true_normal = np.array([100, 150, 200, 250, 300])
y_pred_normal = np.array([105, 145, 195, 255, 295])

mae_normal = mean_absolute_error(y_true_normal, y_pred_normal)
medae_normal = median_absolute_error(y_true_normal, y_pred_normal)

print(f"Normal - MAE: {mae_normal:.2f}, MedAE: {medae_normal:.2f}")

# With outlier
y_true_outlier = np.array([100, 150, 200, 250, 300])
y_pred_outlier = np.array([105, 145, 195, 255, 1000])  # Huge error!

mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)
medae_outlier = median_absolute_error(y_true_outlier, y_pred_outlier)

print(f"Outlier - MAE: {mae_outlier:.2f}, MedAE: {medae_outlier:.2f}")

# MedAE je MNOGO stabilniji!
```

### Kada Koristiti:

✅ **DOBRO za:**
- **Outliers postoje** - Robusna metrika
- **Median je reprezentativniji** - Skewed distribucije
- **Stable evaluation** - Ne varira sa ekstremnim vrednostima

---

## 9. Residual Analysis

**Analiza residuala** je KRITIČNA za razumevanje gde model greši!

### Residual Plot:
```python
from sklearn.linear_model import LinearRegression

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.randn(100) * 2

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Residuals
residuals = y - y_pred

# Residual Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals vs Predicted
axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted (Check for patterns)')
axes[0, 0].grid(True)

# 2. Histogram of Residuals
axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Histogram of Residuals (Should be ~Normal)')
axes[0, 1].grid(True)

# 3. Q-Q Plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Check Normality)')
axes[1, 0].grid(True)

# 4. Residuals vs Index (Time order)
axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Index (Order)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Index (Check for autocorrelation)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

### Što Tražimo u Residual Plot:
```python
# ✅ DOBRO:
# - Residuali random raspršeni oko 0
# - Nema očiglednog patterna
# - Konstantna varijansa (homoscedasticity)
# - Približno normalna distribucija

# ❌ LOŠE:
# - Curve pattern → Model ne hvata non-linearity
# - Funnel shape → Heteroscedasticity (varijansa raste)
# - Sistemski trend → Model miss-uje nešto
# - Outliers dominiraju
```

### Pattern Examples:
```python
# Generate different patterns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Good - Random residuals
X_good = np.linspace(0, 10, 100)
y_good = 2 * X_good + 1 + np.random.randn(100) * 1
model_good = LinearRegression().fit(X_good.reshape(-1, 1), y_good)
residuals_good = y_good - model_good.predict(X_good.reshape(-1, 1))
axes[0, 0].scatter(model_good.predict(X_good.reshape(-1, 1)), residuals_good, alpha=0.6)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('✅ GOOD - Random Pattern')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Residuals')

# 2. Bad - Curve (non-linearity)
X_curve = np.linspace(0, 10, 100)
y_curve = X_curve ** 2 + np.random.randn(100) * 5  # Quadratic relationship!
model_curve = LinearRegression().fit(X_curve.reshape(-1, 1), y_curve)
residuals_curve = y_curve - model_curve.predict(X_curve.reshape(-1, 1))
axes[0, 1].scatter(model_curve.predict(X_curve.reshape(-1, 1)), residuals_curve, alpha=0.6)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title('❌ BAD - Curve (Non-linear Data)')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')

# 3. Bad - Funnel (heteroscedasticity)
X_funnel = np.linspace(0, 10, 100)
y_funnel = 2 * X_funnel + 1 + np.random.randn(100) * X_funnel  # Variance increases!
model_funnel = LinearRegression().fit(X_funnel.reshape(-1, 1), y_funnel)
residuals_funnel = y_funnel - model_funnel.predict(X_funnel.reshape(-1, 1))
axes[1, 0].scatter(model_funnel.predict(X_funnel.reshape(-1, 1)), residuals_funnel, alpha=0.6)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_title('❌ BAD - Funnel (Heteroscedasticity)')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Residuals')

# 4. Bad - Trend
X_trend = np.linspace(0, 10, 100)
y_trend = 2 * X_trend + 1 + np.sin(X_trend) * 5 + np.random.randn(100) * 1  # Sine pattern!
model_trend = LinearRegression().fit(X_trend.reshape(-1, 1), y_trend)
residuals_trend = y_trend - model_trend.predict(X_trend.reshape(-1, 1))
axes[1, 1].scatter(model_trend.predict(X_trend.reshape(-1, 1)), residuals_trend, alpha=0.6)
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_title('❌ BAD - Systematic Trend')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()
```

---

## 10. Homoscedasticity (Konstantna Varijansa)

**Homoscedasticity** = Varijansa residuala je konstantna preko svih predicted values.

### Testiranje:
```python
# Breusch-Pagan test za heteroscedasticity
from scipy import stats

def breusch_pagan_test(residuals, X):
    """
    Breusch-Pagan test za heteroscedasticity.
    
    H0: Homoscedasticity (varijansa konstantna)
    H1: Heteroscedasticity (varijansa se menja)
    """
    # Squared residuals
    squared_residuals = residuals ** 2
    
    # Fit linear model: squared_residuals ~ X
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, squared_residuals)
    
    # R² from this model
    r2 = model.score(X, squared_residuals)
    
    # Test statistic: n × R²
    n = len(residuals)
    test_stat = n * r2
    
    # Chi-square distribution with k degrees of freedom (k = number of features)
    k = X.shape[1]
    p_value = 1 - stats.chi2.cdf(test_stat, k)
    
    return test_stat, p_value

# Test
test_stat, p_value = breusch_pagan_test(residuals, X)
print(f"Breusch-Pagan Test:")
print(f"Test Statistic: {test_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Reject H0: Heteroscedasticity detected! (varijansa se menja)")
else:
    print("→ Fail to reject H0: Homoscedasticity (varijansa konstantna)")
```

---

## Decision Framework - Koja Metrika?
```
Priroda podataka?
│
├─→ Isti importance svih grešaka?
│   └─→ **MAE** (jednostavna, robustna)
│
├─→ Velike greške su skupije?
│   └─→ **RMSE** (penalizuje velike greške)
│
├─→ Relativne greške bitnije od apsolutnih?
│   ├─→ Data pozitivna? → **MAPE** ili **MSLE**
│   └─→ Exponential growth? → **MSLE**
│
├─→ Outliers dominiraju?
│   └─→ **MAE** ili **Median AE** (robustni)
│
├─→ Comparing različite scale?
│   └─→ **R²** ili **MAPE** (scale-independent)
│
├─→ Feature selection (mnogo features)?
│   └─→ **Adjusted R²**
│
└─→ Ne znaš šta koristiti?
    └─→ Default: **RMSE + R² + Residual Analysis**
```

### Default Metrics by Problem:
```
House Price Prediction:
✅ Primary: RMSE (interpretable, penalizuje velike greške)
✅ Secondary: R², MAE
✅ Visual: Residual plots, Actual vs Predicted

Sales Forecasting:
✅ Primary: MAPE (relativna greška bitna)
✅ Secondary: RMSE, MSLE
✅ Visual: Time series residuals

Temperature Prediction:
✅ Primary: MAE (outlieri validni)
✅ Secondary: RMSE, R²
✅ Visual: Residual histogram

Stock Price Prediction:
✅ Primary: MSLE (exponential, relative)
✅ Secondary: MAPE
✅ Visual: Log-scale residuals
```

---

## Complete Example - House Price Prediction
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
import matplotlib.pyplot as plt
from scipy import stats

# ==================== 1. LOAD DATA ====================
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target  # Median house value (in $100k)

print(f"Dataset: {X.shape[0]} houses, {X.shape[1]} features")
print(f"Target range: ${y.min()*100:.1f}K - ${y.max()*100:.1f}K")

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================== 3. TRAIN MODEL ====================
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ==================== 4. PREDICTIONS ====================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ==================== 5. EVALUATION ====================

print("\n" + "="*60)
print("HOUSE PRICE PREDICTION - EVALUATION")
print("="*60)

# Metrics
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
medae_test = median_absolute_error(y_test, y_test_pred)

# Convert to $K
mae_dollars = mae_test * 100
rmse_dollars = rmse_test * 100
medae_dollars = medae_test * 100

print(f"\nTest Set Metrics:")
print(f"MAE:        ${mae_dollars:6.2f}K  (Average absolute error)")
print(f"RMSE:       ${rmse_dollars:6.2f}K  (Root mean squared error)")
print(f"Median AE:  ${medae_dollars:6.2f}K  (Median error - robust)")
print(f"R²:         {r2_test:6.3f}     (Variance explained)")
print(f"MAPE:       {mape_test:6.2%}     (Percentage error)")

# Train metrics (check overfitting)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
r2_train = r2_score(y_train, y_train_pred)

print(f"\nTrain Set Metrics (Overfitting Check):")
print(f"MAE:   ${mae_train*100:6.2f}K")
print(f"RMSE:  ${rmse_train*100:6.2f}K")
print(f"R²:    {r2_train:6.3f}")

# Overfitting gap
print(f"\nTrain-Test Gap:")
print(f"MAE gap:   ${(mae_test - mae_train)*100:6.2f}K")
print(f"RMSE gap:  ${(rmse_test - rmse_train)*100:6.2f}K")
print(f"R² gap:    {r2_train - r2_test:6.3f}")

# ==================== 6. RESIDUAL ANALYSIS ====================

residuals = y_test - y_test_pred

print(f"\nResidual Statistics:")
print(f"Mean:     {residuals.mean()*100:6.2f}K  (Should be ~0)")
print(f"Std Dev:  {residuals.std()*100:6.2f}K")
print(f"Min:      {residuals.min()*100:6.2f}K")
print(f"Max:      {residuals.max()*100:6.2f}K")

# ==================== 7. VISUALIZATIONS ====================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 7.1 Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.3)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price ($100K)')
axes[0, 0].set_ylabel('Predicted Price ($100K)')
axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2_test:.3f})')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 7.2 Residuals vs Predicted
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.3)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Price ($100K)')
axes[0, 1].set_ylabel('Residuals ($100K)')
axes[0, 1].set_title('Residuals vs Predicted')
axes[0, 1].grid(True)

# 7.3 Histogram of Residuals
axes[0, 2].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('Residuals ($100K)')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Distribution of Residuals')
axes[0, 2].grid(True)

# 7.4 Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')
axes[1, 0].grid(True)

# 7.5 Absolute Errors Distribution
abs_errors = np.abs(residuals) * 100  # in $K
axes[1, 1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(mae_dollars, color='red', linestyle='--', 
                   linewidth=2, label=f'MAE = ${mae_dollars:.1f}K')
axes[1, 1].axvline(medae_dollars, color='green', linestyle='--', 
                   linewidth=2, label=f'Median = ${medae_dollars:.1f}K')
axes[1, 1].set_xlabel('Absolute Error ($K)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Absolute Errors Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True)

# 7.6 Error by Price Range
price_bins = np.linspace(y_test.min(), y_test.max(), 10)
bin_indices = np.digitize(y_test, price_bins)
bin_errors = [residuals[bin_indices == i].mean() for i in range(1, len(price_bins))]
bin_centers = (price_bins[:-1] + price_bins[1:]) / 2

axes[1, 2].bar(bin_centers * 100, np.array(bin_errors) * 100, 
               width=(price_bins[1]-price_bins[0])*100*0.8,
               alpha=0.7, edgecolor='black')
axes[1, 2].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Actual Price ($K)')
axes[1, 2].set_ylabel('Mean Residual ($K)')
axes[1, 2].set_title('Error by Price Range')
axes[1, 2].grid(True, axis='y')

plt.tight_layout()
plt.savefig('house_price_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 8. OUTLIER ANALYSIS ====================

# Find worst predictions
abs_errors_array = np.abs(residuals)
worst_indices = np.argsort(abs_errors_array)[-10:]  # Top 10 worst

print("\n" + "="*60)
print("TOP 10 WORST PREDICTIONS")
print("="*60)

for i, idx in enumerate(worst_indices[::-1], 1):
    actual = y_test.iloc[idx]
    predicted = y_test_pred[idx]
    error = residuals.iloc[idx]
    
    print(f"\n{i}. Actual: ${actual*100:.1f}K, "
          f"Predicted: ${predicted*100:.1f}K, "
          f"Error: ${error*100:+.1f}K")

# ==================== 9. BUSINESS IMPACT ====================

print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

# Acceptable error threshold
acceptable_error = 0.5  # $50K acceptable error

within_threshold = np.sum(abs_errors_array < acceptable_error)
percentage_good = within_threshold / len(y_test) * 100

print(f"\nAcceptable Error Threshold: ${acceptable_error*100:.0f}K")
print(f"Predictions within threshold: {within_threshold}/{len(y_test)} ({percentage_good:.1f}%)")

# Error brackets
error_brackets = [
    (0, 0.25, "Excellent (< $25K)"),
    (0.25, 0.5, "Good ($25-50K)"),
    (0.5, 1.0, "Fair ($50-100K)"),
    (1.0, float('inf'), "Poor (> $100K)")
]

print("\nPrediction Quality Distribution:")
for low, high, label in error_brackets:
    count = np.sum((abs_errors_array >= low) & (abs_errors_array < high))
    pct = count / len(y_test) * 100
    print(f"{label:25s}: {count:5d} ({pct:5.1f}%)")
```

---

## Rezime - Regression Metrics

### Quick Reference Table:

| Metric | Formula | Units | Outlier Sensitive? | Range | Best For |
|--------|---------|-------|-------------------|-------|----------|
| **MAE** | mean(\|y - ŷ\|) | Same as y | No | 0-∞ | Robust, interpretable |
| **MSE** | mean((y - ŷ)²) | y² | Yes | 0-∞ | Optimization |
| **RMSE** | √MSE | Same as y | Yes | 0-∞ | **Default metric** |
| **R²** | 1 - SS_res/SS_tot | None | Medium | -∞ to 1 | Variance explained |
| **Adj R²** | Adjusted R² | None | Medium | -∞ to 1 | Many features |
| **MAPE** | mean(\|error/y\|) | Percent | Medium | 0-∞ | Scale-independent |
| **MSLE** | mean((log(y)-log(ŷ))²) | None | No | 0-∞ | Exponential growth |
| **MedAE** | median(\|y - ŷ\|) | Same as y | No | 0-∞ | Very robust |

### Default Strategy:
```
UVEK kombinuj:
1. RMSE ili MAE (primary metric)
2. R² (variance explained)
3. Residual plots (visual check)
4. Business metric (if applicable)

NIKAD ne koristi samo jednu metriku!
```

**Key Takeaway:** Regression metrike mere različite aspekte grešaka. RMSE je default (penalizuje velike greške), MAE je robustan (outliers), R² pokazuje koliko variance je objašnjeno. UVEK analiziraj residuale vizualno - metrike nisu dovoljne! Patterns u residualima otkrivaju probleme koje metrike ne vide. 🎯