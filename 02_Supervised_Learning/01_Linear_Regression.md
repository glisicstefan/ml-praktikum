# Linear Regression

Linear Regression je **najjednostavniji i najčešće korišćeni supervised learning algoritam** za regression probleme. Modeluje **linearnu vezu** između zavisne varijable (target) i jedne ili više nezavisnih varijabli (features).

**Zašto je Linear Regression fundamentalan?**
- **Interpretabilnost** - Najlakši za objašnjenje (svaki koeficijent = uticaj)
- **Baseline model** - Prva stvar koju testirate pre kompleksnijih modela
- **Matematički elegantan** - Closed-form solution (nema iterativno učenje)
- **Brz** - Trenira se instantno čak i na milionima podataka
- **Dobar za linearne odnose** - Kada je relacija zaista linearna, teško ga nadmašiti

**Kada koristiti Linear Regression?**
- ✅ Linearna veza između features i target-a
- ✅ Potrebna interpretabilnost (npr. business stakeholders)
- ✅ Feature engineering može transformisati nelinearne relacije
- ✅ Prediction je prioritet + data je relativno clean

**Kada NE koristiti:**
- ❌ Kompleksne nelinearne veze (koristi tree-based ili polynomial)
- ❌ Visoka multikolinearnost (koeficijenti postaju nestabilni)
- ❌ Outliers dominiraju (Linear Regression je veoma osetljiv!)

---

## Matematička Osnova

### Simple Linear Regression (1 Feature)

**Model:**
```
y = β₀ + β₁x + ε

gde je:
y  = target (zavisna varijabla)
x  = feature (nezavisna varijabla)
β₀ = intercept (vrednost y kada je x = 0)
β₁ = slope (koeficijent, uticaj x na y)
ε  = error term (residual)
```

**Cilj:** Naći β₀ i β₁ tako da **minimizujemo grešku**.

### Multiple Linear Regression (Multiple Features)

**Model:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

ili u matrix formi:
y = Xβ + ε

gde je:
X = [1, x₁, x₂, ..., xₙ]  (design matrix, first column of 1s for intercept)
β = [β₀, β₁, β₂, ..., βₙ]ᵀ (coefficients)
```

---

## Ordinary Least Squares (OLS)

**Cilj:** Minimizovati **Sum of Squared Residuals (SSR)**
```
SSR = Σ(yᵢ - ŷᵢ)²

gde je:
yᵢ  = actual vrednost
ŷᵢ  = predicted vrednost = β₀ + β₁x₁ᵢ + β₂x₂ᵢ + ...
```

### Closed-Form Solution:
```
β = (XᵀX)⁻¹Xᵀy

Ovo je ANALITIČKA formula - nema gradient descent!
Možeš direktno izračunati optimalne koeficijente.
```

**Prednost:** Instant, nema iteracija  
**Mana:** Sporo za MNOGO features (računanje inverzije je O(n³))

---

## Python Implementacija - Simple Linear Regression

### Primer 1: Manual Implementation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature: 0-10
y = 2 + 3 * X.squeeze() + np.random.randn(100) * 2  # y = 2 + 3x + noise

# Vizualizacija
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, edgecolors='k')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data: y = 2 + 3x + noise')
plt.grid(True, alpha=0.3)
plt.show()

print(f"True coefficients: β₀=2, β₁=3")
```

### Primer 2: Manual OLS Calculation
```python
# Dodaj intercept column (column of 1s)
X_with_intercept = np.c_[np.ones((len(X), 1)), X]

# OLS formula: β = (X'X)^(-1) X'y
XtX = X_with_intercept.T @ X_with_intercept
XtX_inv = np.linalg.inv(XtX)
Xty = X_with_intercept.T @ y
beta = XtX_inv @ Xty

print(f"Learned coefficients (manual OLS):")
print(f"β₀ (intercept) = {beta[0]:.3f}")
print(f"β₁ (slope)     = {beta[1]:.3f}")

# Predictions
y_pred = X_with_intercept @ beta

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data', edgecolors='k')
plt.plot(X, y_pred, color='red', linewidth=2, label=f'Fit: y = {beta[0]:.2f} + {beta[1]:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Manual OLS Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Primer 3: sklearn Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Model
model = LinearRegression()
model.fit(X, y)

# Coefficients
print(f"\nsklearn coefficients:")
print(f"β₀ (intercept) = {model.intercept_:.3f}")
print(f"β₁ (slope)     = {model.coef_[0]:.3f}")

# Predictions
y_pred_sklearn = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred_sklearn)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_sklearn)

print(f"\nMetrics:")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# Vizualizacija
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data', edgecolors='k')
plt.plot(X, y_pred_sklearn, color='green', linewidth=2, 
         label=f'sklearn: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'sklearn Linear Regression (R² = {r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Multiple Linear Regression
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data sa 5 features
X_multi, y_multi = make_regression(
    n_samples=200,
    n_features=5,
    n_informative=5,
    noise=10,
    random_state=42
)

# DataFrame za lakše praćenje
feature_names = [f'Feature_{i+1}' for i in range(5)]
df = pd.DataFrame(X_multi, columns=feature_names)
df['Target'] = y_multi

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Coefficients
print(f"\nIntercept: {model_multi.intercept_:.3f}")
print("\nCoefficients:")
for i, coef in enumerate(model_multi.coef_):
    print(f"  {feature_names[i]}: {coef:.3f}")

# Predictions
y_train_pred = model_multi.predict(X_train)
y_test_pred = model_multi.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nTrain R²:  {train_r2:.3f}")
print(f"Test R²:   {test_r2:.3f}")
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE:  {test_rmse:.3f}")
```

---

## Linear Regression Assumptions

Linear Regression ima **4 ključne pretpostavke** koje MORAJU biti zadovoljene za pouzdane rezultate:

### 1. Linearity (Linearnost)

**Šta znači:** Veza između features i target-a je **linearna**.

**Kako proveriti:** Residual plot (residuals vs fitted values)
```python
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

# Residuals
residuals = y_train - y_train_pred

# Residual plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values (ŷ)')
plt.ylabel('Residuals (y - ŷ)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Ako je linear relationship, residuals bi trebali biti RANDOMLY scattered oko 0
# Ako vidiš pattern (U-shape, curve) → NON-LINEAR relationship!

plt.subplot(1, 2, 2)
plt.scatter(y_train, y_train_pred, alpha=0.6, edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Values (y)')
plt.ylabel('Predicted Values (ŷ)')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Šta ako je narušena:**
- ❌ Koeficijenti su **biased** (pogrešni)
- ❌ Predictions su **neprecizne**
- ✅ **Rešenje:** Polynomial features, feature transformations (log, sqrt), koristi nelinearne modele

---

### 2. Homoscedasticity (Homogenost Varijanse)

**Šta znači:** Varijansa residuals je **konstantna** za sve vrednosti fitted values.

**Kako proveriti:** Residual plot (residuals bi trebali imati istu "širinu" svuda)
```python
# Generiši data SA heteroscedasticity (lošu)
np.random.seed(42)
X_hetero = np.random.rand(200, 1) * 10
noise = np.random.randn(200) * X_hetero.squeeze()  # Noise raste sa X!
y_hetero = 2 + 3 * X_hetero.squeeze() + noise

# Fit model
model_hetero = LinearRegression()
model_hetero.fit(X_hetero, y_hetero)
y_hetero_pred = model_hetero.predict(X_hetero)
residuals_hetero = y_hetero - y_hetero_pred

# Plot
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_hetero_pred, residuals_hetero, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Heteroscedasticity (LOŠE!) 🚨\nVarijansa raste → oblik "funnel"')
plt.grid(True, alpha=0.3)

# Generiši data SA homoscedasticity (dobru)
X_homo = np.random.rand(200, 1) * 10
noise_homo = np.random.randn(200) * 2  # Konstantan noise!
y_homo = 2 + 3 * X_homo.squeeze() + noise_homo

model_homo = LinearRegression()
model_homo.fit(X_homo, y_homo)
y_homo_pred = model_homo.predict(X_homo)
residuals_homo = y_homo - y_homo_pred

plt.subplot(1, 2, 2)
plt.scatter(y_homo_pred, residuals_homo, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity (DOBRO!) ✅\nVarijansa konstantna')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Šta ako je narušena:**
- ❌ **Standard errors** koeficijenata su pogrešni
- ❌ **Confidence intervals** su nepouzdani
- ❌ **Hypothesis tests** (p-values) su nevažeći
- ✅ **Rešenje:** Log transformation target-a, Weighted Least Squares, Robust regression

---

### 3. Normality of Residuals (Normalnost Grešaka)

**Šta znači:** Residuals prate **normalnu distribuciju** (Gaussian).

**Kako proveriti:** Q-Q plot, Histogram, Shapiro-Wilk test
```python
from scipy import stats

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
residuals = y_train - y_train_pred

# Vizualizacije
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Histogram
axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of Residuals')
axes[0].grid(True, alpha=0.3)

# 2. Q-Q Plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')
axes[1].grid(True, alpha=0.3)

# 3. KDE Plot
from scipy.stats import norm
axes[2].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black', label='Residuals')
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[2].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
axes[2].set_xlabel('Residuals')
axes[2].set_ylabel('Density')
axes[2].set_title('Residuals vs Normal Distribution')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Shapiro-Wilk Test (statistical test)
statistic, p_value = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value:   {p_value:.4f}")

if p_value > 0.05:
    print("✅ Residuals are normally distributed (p > 0.05)")
else:
    print("❌ Residuals are NOT normally distributed (p < 0.05)")
```

**Šta ako je narušena:**
- ❌ **Confidence intervals** su nepouzdani
- ❌ **P-values** su pogrešni
- ⚠️ **Ali** predictions su i dalje OK!
- ✅ **Rešenje:** Log/Box-Cox transformation, koristiti robust regression, veći sample (Central Limit Theorem)

---

### 4. Independence of Residuals (Nezavisnost Grešaka)

**Šta znači:** Residuals su **nezavisni** jedni od drugih (nema autocorrelation).

**Najčešći problem:** Time series data (današnji residual utiče na sutrašnji)

**Kako proveriti:** Durbin-Watson test, ACF plot
```python
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson test
dw = durbin_watson(residuals)

print(f"Durbin-Watson statistic: {dw:.3f}")
print("\nInterpretacija:")
print("  0-1:   Pozitivna autocorrelation (loše!)")
print("  1-2:   Nema autocorrelation (dobro!)")
print("  2-3:   Negativna autocorrelation (loše!)")
print("  3-4:   Jaka negativna autocorrelation (loše!)")

if 1.5 <= dw <= 2.5:
    print("\n✅ Residuals are independent")
else:
    print("\n❌ Autocorrelation detected!")

# ACF Plot (Autocorrelation Function)
from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(10, 6))
plot_acf(residuals, lags=20, alpha=0.05)
plt.title('Autocorrelation Function (ACF) of Residuals')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True, alpha=0.3)
plt.show()
```

**Šta ako je narušena:**
- ❌ **Standard errors** su underestimated
- ❌ **Confidence intervals** su previše uski
- ❌ **P-values** su misleading
- ✅ **Rešenje:** Time series models (ARIMA, SARIMA), dodaj lag features

---

## Multicollinearity (Multikolinearnost)

**Šta je to?** Visoka **korelacija između features** (nezavisne varijable zavise jedna od druge).

**Problem:**
- Koeficijenti postaju **nestabilni** (mali change u data → veliki change u koeficijentima)
- **Standard errors** rastu
- **Interpretation** postaje nemoguća (koji feature zaista utiče?)
- **Model performance** ne mora biti loš, ali **interpretacija** jeste!

### Provera Multicollinearity: Correlation Matrix
```python
# Generate multicollinear data
np.random.seed(42)
X1 = np.random.randn(100)
X2 = X1 + np.random.randn(100) * 0.1  # X2 je skoro isti kao X1!
X3 = np.random.randn(100)
y_multi = 2 + 3*X1 + 5*X3 + np.random.randn(100)

df_multi = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y_multi
})

# Correlation matrix
corr_matrix = df_multi.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix\n(X1 i X2 su visoko korelisani!)')
plt.show()

print("Correlation Matrix:")
print(corr_matrix)
```

### Provera Multicollinearity: VIF (Variance Inflation Factor)

**VIF Formula:**
```
VIF_i = 1 / (1 - R²_i)

gde je R²_i = R² kada regresujemo feature i na sve ostale features
```

**Interpretacija:**
```
VIF = 1:      Nema korelacije
VIF = 1-5:    Umerena korelacija (OK)
VIF = 5-10:   Visoka korelacija (⚠️ pazi!)
VIF > 10:     EKSTREMNA korelacija (🚨 problem!)
```
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF
X_vif = df_multi[['X1', 'X2', 'X3']].values
vif_data = pd.DataFrame()
vif_data["Feature"] = ['X1', 'X2', 'X3']
vif_data["VIF"] = [variance_inflation_factor(X_vif, i) for i in range(3)]

print("\nVariance Inflation Factor (VIF):")
print(vif_data)
print("\nInterpretacija:")
for idx, row in vif_data.iterrows():
    if row['VIF'] > 10:
        print(f"🚨 {row['Feature']}: VIF = {row['VIF']:.2f} → EKSTREMNA multicollinearity!")
    elif row['VIF'] > 5:
        print(f"⚠️ {row['Feature']}: VIF = {row['VIF']:.2f} → Visoka multicollinearity")
    else:
        print(f"✅ {row['Feature']}: VIF = {row['VIF']:.2f} → OK")
```

### Rešavanje Multicollinearity:
```python
# Opcija 1: Ukloni jedan od korelisanih features
X_fixed = df_multi[['X1', 'X3']]  # Uklonili smo X2
y_fixed = df_multi['y']

model_fixed = LinearRegression()
model_fixed.fit(X_fixed, y_fixed)

print("\nModel BEZ multicollinearity:")
print(f"Coefficients: {model_fixed.coef_}")
print(f"Intercept: {model_fixed.intercept_:.3f}")

# Opcija 2: PCA (Principal Component Analysis)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_multi[['X1', 'X2', 'X3']])

model_pca = LinearRegression()
model_pca.fit(X_pca, y_fixed)
print(f"\nModel sa PCA:")
print(f"Coefficients: {model_pca.coef_}")

# Opcija 3: Regularization (Ridge/Lasso)
# Vidi folder: 05_Model_Evaluation_and_Tuning/06_Regularization.md
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(df_multi[['X1', 'X2', 'X3']], y_fixed)
print(f"\nRidge Regression:")
print(f"Coefficients: {ridge.coef_}")
```

---

## Outliers i Influence

**Outliers** mogu **drastično** uticati na Linear Regression!

### Vizualizacija Uticaja Outliers
```python
# Data bez outliers
np.random.seed(42)
X_clean = np.random.rand(50, 1) * 10
y_clean = 2 + 3 * X_clean.squeeze() + np.random.randn(50)

# Dodaj outlier
X_outlier = np.vstack([X_clean, [[9]]])
y_outlier = np.append(y_clean, [50])  # Ekstremna vrednost!

# Fit models
model_clean = LinearRegression().fit(X_clean, y_clean)
model_outlier = LinearRegression().fit(X_outlier, y_outlier)

# Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_clean, y_clean, alpha=0.6, edgecolors='k', label='Clean Data')
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line_clean = model_clean.predict(x_line)
plt.plot(x_line, y_line_clean, 'g-', linewidth=2, label='Fit (Clean)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression - Clean Data ✅')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_clean, y_clean, alpha=0.6, edgecolors='k', label='Clean Data')
plt.scatter([9], [50], color='red', s=200, marker='X', edgecolors='black', 
            linewidths=2, label='Outlier', zorder=5)
y_line_outlier = model_outlier.predict(x_line)
plt.plot(x_line, y_line_outlier, 'r-', linewidth=2, label='Fit (With Outlier)')
plt.plot(x_line, y_line_clean, 'g--', linewidth=1, alpha=0.5, label='Fit (Clean)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression - With Outlier 🚨\nLinija se DRASTIČNO promenila!')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Clean model:   β₁ = {model_clean.coef_[0]:.3f}")
print(f"Outlier model: β₁ = {model_outlier.coef_[0]:.3f}")
print(f"Difference:    Δβ₁ = {abs(model_outlier.coef_[0] - model_clean.coef_[0]):.3f}")
```

### Detekcija Outliers: Leverage & Cook's Distance
```python
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm

# Fit model sa statsmodels (za diagnostic tools)
X_sm = sm.add_constant(X_outlier)  # Dodaj intercept
model_sm = sm.OLS(y_outlier, X_sm).fit()

# Influence measures
influence = OLSInfluence(model_sm)

# Leverage (hat values)
leverage = influence.hat_matrix_diag

# Cook's distance
cooks_d = influence.cooks_distance[0]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Leverage plot
axes[0].stem(range(len(leverage)), leverage, basefmt=" ")
axes[0].axhline(y=2*len(X_sm[0])/len(X_sm), color='r', linestyle='--', 
                label='Threshold (2p/n)')
axes[0].set_xlabel('Observation Index')
axes[0].set_ylabel('Leverage')
axes[0].set_title('Leverage (Hat Values)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cook's distance
axes[1].stem(range(len(cooks_d)), cooks_d, basefmt=" ")
axes[1].axhline(y=4/len(X_sm), color='r', linestyle='--', label='Threshold (4/n)')
axes[1].set_xlabel('Observation Index')
axes[1].set_ylabel("Cook's Distance")
axes[1].set_title("Cook's Distance (Influence)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identificiraj outliers
print("Outliers detected:")
threshold_cooks = 4 / len(X_sm)
outlier_indices = np.where(cooks_d > threshold_cooks)[0]
print(f"Indices: {outlier_indices}")
print(f"Cook's Distance values: {cooks_d[outlier_indices]}")
```

**Rešavanje Outliers:**
- ✅ Ukloni ih (ako su greška u merenju)
- ✅ Transformuj target (log, sqrt)
- ✅ Koristi **Robust Regression** (RANSAC, Huber)
- ✅ Koristi tree-based modele (manje osetljivi)

---

## Polynomial Regression (Feature Engineering)

**Problem:** Linearni model ne može da uhvati **nelinearne veze**.

**Rešenje:** Transformiši features u **polinomske** verzije!
```python
from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear data
np.random.seed(42)
X_poly = np.random.rand(100, 1) * 10
y_poly = 2 + 3*X_poly.squeeze() + 0.5*X_poly.squeeze()**2 + np.random.randn(100)*5

# 1. Linear model (loš!)
model_linear = LinearRegression()
model_linear.fit(X_poly, y_poly)
y_pred_linear = model_linear.predict(X_poly)

# 2. Polynomial model (degree=2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_transformed = poly_features.fit_transform(X_poly)

print("Original features:", X_poly.shape)
print("Polynomial features:", X_poly_transformed.shape)
print("Feature names:", poly_features.get_feature_names_out(['X']))
# ['X', 'X^2']

model_poly = LinearRegression()
model_poly.fit(X_poly_transformed, y_poly)
y_pred_poly = model_poly.predict(X_poly_transformed)

# Metrics
from sklearn.metrics import r2_score

r2_linear = r2_score(y_poly, y_pred_linear)
r2_poly = r2_score(y_poly, y_pred_poly)

print(f"\nLinear Model R²:     {r2_linear:.3f}")
print(f"Polynomial Model R²: {r2_poly:.3f}")
print(f"Improvement: +{(r2_poly - r2_linear):.3f}")

# Visualization
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot_linear = model_linear.predict(X_plot)
y_plot_poly = model_poly.predict(X_plot_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X_poly, y_poly, alpha=0.6, edgecolors='k', label='Data')
plt.plot(X_plot, y_plot_linear, 'r--', linewidth=2, label=f'Linear (R²={r2_linear:.2f})')
plt.plot(X_plot, y_plot_poly, 'g-', linewidth=2, label=f'Polynomial (R²={r2_poly:.2f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Choosing Polynomial Degree
```python
from sklearn.model_selection import train_test_split

# Split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

# Test different degrees
degrees = range(1, 10)
train_scores = []
test_scores = []

for degree in degrees:
    # Transform
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_p)
    X_test_poly = poly.transform(X_test_p)
    
    # Train
    model = LinearRegression()
    model.fit(X_train_poly, y_train_p)
    
    # Score
    train_score = model.score(X_train_poly, y_train_p)
    test_score = model.score(X_test_poly, y_test_p)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Degree {degree}: Train R²={train_score:.3f}, Test R²={test_score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'o-', label='Train R²', linewidth=2)
plt.plot(degrees, test_scores, 's-', label='Test R²', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Performance vs Polynomial Degree\n(Test R² pada nakon degree=2 → Overfitting!)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(degrees)
plt.show()

best_degree = degrees[np.argmax(test_scores)]
print(f"\n🏆 Best degree: {best_degree} (Test R² = {max(test_scores):.3f})")
```

**Upozorenje:**
- ⚠️ Visoki degree (npr. 8, 9, 10) → **Overfitting**!
- ⚠️ Broj features eksplozivno raste (npr. 10 features, degree=3 → 220 features!)
- ✅ Koristi **Regularization** (Ridge/Lasso) za visoke degree

---

## Regularization za Linear Regression

**Problem:** Obični Linear Regression može overfitovati sa mnogo features.

**Rešenje:** Dodaj **penalty** na magnitude koeficijenata.
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Generate data sa mnogo features (overfitting scenario)
X_reg, y_reg = make_regression(
    n_samples=100,
    n_features=50,
    n_informative=10,
    noise=10,
    random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

results = []

for name, model in models.items():
    model.fit(X_train_r, y_train_r)
    train_score = model.score(X_train_r, y_train_r)
    test_score = model.score(X_test_r, y_test_r)
    
    # Broj non-zero coefficients
    non_zero_coefs = np.sum(np.abs(model.coef_) > 1e-5)
    
    results.append({
        'Model': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'Non-Zero Coefs': non_zero_coefs
    })
    
    print(f"{name}:")
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²:  {test_score:.3f}")
    print(f"  Non-Zero Coefficients: {non_zero_coefs} / {len(model.coef_)}")
    print()

results_df = pd.DataFrame(results)
print(results_df)
```

**Poređenje:**
- **Linear Regression**: Može overfitovati (train R² > test R²)
- **Ridge (L2)**: Smanjuje magnitude svih koeficijenata (ali ih ne svodi na 0)
- **Lasso (L1)**: Svodi neke koeficijente na **0** (feature selection!)
- **ElasticNet**: Kombinacija L1 + L2

**Za detalje o Regularization, vidi:** `05_Model_Evaluation_and_Tuning/06_Regularization.md`

---

## Statistical Analysis sa statsmodels

`sklearn` je odličan za **predictions**, ali ne daje **statistical inference** (p-values, confidence intervals).

Za to koristimo **statsmodels**!
```python
import statsmodels.api as sm

# Generate data
np.random.seed(42)
X_stats = np.random.rand(100, 3) * 10
y_stats = 2 + 3*X_stats[:, 0] + 5*X_stats[:, 1] + 1*X_stats[:, 2] + np.random.randn(100)*2

# Add intercept (statsmodels ne dodaje automatski!)
X_stats_with_const = sm.add_constant(X_stats)

# Fit model
model_stats = sm.OLS(y_stats, X_stats_with_const).fit()

# Summary (pun statistical output!)
print(model_stats.summary())
```

**Output interpretacija:**
```
==============================================================================
Dep. Variable:                      y   R-squared:                       0.985
Model:                            OLS   Adj. R-squared:                  0.984
Method:                 Least Squares   F-statistic:                     2087.
No. Observations:                 100   Prob (F-statistic):           4.29e-85
Df Residuals:                      96   
Df Model:                           3
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.1234      0.205     10.345      0.000       1.716       2.531
x1             2.9876      0.019    157.240      0.000       2.950       3.025
x2             5.0123      0.019    263.842      0.000       4.975       5.050
x3             0.9987      0.019     52.563      0.000       0.961       1.036
==============================================================================
```

**Ključni Elementi:**

1. **R-squared (R²)**: 0.985 → Model objašnjava 98.5% varijanse
2. **Adj. R-squared**: Adjusted za broj features
3. **F-statistic**: Test da li je model statistički značajan (p < 0.05 → jeste!)
4. **Coefficients (coef)**: β₀=2.12, β₁=2.99, β₂=5.01, β₃=1.00
5. **P>|t| (p-value)**: Ako < 0.05 → koeficijent je **statistički značajan**
6. **[0.025, 0.975]**: 95% confidence interval za koeficijent

### Hypothesis Testing
```python
# Test: Da li je β₁ = 0? (Da li X1 utiče na y?)
# H₀: β₁ = 0 (nema uticaja)
# H₁: β₁ ≠ 0 (ima uticaja)

p_value_x1 = model_stats.pvalues[1]  # Index 1 = x1 (0 je intercept)

print(f"\nHypothesis Test za X1:")
print(f"P-value: {p_value_x1:.4f}")

if p_value_x1 < 0.05:
    print("✅ Reject H₀: X1 ima STATISTIČKI ZNAČAJAN uticaj na y (p < 0.05)")
else:
    print("❌ Fail to reject H₀: X1 NEMA statistički značajan uticaj (p > 0.05)")

# Confidence intervals
conf_int = model_stats.conf_int(alpha=0.05)
print("\n95% Confidence Intervals:")
print(conf_int)
```

### Predictions sa Confidence Intervals
```python
# Predict
X_new = np.array([[1, 5, 7, 2]])  # 1 = const, 5, 7, 2 = x1, x2, x3
predictions = model_stats.get_prediction(X_new)

# Summary
pred_summary = predictions.summary_frame(alpha=0.05)
print("\nPrediction Summary:")
print(pred_summary)
print(f"\nPredicted value: {pred_summary['mean'].values[0]:.2f}")
print(f"95% CI: [{pred_summary['mean_ci_lower'].values[0]:.2f}, "
      f"{pred_summary['mean_ci_upper'].values[0]:.2f}]")
```

---

## Complete End-to-End Example: House Price Prediction
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION")
print("="*60)

# ==================== 1. LOAD DATA ====================
data = fetch_california_housing()
X_house = data.data
y_house = data.target
feature_names = data.feature_names

print(f"\nDataset shape: {X_house.shape}")
print(f"Features: {feature_names}")
print(f"Target: Median house value (in $100k)")

# DataFrame
df_house = pd.DataFrame(X_house, columns=feature_names)
df_house['MedianHouseValue'] = y_house

print("\nFirst 5 rows:")
print(df_house.head())

print("\nDescriptive statistics:")
print(df_house.describe())

# ==================== 2. EDA ====================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix_house = df_house.corr()
sns.heatmap(corr_matrix_house, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Najvažnije korelacije sa targetom
target_corr = corr_matrix_house['MedianHouseValue'].sort_values(ascending=False)
print("\nCorrelation sa Target (MedianHouseValue):")
print(target_corr)

# Scatter plots - top 3 features
top_features = target_corr.index[1:4]  # Skip target itself

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, feature in enumerate(top_features):
    axes[idx].scatter(df_house[feature], df_house['MedianHouseValue'], 
                      alpha=0.3, edgecolors='k', linewidths=0.5)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('MedianHouseValue')
    axes[idx].set_title(f'{feature} vs MedianHouseValue\nCorr={target_corr[feature]:.3f}')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 3. TRAIN-TEST SPLIT ====================
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train_h.shape}")
print(f"Test set:  {X_test_h.shape}")

# ==================== 4. FEATURE SCALING ====================
# Linear Regression ne zahteva scaling, ali ga radimo za:
# - Multicollinearity check (VIF)
# - Coefficient interpretation
# Vidi folder: 01_Data_Preprocessing/06_Feature_Scaling.md

scaler = StandardScaler()
X_train_h_scaled = scaler.fit_transform(X_train_h)
X_test_h_scaled = scaler.transform(X_test_h)

print("\n✅ Features scaled (StandardScaler)")

# ==================== 5. BASELINE MODEL ====================
print("\n" + "="*60)
print("BASELINE MODEL - MEAN PREDICTION")
print("="*60)

# Predviđaj mean vrednost za sve
y_baseline_pred = np.full(len(y_test_h), y_train_h.mean())

baseline_mae = mean_absolute_error(y_test_h, y_baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test_h, y_baseline_pred))
baseline_r2 = r2_score(y_test_h, y_baseline_pred)

print(f"\nBaseline (Mean Prediction):")
print(f"  MAE:  {baseline_mae:.3f}")
print(f"  RMSE: {baseline_rmse:.3f}")
print(f"  R²:   {baseline_r2:.3f}")

# ==================== 6. LINEAR REGRESSION MODEL ====================
print("\n" + "="*60)
print("LINEAR REGRESSION MODEL")
print("="*60)

# Train
model_house = LinearRegression()
model_house.fit(X_train_h_scaled, y_train_h)

# Predictions
y_train_pred_h = model_house.predict(X_train_h_scaled)
y_test_pred_h = model_house.predict(X_test_h_scaled)

# Metrics - Train
train_mae = mean_absolute_error(y_train_h, y_train_pred_h)
train_rmse = np.sqrt(mean_squared_error(y_train_h, y_train_pred_h))
train_r2 = r2_score(y_train_h, y_train_pred_h)

# Metrics - Test
test_mae = mean_absolute_error(y_test_h, y_test_pred_h)
test_rmse = np.sqrt(mean_squared_error(y_test_h, y_test_pred_h))
test_r2 = r2_score(y_test_h, y_test_pred_h)

print("\nTrain Performance:")
print(f"  MAE:  {train_mae:.3f}")
print(f"  RMSE: {train_rmse:.3f}")
print(f"  R²:   {train_r2:.3f}")

print("\nTest Performance:")
print(f"  MAE:  {test_mae:.3f}")
print(f"  RMSE: {test_rmse:.3f}")
print(f"  R²:   {test_r2:.3f}")

print(f"\nImprovement over Baseline:")
print(f"  MAE:  {baseline_mae - test_mae:+.3f} ({(baseline_mae - test_mae)/baseline_mae*100:.1f}%)")
print(f"  RMSE: {baseline_rmse - test_rmse:+.3f} ({(baseline_rmse - test_rmse)/baseline_rmse*100:.1f}%)")
print(f"  R²:   {test_r2 - baseline_r2:+.3f}")

# ==================== 7. COEFFICIENT INTERPRETATION ====================
print("\n" + "="*60)
print("COEFFICIENT INTERPRETATION")
print("="*60)

# Create DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model_house.coef_
}).sort_values('Coefficient', ascending=False)

print(f"\nIntercept: {model_house.intercept_:.3f}")
print("\nCoefficients:")
print(coef_df.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Linear Regression Coefficients)\nGreen = Positive impact, Red = Negative impact')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nInterpretacija:")
print("  - Pozitivan koeficijent → Porast feature → Porast cene")
print("  - Negativan koeficijent → Porast feature → Pad cene")

# ==================== 8. MULTICOLLINEARITY CHECK (VIF) ====================
print("\n" + "="*60)
print("MULTICOLLINEARITY CHECK")
print("="*60)

vif_data_house = pd.DataFrame()
vif_data_house["Feature"] = feature_names
vif_data_house["VIF"] = [variance_inflation_factor(X_train_h_scaled, i) 
                          for i in range(len(feature_names))]

print("\nVariance Inflation Factor:")
print(vif_data_house.to_string(index=False))

print("\nInterpretacija:")
for idx, row in vif_data_house.iterrows():
    if row['VIF'] > 10:
        print(f"🚨 {row['Feature']}: VIF = {row['VIF']:.2f} → VISOKA multicollinearity!")
    elif row['VIF'] > 5:
        print(f"⚠️ {row['Feature']}: VIF = {row['VIF']:.2f} → Umerena multicollinearity")
    else:
        print(f"✅ {row['Feature']}: VIF = {row['VIF']:.2f} → OK")

# ==================== 9. RESIDUAL ANALYSIS ====================
print("\n" + "="*60)
print("RESIDUAL ANALYSIS")
print("="*60)

# Residuals
residuals_h = y_test_h - y_test_pred_h

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals vs Fitted
axes[0, 0].scatter(y_test_pred_h, residuals_h, alpha=0.5, edgecolors='k', linewidths=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values (ŷ)')
axes[0, 0].set_ylabel('Residuals (y - ŷ)')
axes[0, 0].set_title('Residuals vs Fitted Values\n(Trebaju biti random oko 0)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram of Residuals
axes[0, 1].hist(residuals_h, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Histogram of Residuals\n(Trebaju biti približno normalni)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q Plot
stats.probplot(residuals_h, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot\n(Tačke treba da budu blizu linije)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Actual vs Predicted
axes[1, 1].scatter(y_test_h, y_test_pred_h, alpha=0.5, edgecolors='k', linewidths=0.5)
axes[1, 1].plot([y_test_h.min(), y_test_h.max()], 
                [y_test_h.min(), y_test_h.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Shapiro-Wilk test
stat_shapiro, p_shapiro = stats.shapiro(residuals_h[:5000])  # Shapiro max 5000 samples
print(f"\nShapiro-Wilk Test (Normality):")
print(f"  Statistic: {stat_shapiro:.4f}")
print(f"  P-value:   {p_shapiro:.4f}")
if p_shapiro > 0.05:
    print("  ✅ Residuals are approximately normal (p > 0.05)")
else:
    print("  ❌ Residuals are NOT normal (p < 0.05)")

# Durbin-Watson (Independence)
dw_house = durbin_watson(residuals_h)
print(f"\nDurbin-Watson Test (Independence):")
print(f"  Statistic: {dw_house:.3f}")
if 1.5 <= dw_house <= 2.5:
    print("  ✅ No significant autocorrelation")
else:
    print("  ⚠️ Potential autocorrelation detected")

# ==================== 10. FEATURE ENGINEERING - POLYNOMIAL ====================
print("\n" + "="*60)
print("FEATURE ENGINEERING - POLYNOMIAL FEATURES")
print("="*60)

# Test polynomial degree=2
poly_house = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_h = poly_house.fit_transform(X_train_h_scaled)
X_test_poly_h = poly_house.transform(X_test_h_scaled)

print(f"\nOriginal features: {X_train_h_scaled.shape[1]}")
print(f"Polynomial features (degree=2): {X_train_poly_h.shape[1]}")

# Train polynomial model
model_poly_house = LinearRegression()
model_poly_house.fit(X_train_poly_h, y_train_h)

# Predictions
y_train_pred_poly = model_poly_house.predict(X_train_poly_h)
y_test_pred_poly = model_poly_house.predict(X_test_poly_h)

# Metrics
train_r2_poly = r2_score(y_train_h, y_train_pred_poly)
test_r2_poly = r2_score(y_test_h, y_test_pred_poly)
test_rmse_poly = np.sqrt(mean_squared_error(y_test_h, y_test_pred_poly))

print(f"\nPolynomial Model Performance:")
print(f"  Train R²: {train_r2_poly:.3f}")
print(f"  Test R²:  {test_r2_poly:.3f}")
print(f"  Test RMSE: {test_rmse_poly:.3f}")

print(f"\nComparison:")
print(f"  Linear Model Test R²:     {test_r2:.3f}")
print(f"  Polynomial Model Test R²: {test_r2_poly:.3f}")
print(f"  Improvement: {test_r2_poly - test_r2:+.3f}")

if train_r2_poly - test_r2_poly > 0.1:
    print("  ⚠️ Overfitting detected! (Train R² >> Test R²)")
else:
    print("  ✅ No significant overfitting")

# ==================== 11. REGULARIZATION ====================
print("\n" + "="*60)
print("REGULARIZATION (Ridge)")
print("="*60)

# Ridge regression (combat overfitting from polynomial features)
ridge_house = Ridge(alpha=1.0)
ridge_house.fit(X_train_poly_h, y_train_h)

y_test_pred_ridge = ridge_house.predict(X_test_poly_h)
test_r2_ridge = r2_score(y_test_h, y_test_pred_ridge)
test_rmse_ridge = np.sqrt(mean_squared_error(y_test_h, y_test_pred_ridge))

print(f"\nRidge Model (alpha=1.0) Performance:")
print(f"  Test R²:  {test_r2_ridge:.3f}")
print(f"  Test RMSE: {test_rmse_ridge:.3f}")

# ==================== 12. FINAL COMPARISON ====================
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['Baseline (Mean)', 'Linear Regression', 'Polynomial (deg=2)', 'Ridge (Poly)'],
    'Test R²': [baseline_r2, test_r2, test_r2_poly, test_r2_ridge],
    'Test RMSE': [baseline_rmse, test_rmse, test_rmse_poly, test_rmse_ridge]
})

print("\n" + comparison_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
axes[0].bar(comparison_df['Model'], comparison_df['Test R²'], 
            color=['gray', 'blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Comparison - R² (Higher is Better)')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=15)

# RMSE comparison
axes[1].bar(comparison_df['Model'], comparison_df['Test RMSE'], 
            color=['gray', 'blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Model Comparison - RMSE (Lower is Better)')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# Best model
best_idx = comparison_df['Test R²'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']
best_r2 = comparison_df.loc[best_idx, 'Test R²']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {best_r2:.3f}")

# ==================== 13. SAVE MODEL ====================
import joblib

# Save best model (pretpostavimo da je Linear Regression najbolji za deployment)
joblib.dump(model_house, 'linear_regression_house_price.pkl')
joblib.dump(scaler, 'scaler_house_price.pkl')

print("\n✅ Model saved: linear_regression_house_price.pkl")
print("✅ Scaler saved: scaler_house_price.pkl")

# ==================== 14. PREDICTIONS NA NOVIM PODACIMA ====================
print("\n" + "="*60)
print("PREDICTIONS ON NEW DATA")
print("="*60)

# Simulate new house
new_house = np.array([[
    2.5,    # MedInc
    35.0,   # HouseAge
    5.0,    # AveRooms
    1.2,    # AveBedrms
    1000,   # Population
    3.5,    # AveOccup
    35.5,   # Latitude
    -120.0  # Longitude
]])

print("\nNew House Features:")
for i, feature in enumerate(feature_names):
    print(f"  {feature}: {new_house[0, i]}")

# Scale
new_house_scaled = scaler.transform(new_house)

# Predict
predicted_price = model_house.predict(new_house_scaled)[0]

print(f"\n🏠 Predicted Median House Value: ${predicted_price*100:.2f}k")
print(f"   (= ${predicted_price*100*1000:.0f})")

print("\n" + "="*60)
print("LINEAR REGRESSION ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Best Practices

### ✅ DO:

1. **Proveri assumptions** - Linearity, Homoscedasticity, Normality, Independence
2. **Check multicollinearity** - VIF < 10
3. **Residual analysis** - Uvek pregledaj residual plots
4. **Feature scaling** - Za coefficient interpretation i VIF
5. **Outliers** - Identifikuj i odluči šta sa njima (ukloni / transformiši / robust regression)
6. **Baseline model** - Uporedi sa mean/median prediction
7. **Train-test split** - Nikad ne evaluiraj na train setu
8. **Feature engineering** - Polynomial features za nelinearne veze
9. **Regularization** - Ako imaš mnogo features ili overfitting
10. **Interpretation** - Koristi statsmodels za p-values i confidence intervals

### ❌ DON'T:

1. **Ne koristi Linear Regression za kompleksne nelinearne veze** - Koristi tree-based ili NN
2. **Ne ignoriši outliers** - Drastično utiču na koeficijente
3. **Ne zanemari multicollinearity** - Koeficijenti postaju nestabilni
4. **Ne koristi visok polynomial degree bez regularization** - Overfitting!
5. **Ne tumači koeficijente ako features nisu skalirani** - Magnitude zavisi od scale
6. **Ne dodavaj sve features nasumično** - Feature selection je bitan
7. **Ne koristi R² kao jedinu metriku** - Kombinuj sa RMSE/MAE i residual analysis

---

## Common Pitfalls (Česte Greške)

### Greška 1: Interpretiranje Koeficijenata Bez Scaling
```python
# ❌ LOŠE - Coefficients su u različitim scales
X = np.array([[1, 1000], [2, 2000], [3, 3000]])  # Feature 1: 1-3, Feature 2: 1000-3000
y = np.array([10, 20, 30])

model = LinearRegression().fit(X, y)
print(f"Coefficients: {model.coef_}")
# Output: [0.001, 0.009] - Izgleda da je Feature 2 važniji? NE!

# ✅ DOBRO - Scale prvo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = LinearRegression().fit(X_scaled, y)
print(f"Coefficients (scaled): {model_scaled.coef_}")
# Sada možeš uporediti magnitude!
```

### Greška 2: Ignoring Outliers
```python
# ❌ LOŠE - Outlier drastično utiče
X_with_outlier = np.append(X, [[100]], axis=0)
y_with_outlier = np.append(y, [1000])

model_bad = LinearRegression().fit(X_with_outlier, y_with_outlier)
# Model je sjeban!

# ✅ DOBRO - Ukloni outlier ili koristi robust regression
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor()
huber.fit(X_with_outlier, y_with_outlier)
# Mnogo bolje!
```

### Greška 3: Polynomial Overfitting
```python
# ❌ LOŠE - Degree=10 bez regularization
poly_high = PolynomialFeatures(degree=10)
X_poly_high = poly_high.fit_transform(X)
model_overfit = LinearRegression().fit(X_poly_high, y)
# Overfit na train, loš na test!

# ✅ DOBRO - Koristi regularization ili niži degree
ridge_poly = Ridge(alpha=10.0)
ridge_poly.fit(X_poly_high, y)
# Mnogo bolje!
```

### Greška 4: Ne Proveravaju Assumptions
```python
# ❌ LOŠE - Fit model i gotovo
model.fit(X_train, y_train)
print(f"R² = {model.score(X_test, y_test)}")
# Ne znaš da li su assumptions zadovoljene!

# ✅ DOBRO - Proveri residual plots, VIF, Shapiro-Wilk
residuals = y_test - model.predict(X_test)
# ... residual analysis kao gore
```

---

## Kada Koristiti Linear Regression?

### ✅ Idealno Za:
```
Problem Type:
├─ Regression (continuous target)
├─ Linearna ili skoro-linearna veza
├─ Feature engineering može transformisati nelinearne veze
└─ Interpretability je kritična

Data Characteristics:
├─ Clean data (malo outliers)
├─ Niska do umerena multicollinearity
├─ Features nisu suviše korelisani
├─ Normalna ili skoro-normalna distribucija residuals
└─ Homoscedasticity (konstantna variance)

Business Context:
├─ Potrebna interpretabilnost za stakeholders
├─ Regulatorni zahtevi (explainability)
├─ Baseline model pre kompleksnijih
└─ Brz trening i deployment
```

### ❌ Izbegavaj Za:
```
- Ekstremno nelinearne veze (koristi Random Forest, XGBoost, Neural Networks)
- Mnogo outliers (koristi Robust Regression ili tree-based)
- Visoka multicollinearity koju ne možeš rešiti (koristi Regularization ili PCA)
- Kompleksne interakcije između features (koristi tree-based ili polynomial sa regularization)
- Classification problemi (koristi Logistic Regression)
```

---

## Rezime

| Aspekt | Opis |
|--------|------|
| **Tip** | Supervised Learning - Regression |
| **Kompleksnost** | Niska (najjednostavniji model) |
| **Interpretabilnost** | ⭐⭐⭐⭐⭐ Excellent (koeficijenti = uticaj) |
| **Training Speed** | ⭐⭐⭐⭐⭐ Instant (closed-form solution) |
| **Prediction Speed** | ⭐⭐⭐⭐⭐ Instant (matrix multiplication) |
| **Linearity Required** | Da (ili feature engineering) |
| **Handles Non-linearity** | ❌ (osim sa polynomial features) |
| **Handles Outliers** | ❌ Veoma osetljiv |
| **Handles Missing Values** | ❌ Ne (mora se impute) |
| **Feature Scaling** | Opciono (ali preporučeno za interpretation) |
| **Regularization** | Opciono (Ridge/Lasso/ElasticNet) |
| **Overfitting Risk** | Nizak (osim sa mnogo features) |
| **Best For** | Baseline, linearne veze, interpretabilnost |

### Key Hyperparameters:

**LinearRegression (sklearn):**
- `fit_intercept`: True/False (da li fituje β₀)
- `normalize`: Deprecated (koristi StandardScaler umesto)
- `n_jobs`: Broj cores (za parallelization)

**Ridge/Lasso/ElasticNet:**
- `alpha`: Regularization strength (za detalje vidi folder 05/06_Regularization.md)

### Metrics za Evaluaciju:

- **MAE** (Mean Absolute Error) - Prosečna apsolutna greška
- **MSE** (Mean Squared Error) - Kažnjava velike greške više
- **RMSE** (Root MSE) - U istim jedinicama kao target
- **R²** (R-squared) - Procenat varijanse objašnjene modelom
- **Adjusted R²** - R² adjusted za broj features

**Za detalje o metrics, vidi:** `05_Model_Evaluation_and_Tuning/02_Regression_Metrics.md`

---

## Quick Decision Tree
```
Start
  ↓
Regression problem?
  ↓ Yes
Linearna veza između X i y?
  ↓ Yes
Interpretabilnost je važna?
  ↓ Yes
Data je clean (malo outliers)?
  ↓ Yes
→ LINEAR REGRESSION ✅

Ako bilo šta "No":
  ├─ Nelinearna veza? → Polynomial features + Regularization
  ├─ Ne treba interpretacija? → Random Forest / XGBoost
  ├─ Mnogo outliers? → Robust Regression / Tree-based
  └─ Multicollinearity? → Ridge/Lasso ili PCA
```

---

**Key Takeaway:** Linear Regression je **fundamentalan** i **najlakši za razumevanje** supervised learning algoritam. Odličan je za **baseline modele** i situacije gde je **interpretabilnost kritična**. Za kompleksnije nelinearne veze, postoje moćniji algoritmi (tree-based, boosting, neural networks), ali Linear Regression je **uvek dobra startna tačka**! 🎯