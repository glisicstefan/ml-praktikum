# Regularization

Regularization je tehnika **sprečavanja overfitting-a** dodavanjem **penalty-a** na kompleksnost modela. Cilj je da model generalizuje bolje, čak i ako to znači da će biti malo gori na training data.

**Zašto je regularizacija kritična?**
- **Sprečava overfitting** - Model ne memorise training data
- **Poboljšava generalizaciju** - Bolji performance na test/unseen data
- **Feature selection** (L1) - Automatski eliminiše nebitne features
- **Numerička stabilnost** - Rešava multicollinearity probleme
- **Kontroliše kompleksnost** - Jednostavniji, robusniji model

**VAŽNO:** Regularizacija je jedna od najmoćnijih tehnika u ML! Razumevanje L1 vs L2 je esencijalno.

---

## Problem bez Regularizacije

### Demonstracija Overfitting-a:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate data sa šumom
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.randn(100) * 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model BEZ regularizacije - High degree polynomial
model_no_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('linear', LinearRegression())
])

model_no_reg.fit(X_train, y_train)

# Predictions
y_train_pred = model_no_reg.predict(X_train)
y_test_pred = model_no_reg.predict(X_test)

# Errors
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("Model WITHOUT Regularization:")
print(f"Train MSE: {train_mse:.3f}")  # Veoma niska
print(f"Test MSE:  {test_mse:.3f}")   # VISOKA! Overfitting!

# Coefficients magnitude
coefficients = model_no_reg.named_steps['linear'].coef_
print(f"\nCoefficient magnitudes:")
print(f"Max: {np.max(np.abs(coefficients)):.2f}")
print(f"Min: {np.min(np.abs(coefficients)):.2f}")
# HUGE coefficients → Overfitting!

# Visualization
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot = model_no_reg.predict(X_plot)

plt.figure(figsize=(12, 5))

# Left: Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, s=30, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, s=30, label='Test')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model (no reg)')
plt.plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, alpha=0.7, label='True function')
plt.title('Predictions - NO Regularization\n(Overfitting!)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 2)

# Right: Coefficients
plt.subplot(1, 2, 2)
plt.bar(range(len(coefficients)), coefficients)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Model Coefficients\n(HUGE values → Overfitting)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Problem: Model "uči" šum umesto pattern-a!
# Rešenje: REGULARIZACIJA!
```

---

## 1. L2 Regularization (Ridge)

**L2 Regularization** dodaje **kvadrat koeficijenata** u loss funkciju - "shrinkage".

### Matematika:
```
BEZ Regularizacije:
Loss = MSE = Σ(y - ŷ)²

SA L2 Regularizacija (Ridge):
Loss = MSE + α × Σ(w²)
     = Σ(y - ŷ)² + α × ||w||²₂

Gde:
α (alpha) = Regularization strength (hyperparameter)
w = Model coefficients (weights)
||w||²₂ = Sum of squared coefficients (L2 norm)

α = 0     → No regularization (standard linear regression)
α → ∞     → All coefficients → 0 (underfitting)
α optimal → Balance between fit and simplicity
```

### Karakteristike L2:
```
✅ Shrinks coefficients TOWARDS zero (but never exactly 0)
✅ All features retained (no feature selection)
✅ Good for multicollinearity (correlated features)
✅ Smooth, differentiable everywhere
✅ Numerically stable
```

### Python - Ridge Regression:
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Generate data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ridge model (α = 10)
model_ridge = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('scaler', StandardScaler()),  # IMPORTANT for regularization!
    ('ridge', Ridge(alpha=10))
])

model_ridge.fit(X_train, y_train)

# Predictions
y_train_pred_ridge = model_ridge.predict(X_train)
y_test_pred_ridge = model_ridge.predict(X_test)

# Errors
train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)

print("Model WITH Ridge Regularization (α=10):")
print(f"Train MSE: {train_mse_ridge:.3f}")  # Malo veća nego bez reg
print(f"Test MSE:  {test_mse_ridge:.3f}")   # MNOGO bolja! Generalizuje!

# Coefficients - MUCH smaller!
coefficients_ridge = model_ridge.named_steps['ridge'].coef_
print(f"\nCoefficient magnitudes (Ridge):")
print(f"Max: {np.max(np.abs(coefficients_ridge)):.2f}")
print(f"Min: {np.min(np.abs(coefficients_ridge)):.2f}")
# Much smaller than without regularization!

# Visualization
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot_ridge = model_ridge.predict(X_plot)

plt.figure(figsize=(12, 5))

# Left: Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, s=30, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, s=30, label='Test')
plt.plot(X_plot, y_plot_ridge, 'b-', linewidth=2, label='Ridge (α=10)')
plt.plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, alpha=0.7, label='True function')
plt.title('Predictions - WITH Ridge Regularization\n(Better fit!)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 2)

# Right: Coefficients comparison
plt.subplot(1, 2, 2)
plt.bar(range(len(coefficients)), coefficients, alpha=0.5, label='No Reg')
plt.bar(range(len(coefficients_ridge)), coefficients_ridge, alpha=0.5, label='Ridge')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficients Comparison\n(Ridge shrinks coefficients)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Alpha Tuning - Kako Izabrati?
```python
# Test different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []

for alpha in alphas:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    
    model.fit(X_train, y_train)
    
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    
    train_scores.append(train_mse)
    test_scores.append(test_mse)
    
    print(f"α={alpha:7.3f}: Train MSE={train_mse:.3f}, Test MSE={test_mse:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_scores, 'o-', label='Train MSE', linewidth=2)
plt.plot(alphas, test_scores, 's-', label='Test MSE', linewidth=2)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('MSE')
plt.title('Ridge Regularization - Alpha Tuning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(alphas[np.argmin(test_scores)], color='green', linestyle='--', 
            alpha=0.7, label=f'Optimal α = {alphas[np.argmin(test_scores)]}')
plt.legend()
plt.show()

# Optimal alpha
optimal_alpha = alphas[np.argmin(test_scores)]
print(f"\n✅ Optimal Alpha: {optimal_alpha}")
```

### Ridge za Classification:
```python
from sklearn.linear_model import LogisticRegression

# Logistic Regression sa L2 (Ridge)
model_logreg_ridge = LogisticRegression(
    penalty='l2',        # L2 regularization
    C=1.0,               # Inverse of regularization strength (C = 1/α)
    solver='lbfgs',
    max_iter=1000
)

# Note: C parametar u LogisticRegression
# C large → weak regularization (α small)
# C small → strong regularization (α large)
```

---

## 2. L1 Regularization (Lasso)

**L1 Regularization** dodaje **apsolutnu vrednost koeficijenata** u loss funkciju - **feature selection**!

### Matematika:
```
SA L1 Regularizacija (Lasso):
Loss = MSE + α × Σ|w|
     = Σ(y - ŷ)² + α × ||w||₁

Gde:
α (alpha) = Regularization strength
w = Model coefficients
||w||₁ = Sum of absolute values (L1 norm)

KEY DIFFERENCE vs L2:
L1 može da stavi koeficijente na TAČNO 0 → Feature selection!
```

### Karakteristike L1:
```
✅ Sets some coefficients to EXACTLY 0 (feature selection!)
✅ Sparse solutions (many coefficients = 0)
✅ Built-in feature selection
✅ Good when many features are irrelevant
❌ Not differentiable at 0 (harder to optimize)
❌ Can be unstable with correlated features
```

### Python - Lasso Regression:
```python
from sklearn.linear_model import Lasso

# Lasso model (α = 0.1)
model_lasso = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('scaler', StandardScaler()),  # MUST scale for Lasso!
    ('lasso', Lasso(alpha=0.1))
])

model_lasso.fit(X_train, y_train)

# Predictions
y_train_pred_lasso = model_lasso.predict(X_train)
y_test_pred_lasso = model_lasso.predict(X_test)

# Errors
train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)

print("Model WITH Lasso Regularization (α=0.1):")
print(f"Train MSE: {train_mse_lasso:.3f}")
print(f"Test MSE:  {test_mse_lasso:.3f}")

# Coefficients - MANY are ZERO!
coefficients_lasso = model_lasso.named_steps['lasso'].coef_
n_zero = np.sum(coefficients_lasso == 0)
n_nonzero = np.sum(coefficients_lasso != 0)

print(f"\nCoefficients (Lasso):")
print(f"Zero coefficients:     {n_zero}")
print(f"Non-zero coefficients: {n_nonzero}")
print(f"Feature selection:     {n_zero}/{len(coefficients_lasso)} features removed!")

# Visualization
plt.figure(figsize=(14, 5))

# Left: Predictions
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, alpha=0.6, s=30, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, s=30, label='Test')
y_plot_lasso = model_lasso.predict(X_plot)
plt.plot(X_plot, y_plot_lasso, 'purple', linewidth=2, label='Lasso (α=0.1)')
plt.plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, alpha=0.7, label='True function')
plt.title('Predictions - Lasso Regularization')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 2)

# Middle: Coefficients (No Reg vs Lasso)
plt.subplot(1, 3, 2)
plt.bar(range(len(coefficients)), coefficients, alpha=0.5, label='No Reg')
plt.bar(range(len(coefficients_lasso)), coefficients_lasso, alpha=0.5, label='Lasso')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficients - Lasso sets many to 0!')
plt.legend()
plt.grid(True, alpha=0.3)

# Right: Non-zero coefficients only
plt.subplot(1, 3, 3)
nonzero_idx = np.where(coefficients_lasso != 0)[0]
plt.bar(nonzero_idx, coefficients_lasso[nonzero_idx], color='purple', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title(f'Selected Features (Lasso)\n{n_nonzero} out of {len(coefficients_lasso)}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### L1 za Feature Selection:
```python
# Feature selection sa Lasso
from sklearn.datasets import make_regression

# Generate data sa mnogo features (većina irrelevantna)
X_many, y_many = make_regression(
    n_samples=500, 
    n_features=100,
    n_informative=10,  # Only 10 relevant!
    noise=10,
    random_state=42
)

X_train_many, X_test_many, y_train_many, y_test_many = train_test_split(
    X_many, y_many, test_size=0.3, random_state=42
)

# Lasso za feature selection
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_many)
X_test_scaled = scaler.transform(X_test_many)

lasso_selector = Lasso(alpha=1.0)
lasso_selector.fit(X_train_scaled, y_train_many)

# Selected features
selected_features = np.where(lasso_selector.coef_ != 0)[0]
print(f"Features selected by Lasso: {len(selected_features)}/{X_many.shape[1]}")

# Train model sa samo selected features
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

from sklearn.linear_model import LinearRegression
model_selected = LinearRegression()
model_selected.fit(X_train_selected, y_train_many)

# Compare: All features vs Selected
model_all = LinearRegression()
model_all.fit(X_train_scaled, y_train_many)

score_all = model_all.score(X_test_scaled, y_test_many)
score_selected = model_selected.score(X_test_selected, y_test_many)

print(f"\nR² Score (All 100 features):     {score_all:.3f}")
print(f"R² Score (Selected {len(selected_features)} features): {score_selected:.3f}")
print(f"Difference:                      {score_selected - score_all:+.3f}")
# Često skoro isti score sa MNOGO manje features!
```

### Lasso za Classification:
```python
# Logistic Regression sa L1 (Lasso)
model_logreg_lasso = LogisticRegression(
    penalty='l1',        # L1 regularization
    C=1.0,
    solver='liblinear',  # liblinear ili saga za L1
    max_iter=1000
)

# Note: mora solver='liblinear' ili 'saga' za L1!
```

---

## 3. L1 vs L2 Comparison

### Vizualna Razlika:
```python
# Generate simple 2D data
np.random.seed(42)
X_2d = np.random.randn(100, 2)
y_2d = 2 * X_2d[:, 0] + 0.5 * X_2d[:, 1] + np.random.randn(100) * 0.5

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)

# Test different alphas
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_2d, y_train_2d)
    ridge_coefs.append(ridge.coef_)
    
    # Lasso
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_2d, y_train_2d)
    lasso_coefs.append(lasso.coef_)

# Plot coefficient paths
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ridge coefficient path
for i in range(2):
    axes[0].plot(alphas, [coef[i] for coef in ridge_coefs], 
                 'o-', linewidth=2, label=f'Feature {i}')
axes[0].set_xscale('log')
axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Coefficient Value')
axes[0].set_title('Ridge (L2) - Coefficient Path\nCoefficients shrink towards 0')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Lasso coefficient path
for i in range(2):
    axes[1].plot(alphas, [coef[i] for coef in lasso_coefs], 
                 's-', linewidth=2, label=f'Feature {i}')
axes[1].set_xscale('log')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Coefficient Value')
axes[1].set_title('Lasso (L1) - Coefficient Path\nCoefficients become exactly 0')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Key observation: Lasso hits 0, Ridge asymptotes to 0!
```

### Comparison Table:
```python
import pandas as pd

comparison = pd.DataFrame({
    'Property': [
        'Penalty Term',
        'Coefficient Behavior',
        'Feature Selection',
        'Sparsity',
        'Correlated Features',
        'Optimization',
        'Use When'
    ],
    'L2 (Ridge)': [
        'Σw²',
        'Shrinks towards 0',
        'No (all kept)',
        'No (all non-zero)',
        'Handles well',
        'Easy (differentiable)',
        'All features relevant'
    ],
    'L1 (Lasso)': [
        'Σ|w|',
        'Sets to exactly 0',
        'Yes (automatic)',
        'Yes (sparse)',
        'Can be unstable',
        'Harder (non-differentiable at 0)',
        'Many irrelevant features'
    ]
})

print(comparison.to_string(index=False))
```

### Kada Koristiti L1 vs L2:
```
Use L2 (Ridge) when:
✅ All features are potentially relevant
✅ Features are highly correlated (multicollinearity)
✅ Want smooth, stable coefficients
✅ Default choice for regularization

Use L1 (Lasso) when:
✅ Many features are irrelevant
✅ Want automatic feature selection
✅ Need sparse model (interpretability)
✅ Feature selection is priority

Not sure?
→ Use Elastic Net (combines both!)
```

---

## 4. Elastic Net (L1 + L2)

**Elastic Net** kombinuje L1 i L2 - **best of both worlds**!

### Matematika:
```
Elastic Net:
Loss = MSE + α × (l1_ratio × Σ|w| + (1 - l1_ratio) × Σw²)
     = MSE + α × (l1_ratio × ||w||₁ + (1 - l1_ratio) × ||w||₂²)

Parameters:
α (alpha)         = Overall regularization strength
l1_ratio (ρ)      = Balance between L1 and L2

l1_ratio = 0   → Pure L2 (Ridge)
l1_ratio = 1   → Pure L1 (Lasso)
l1_ratio = 0.5 → Equal mix of L1 and L2
```

### Python:
```python
from sklearn.linear_model import ElasticNet

# Elastic Net
model_elastic = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('scaler', StandardScaler()),
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5))  # 50% L1, 50% L2
])

model_elastic.fit(X_train, y_train)

# Predictions
y_pred_elastic = model_elastic.predict(X_test)
test_mse_elastic = mean_squared_error(y_test, y_pred_elastic)

print("Elastic Net (α=0.1, l1_ratio=0.5):")
print(f"Test MSE: {test_mse_elastic:.3f}")

# Coefficients
coefficients_elastic = model_elastic.named_steps['elastic'].coef_
n_zero_elastic = np.sum(coefficients_elastic == 0)

print(f"Zero coefficients: {n_zero_elastic}/{len(coefficients_elastic)}")
print("→ Feature selection like Lasso, but more stable!")
```

### Tuning l1_ratio:
```python
# Test different l1_ratio values
l1_ratios = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
test_scores = []
n_zeros = []

for ratio in l1_ratios:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=10)),
        ('scaler', StandardScaler()),
        ('elastic', ElasticNet(alpha=0.1, l1_ratio=ratio))
    ])
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    n_zero = np.sum(model.named_steps['elastic'].coef_ == 0)
    
    test_scores.append(score)
    n_zeros.append(n_zero)
    
    print(f"l1_ratio={ratio:.1f}: R²={score:.3f}, Zero coefs={n_zero}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(l1_ratios, test_scores, 'o-', linewidth=2)
axes[0].set_xlabel('l1_ratio (0=Ridge, 1=Lasso)')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Elastic Net - l1_ratio vs Performance')
axes[0].grid(True, alpha=0.3)

axes[1].plot(l1_ratios, n_zeros, 's-', linewidth=2, color='orange')
axes[1].set_xlabel('l1_ratio (0=Ridge, 1=Lasso)')
axes[1].set_ylabel('Number of Zero Coefficients')
axes[1].set_title('Elastic Net - l1_ratio vs Sparsity')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Prednosti Elastic Net:
```
✅ Feature selection (like Lasso)
✅ Handles correlated features (better than Lasso)
✅ More stable than pure Lasso
✅ Flexible (tune l1_ratio za balance)
✅ Good default for unknown problems
```

---

## 5. Regularization za Različite Algoritme

### Linear Models (Ridge, Lasso, Elastic Net):
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge
ridge = Ridge(alpha=1.0)

# Lasso
lasso = Lasso(alpha=1.0)

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)

# IMPORTANT: ALWAYS scale features za L1/L2!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Logistic Regression:
```python
from sklearn.linear_model import LogisticRegression

# L2 (default)
logreg_l2 = LogisticRegression(
    penalty='l2',
    C=1.0,            # C = 1/α (inverse)
    solver='lbfgs',
    max_iter=1000
)

# L1
logreg_l1 = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',  # Must use liblinear ili saga za L1!
    max_iter=1000
)

# Elastic Net
logreg_elastic = LogisticRegression(
    penalty='elasticnet',
    C=1.0,
    l1_ratio=0.5,
    solver='saga',      # Must use saga za elasticnet!
    max_iter=1000
)

# No regularization
logreg_none = LogisticRegression(
    penalty=None,
    solver='lbfgs',
    max_iter=1000
)
```

### SVM:
```python
from sklearn.svm import SVC, SVR

# C parametar kontroliše regularizaciju (inverse!)
# C large → Weak regularization (α small)
# C small → Strong regularization (α large)

svm_weak_reg = SVC(C=10)     # Weak regularization
svm_strong_reg = SVC(C=0.1)  # Strong regularization

# L2 by default za SVM
```

### Random Forest (Implicit Regularization):
```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest nema eksplicitan L1/L2, ali ima regularization kroz:

rf_regularized = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # Limit depth (regularization!)
    min_samples_split=10,   # Require more samples (regularization!)
    min_samples_leaf=5,     # Require more samples in leaves
    max_features='sqrt',    # Limit features per split
    max_samples=0.8         # Bootstrap sample size (< 1.0 = regularization)
)

# Smaller max_depth = stronger regularization
# Larger min_samples_split = stronger regularization
```

### XGBoost:
```python
import xgboost as xgb

# XGBoost ima MNOGO regularization parametara!

xgb_regularized = xgb.XGBClassifier(
    # L1 & L2 regularization
    reg_alpha=1.0,        # L1 regularization on weights
    reg_lambda=1.0,       # L2 regularization on weights
    
    # Tree structure regularization
    max_depth=5,          # Limit depth
    min_child_weight=3,   # Minimum sum of instance weight in child
    gamma=0.1,            # Minimum loss reduction for split (pruning)
    
    # Stochastic regularization (variance reduction)
    subsample=0.8,        # Row sampling
    colsample_bytree=0.8, # Column sampling per tree
    
    # Learning
    learning_rate=0.1,    # Shrinkage (regularization!)
    n_estimators=100
)

# Higher reg_alpha/reg_lambda = stronger regularization
# Smaller subsample = stronger regularization
# Lower learning_rate = implicit regularization (with more trees)
```

### Neural Networks (Multiple Types!):
```python
from sklearn.neural_network import MLPClassifier

# 1. L2 Regularization (Weight Decay)
mlp_l2 = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    alpha=0.001,          # L2 penalty (weight decay)
    max_iter=1000,
    random_state=42
)

# 2. Early Stopping (implicit regularization)
mlp_early = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 epochs
    max_iter=1000,
    random_state=42
)

# 3. Batch Normalization, Dropout (mora koristiti Keras/PyTorch)
# sklearn MLP nema dropout built-in
```

---

## 6. Dropout (Neural Networks)

**Dropout** random-ly "isključuje" neurone tokom treniranja - sprečava co-adaptation.

### Kako Radi:
```
Tokom treniranja:
├─ Za svaki batch/iteration
├─ Random-ly set p% neurona na 0 (dropout rate)
├─ Treniraj sa preostalim neuronima
└─ Repeat

Tokom inference (prediction):
└─ Use ALL neurons (no dropout)

Effect: Ensemble of sub-networks → Reduces overfitting!
```

### Keras/TensorFlow Example:
```python
# Dropout sa Keras (sklearn MLP nema dropout)
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dropout(0.5),    # Dropout 50% neurona
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),    # Dropout 30% neurona
    
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=32,
    verbose=0
)

# Plot train vs validation
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Dropout Regularization - Train vs Validation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Dropout Rate Guidelines:
```
Dropout Rate = 0.0  → No regularization (overfitting risk)
Dropout Rate = 0.2  → Mild regularization
Dropout Rate = 0.5  → Strong regularization (common default)
Dropout Rate = 0.8  → Very strong (može underfitting)

Common practice:
├─ Input layer: 0.1-0.2 (weak)
├─ Hidden layers: 0.3-0.5 (moderate to strong)
└─ Before output: 0.5 (strong)

Note: Dropout SAMO tokom treniranja, NE tokom inference!
```

---

## 7. Early Stopping

**Early Stopping** zaustavlja trening kada validation performance prestane da se poboljšava.

### Kako Radi:
```
1. Split data: Train + Validation
2. Train model
3. Monitor validation loss svaki epoch
4. Ako validation loss ne pada za N epochs (patience):
   └─ Stop training
5. Restore weights from best epoch
```

### Python - sklearn:
```python
from sklearn.neural_network import MLPClassifier

# Neural Network sa Early Stopping
mlp_early = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,             # Maximum iterations
    early_stopping=True,       # Enable early stopping
    validation_fraction=0.2,   # Use 20% of train as validation
    n_iter_no_change=10,       # Patience: stop if no improvement for 10 epochs
    random_state=42
)

mlp_early.fit(X_train, y_train)

print(f"Training stopped at iteration: {mlp_early.n_iter_}")
print(f"Best validation score: {mlp_early.best_validation_score_:.3f}")
```

### Keras/TensorFlow:
```python
from tensorflow import keras

# Early Stopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=10,             # Stop if no improvement for 10 epochs
    restore_best_weights=True, # Restore weights from best epoch
    verbose=1
)

# Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train sa Early Stopping
history = model.fit(
    X_train, y_train,
    epochs=200,              # Large max, but will stop early
    validation_split=0.2,
    callbacks=[early_stop],  # Early stopping callback
    verbose=0
)

print(f"Training stopped at epoch: {early_stop.stopped_epoch}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.axvline(early_stop.stopped_epoch, color='red', linestyle='--', 
            label='Early Stop')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 8. Regularization Strength Tuning

### GridSearchCV za Alpha/C:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Parameter grid - LOG SCALE za alpha!
param_grid = {
    'alpha': np.logspace(-3, 3, 20)  # 0.001 to 1000 on log scale
}

# Grid Search
grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']:.4f}")
print(f"Best CV R²: {grid_search.best_score_:.3f}")

# Test score
test_score = grid_search.best_estimator_.score(X_test_scaled, y_test)
print(f"Test R²: {test_score:.3f}")
```

### RidgeCV / LassoCV (Built-in CV):
```python
from sklearn.linear_model import RidgeCV, LassoCV

# RidgeCV - efficient cross-validation za Ridge
alphas = np.logspace(-3, 3, 50)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha (RidgeCV): {ridge_cv.alpha_:.4f}")
print(f"Test R²: {ridge_cv.score(X_test_scaled, y_test):.3f}")

# LassoCV - efficient CV za Lasso
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"\nOptimal alpha (LassoCV): {lasso_cv.alpha_:.4f}")
print(f"Test R²: {lasso_cv.score(X_test_scaled, y_test):.3f}")
```

---

## Decision Framework - Koja Regularizacija?
```python
def recommend_regularization(n_features, n_samples, feature_relevance):
    """
    Preporuči regularizaciju na osnovu podataka.
    
    Parameters:
    - n_features: Number of features
    - n_samples: Number of samples
    - feature_relevance: 'all_relevant', 'some_irrelevant', 'many_irrelevant'
    """
    print("REGULARIZATION RECOMMENDATION")
    print("="*60)
    print(f"Features: {n_features}")
    print(f"Samples: {n_samples}")
    print(f"Feature relevance: {feature_relevance}")
    print("")
    
    ratio = n_features / n_samples
    
    # High dimensional (more features than samples)
    if ratio > 1:
        print("⚠️  HIGH DIMENSIONAL DATA (p > n)")
        print("   Strong regularization REQUIRED!")
        print("")
        
        if feature_relevance == 'many_irrelevant':
            print("✅ RECOMMENDATION: L1 (Lasso)")
            print("   - Automatic feature selection")
            print("   - Sparse solution")
        else:
            print("✅ RECOMMENDATION: Elastic Net")
            print("   - Combines L1 + L2")
            print("   - More stable than pure Lasso")
    
    # Many features, balanced samples
    elif ratio > 0.5:
        print("📊 MANY FEATURES relative to samples")
        print("   Regularization RECOMMENDED")
        print("")
        
        if feature_relevance == 'all_relevant':
            print("✅ RECOMMENDATION: L2 (Ridge)")
            print("   - Keep all features")
            print("   - Shrink coefficients")
        elif feature_relevance == 'some_irrelevant':
            print("✅ RECOMMENDATION: Elastic Net")
            print("   - l1_ratio=0.5 (balanced)")
        else:
            print("✅ RECOMMENDATION: L1 (Lasso)")
            print("   - Automatic feature selection")
    
    # Normal case
    else:
        print("✅ NORMAL DATA (more samples than features)")
        print("")
        
        if feature_relevance == 'all_relevant':
            print("✅ RECOMMENDATION: L2 (Ridge) or Light Regularization")
            print("   - alpha=0.1-1.0")
        elif feature_relevance == 'many_irrelevant':
            print("✅ RECOMMENDATION: L1 (Lasso)")
            print("   - Feature selection priority")
        else:
            print("✅ RECOMMENDATION: Elastic Net or Light L2")
            print("   - Default choice")
    
    print("\n" + "="*60)

# Examples
recommend_regularization(100, 50, 'many_irrelevant')   # p > n
recommend_regularization(50, 100, 'some_irrelevant')   # p < n but many features
recommend_regularization(10, 1000, 'all_relevant')     # Normal case
```

---

## Complete Example - Regularization Comparison
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 1. GENERATE DATA ====================
# High-dimensional data (many features, some irrelevant)
X, y = make_regression(
    n_samples=200,
    n_features=50,
    n_informative=10,  # Only 10 relevant
    n_redundant=10,
    noise=20,
    random_state=42
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Informative features: 10/{X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale (IMPORTANT for regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 2. MODELS ====================
models = {
    'No Regularization': LinearRegression(),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10)': Ridge(alpha=10),
    'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'Lasso (α=1.0)': Lasso(alpha=1.0, max_iter=10000),
    'Elastic Net (0.5)': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                scoring='r2')
    
    # Coefficients
    if hasattr(model, 'coef_'):
        n_zero = np.sum(model.coef_ == 0)
        max_coef = np.max(np.abs(model.coef_))
    else:
        n_zero = 0
        max_coef = 0
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'CV R²': cv_scores.mean(),
        'Gap': train_r2 - test_r2,
        'Zero Coefs': n_zero,
        'Max |Coef|': max_coef
    })

# Results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("REGULARIZATION COMPARISON")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# ==================== 3. VISUALIZATIONS ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 Train vs Test R²
ax = axes[0, 0]
x_pos = np.arange(len(results_df))
width = 0.35

ax.bar(x_pos - width/2, results_df['Train R²'], width, 
       label='Train R²', alpha=0.8)
ax.bar(x_pos + width/2, results_df['Test R²'], width, 
       label='Test R²', alpha=0.8)

ax.set_ylabel('R² Score')
ax.set_title('Train vs Test Performance')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3.2 Overfitting Gap
ax = axes[0, 1]
colors = ['red' if gap > 0.1 else 'green' for gap in results_df['Gap']]
ax.bar(results_df['Model'], results_df['Gap'], color=colors, alpha=0.7)
ax.axhline(0.1, color='orange', linestyle='--', 
           label='Overfitting Threshold (0.1)')
ax.set_ylabel('Train R² - Test R² (Gap)')
ax.set_title('Overfitting Check\n(Lower is Better)')
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3.3 Feature Selection (Zero Coefficients)
ax = axes[1, 0]
ax.bar(results_df['Model'], results_df['Zero Coefs'], 
       color='purple', alpha=0.7)
ax.set_ylabel('Number of Zero Coefficients')
ax.set_title('Feature Selection\n(Lasso sets coefficients to 0)')
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 3.4 Coefficient Magnitude
ax = axes[1, 1]
ax.bar(results_df['Model'], results_df['Max |Coef|'], 
       color='orange', alpha=0.7)
ax.set_ylabel('Max |Coefficient|')
ax.set_title('Coefficient Shrinkage\n(Regularization reduces magnitude)')
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 4. BEST MODEL ====================
best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
best_test_r2 = results_df['Test R²'].max()

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {best_test_r2:.3f}")

# ==================== 5. ALPHA TUNING ====================
print("\n" + "="*80)
print("ALPHA TUNING FOR RIDGE")
print("="*80)

from sklearn.linear_model import RidgeCV

alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal Alpha: {ridge_cv.alpha_:.4f}")
print(f"Test R²: {ridge_cv.score(X_test_scaled, y_test):.3f}")

# Plot alpha tuning
ridge_scores = []
for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    scores = cross_val_score(ridge_temp, X_train_scaled, y_train, cv=5)
    ridge_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_scores, 'o-', linewidth=2)
plt.axvline(ridge_cv.alpha_, color='red', linestyle='--', 
            label=f'Optimal α = {ridge_cv.alpha_:.3f}')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Cross-Validation R²')
plt.title('Ridge Regularization - Alpha Tuning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('alpha_tuning.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Rezime - Regularization

### Quick Reference:

| Regularization | Type | Effect | Use When |
|----------------|------|--------|----------|
| **L2 (Ridge)** | Σw² | Shrinks towards 0 | All features relevant, multicollinearity |
| **L1 (Lasso)** | Σ\|w\| | Sets to exactly 0 | Feature selection, many irrelevant features |
| **Elastic Net** | L1 + L2 | Both | Unsure, want flexibility |
| **Dropout** | Neural | Random deactivation | Deep neural networks |
| **Early Stopping** | Iterative | Stop before overfitting | Neural nets, boosting |

### Parameter Relationships:
```
sklearn:
├─ Ridge, Lasso, ElasticNet: alpha ↑ → Stronger regularization
└─ LogisticRegression, SVM: C ↑ → WEAKER regularization (C = 1/α)

XGBoost:
├─ reg_alpha (L1), reg_lambda (L2) ↑ → Stronger regularization
└─ learning_rate ↓ → Implicit regularization (with more trees)

Neural Networks:
├─ alpha (L2) ↑ → Stronger regularization
├─ dropout_rate ↑ → Stronger regularization
└─ early_stopping: patience ↓ → Stop sooner
```

### Default Strategy:
```
Step 1: Check if overfitting exists
  ├─ Train-test gap > 0.1? → Yes, need regularization
  └─ Train-test gap < 0.05? → Maybe no regularization needed

Step 2: Choose regularization type
  ├─ Many irrelevant features? → L1 (Lasso)
  ├─ All features relevant? → L2 (Ridge)
  └─ Unsure? → Elastic Net

Step 3: Tune regularization strength
  ├─ Use RidgeCV/LassoCV for efficient tuning
  ├─ Or GridSearchCV with log-scale alphas
  └─ Monitor train-test gap

Step 4: Validate
  └─ Cross-validation → Confirm generalization
```

### Common Pitfalls:
```
❌ Forgetting to scale data (CRITICAL for L1/L2!)
❌ Using C when expecting alpha (inverted!)
❌ Not tuning regularization strength
❌ Too strong regularization → Underfitting
❌ Regularization on already regularized models (trees)
```

**Key Takeaway:** Regularization je jedna od najmoćnijih tehnika protiv overfitting-a! L2 (Ridge) je default, L1 (Lasso) za feature selection, Elastic Net za kombinaciju. UVEK scale podatke pre L1/L2! Tune regularization strength sa CV. Regularizacija može značajno poboljšati generalizaciju! 🎯