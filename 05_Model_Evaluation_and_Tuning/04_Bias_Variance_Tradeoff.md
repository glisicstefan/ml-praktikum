# Bias-Variance Tradeoff

Bias-Variance Tradeoff je **fundamentalni koncept u machine learning-u** koji objašnjava zašto modeli greše i kako balansirati kompleksnost modela za najbolju generalizaciju.

**Zašto je ovo kritično?**
- **Razumevanje grešaka** - Svaka greška modela dolazi iz tri izvora: bias, variance, i irreducible noise
- **Overfitting vs Underfitting** - Bias-Variance objašnjava oba problema
- **Model selection** - Pomaže u izboru prave kompleksnosti modela
- **Hyperparameter tuning** - Pokazuje kako parametri utiču na performanse
- **Debugging modela** - Dijagnoza šta nije u redu sa modelom

**VAŽNO:** Ovo nije samo teorija - razumevanje bias-variance je ključ za pravljenje dobrih production modela!

---

## Total Error Decomposition

**Svaka greška modela** se može dekomponovati na tri dela:
```
Total Error = Bias² + Variance + Irreducible Noise

Gde:
- Bias²              = Greška zbog pretpostavki modela (model previše jednostavan)
- Variance           = Greška zbog osetljivosti na training data (model previše kompleksan)
- Irreducible Noise  = Greška u samim podacima (ne možemo smanjiti)
```

### Vizualizacija Koncepta:
```python
import numpy as np
import matplotlib.pyplot as plt

# Target function (prava relacija)
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_function = np.sin(X)

# Irreducible noise
noise = np.random.randn(100) * 0.2
y = true_function + noise

# Predictions from different model complexities
# High Bias (too simple)
from sklearn.linear_model import LinearRegression
model_simple = LinearRegression()
model_simple.fit(X.reshape(-1, 1), y)
y_pred_simple = model_simple.predict(X.reshape(-1, 1))

# Good fit
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))
model_good = LinearRegression()
model_good.fit(X_poly, y)
y_pred_good = model_good.predict(X_poly)

# High Variance (too complex)
poly_features_complex = PolynomialFeatures(degree=15)
X_poly_complex = poly_features_complex.fit_transform(X.reshape(-1, 1))
model_complex = LinearRegression()
model_complex.fit(X_poly_complex, y)
y_pred_complex = model_complex.predict(X_poly_complex)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# High Bias
axes[0].scatter(X, y, alpha=0.3, s=20, label='Data (with noise)')
axes[0].plot(X, true_function, 'g-', linewidth=2, label='True Function')
axes[0].plot(X, y_pred_simple, 'r-', linewidth=2, label='Model Prediction')
axes[0].set_title('HIGH BIAS (Underfitting)\nModel Too Simple', fontsize=12, fontweight='bold')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Good Fit
axes[1].scatter(X, y, alpha=0.3, s=20, label='Data (with noise)')
axes[1].plot(X, true_function, 'g-', linewidth=2, label='True Function')
axes[1].plot(X, y_pred_good, 'b-', linewidth=2, label='Model Prediction')
axes[1].set_title('GOOD FIT (Balanced)\nOptimal Complexity', fontsize=12, fontweight='bold')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# High Variance
axes[2].scatter(X, y, alpha=0.3, s=20, label='Data (with noise)')
axes[2].plot(X, true_function, 'g-', linewidth=2, label='True Function')
axes[2].plot(X, y_pred_complex, 'orange', linewidth=2, label='Model Prediction')
axes[2].set_title('HIGH VARIANCE (Overfitting)\nModel Too Complex', fontsize=12, fontweight='bold')
axes[2].set_xlabel('X')
axes[2].set_ylabel('y')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 1. Bias (Pristrasnost)

**Bias** je greška koja dolazi od **previše jednostavnih pretpostavki** u algoritmu učenja.

### Definicija:
```
Bias = E[ŷ] - y_true

Gde:
E[ŷ] = Očekivana vrednost predvikcija preko različitih training setova
y_true = Prava vrednost

Visok Bias → Model konzistentno greši u istom pravcu
```

### Karakteristike High Bias (Underfitting):
```
✅ Konzistentne predikcije (niska variance)
❌ Loše predikcije (daleko od truth)
❌ Ne hvata kompleksnost podataka
❌ Loš na train data
❌ Loš na test data
```

### Demonstracija:
```python
# Generate complex data (non-linear)
np.random.seed(42)
X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.randn(100) * 0.2

X_test = np.linspace(0, 10, 50).reshape(-1, 1)
y_test = np.sin(X_test).ravel() + np.random.randn(50) * 0.2

# Model sa HIGH BIAS - Linear model za non-linear data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model_bias = LinearRegression()
model_bias.fit(X_train, y_train)

# Predictions
y_train_pred = model_bias.predict(X_train)
y_test_pred = model_bias.predict(X_test)

# Errors
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("HIGH BIAS Model (Linear for Non-linear Data):")
print(f"Train MSE: {train_mse:.3f}")  # Loš
print(f"Test MSE:  {test_mse:.3f}")   # Loš
print(f"Gap:       {abs(train_mse - test_mse):.3f}")  # Mali gap

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, s=30, label='Train Data')
plt.scatter(X_test, y_test, alpha=0.5, s=30, label='Test Data')
plt.plot(X_train, y_train_pred, 'r-', linewidth=2, label='Model (Linear)')
plt.plot(X_train, np.sin(X_train), 'g--', linewidth=2, alpha=0.7, label='True Function')
plt.title('HIGH BIAS (Underfitting) - Model Too Simple')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Model ne hvata sine pattern - previše jednostavan!
```

### Uzroci High Bias:

1. **Model previše jednostavan** - Linear model za non-linear data
2. **Premalo features** - Bitne informacije nedostaju
3. **Previše regularizacija** - Model je previše ograničen
4. **Premalo trening iteracija** - Model nije dovoljno naučio

---

## 2. Variance (Varijabilnost)

**Variance** je greška koja dolazi od **prevelike osetljivosti na training data**.

### Definicija:
```
Variance = E[(ŷ - E[ŷ])²]

Gde:
E[ŷ] = Očekivana vrednost predvikcija
ŷ = Konkretna predikcija

Visoka Variance → Model se MNOGO menja sa različitim training data
```

### Karakteristike High Variance (Overfitting):
```
❌ Nestabilne predikcije (različite za različite training setove)
✅ Perfektne predikcije na train
❌ Loše predikcije na test
❌ "Uči šum" umesto pattern-a
❌ Memorisanje podataka
```

### Demonstracija:
```python
# Model sa HIGH VARIANCE - Polynomial degree 20 za simple data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

model_variance = Pipeline([
    ('poly', PolynomialFeatures(degree=20)),
    ('linear', LinearRegression())
])

model_variance.fit(X_train, y_train)

# Predictions
y_train_pred_var = model_variance.predict(X_train)
y_test_pred_var = model_variance.predict(X_test)

# Errors
train_mse_var = mean_squared_error(y_train, y_train_pred_var)
test_mse_var = mean_squared_error(y_test, y_test_pred_var)

print("\nHIGH VARIANCE Model (Polynomial degree 20):")
print(f"Train MSE: {train_mse_var:.3f}")  # Odličan!
print(f"Test MSE:  {test_mse_var:.3f}")   # KATASTROFA!
print(f"Gap:       {abs(train_mse_var - test_mse_var):.3f}")  # OGROMAN gap!

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, s=30, label='Train Data')
plt.scatter(X_test, y_test, alpha=0.5, s=30, label='Test Data')
plt.plot(X_train, y_train_pred_var, 'orange', linewidth=2, label='Model (Poly 20)')
plt.plot(X_train, np.sin(X_train), 'g--', linewidth=2, alpha=0.7, label='True Function')
plt.title('HIGH VARIANCE (Overfitting) - Model Too Complex')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-2, 2)  # Limit y da vidimo pattern
plt.show()

# Model "uči" svaki noise point - previše kompleksan!
```

### Uzroci High Variance:

1. **Model previše kompleksan** - Polynomial degree 20 za simple pattern
2. **Previše features** - Model može fitovati noise
3. **Premalo training data** - Ne može generalizovati
4. **Premalo regularizacija** - Model nije ograničen
5. **Previše trening iteracija** - Neural Networks memorišu data

---

## 3. Bias-Variance Tradeoff

**Ne možeš imati i nizak bias I nisku variance** - mora biti kompromis!

### Tradeoff Visualizacija:
```python
# Simulate bias-variance tradeoff
model_complexities = range(1, 21)  # Polynomial degrees 1 to 20
train_errors = []
test_errors = []
bias_squared = []
variance = []

# Generate multiple datasets
n_datasets = 50
predictions_per_complexity = {deg: [] for deg in model_complexities}

for deg in model_complexities:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('linear', LinearRegression())
    ])
    
    # Fit on main training set
    model.fit(X_train, y_train)
    
    # Errors
    train_error = mean_squared_error(y_train, model.predict(X_train))
    test_error = mean_squared_error(y_test, model.predict(X_test))
    
    train_errors.append(train_error)
    test_errors.append(test_error)
    
    # Simulate bias and variance (multiple datasets)
    dataset_predictions = []
    for _ in range(n_datasets):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Train
        model_boot = Pipeline([
            ('poly', PolynomialFeatures(degree=deg)),
            ('linear', LinearRegression())
        ])
        model_boot.fit(X_boot, y_boot)
        
        # Predict on test
        pred = model_boot.predict(X_test)
        dataset_predictions.append(pred)
    
    # Calculate bias and variance
    predictions_array = np.array(dataset_predictions)
    mean_predictions = predictions_array.mean(axis=0)
    
    # Bias² = (E[ŷ] - y_true)²
    bias_sq = np.mean((mean_predictions - y_test) ** 2)
    bias_squared.append(bias_sq)
    
    # Variance = E[(ŷ - E[ŷ])²]
    var = np.mean(np.var(predictions_array, axis=0))
    variance.append(var)

# Plot Bias-Variance Tradeoff
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Train vs Test Error
axes[0].plot(model_complexities, train_errors, 'o-', linewidth=2, label='Train Error')
axes[0].plot(model_complexities, test_errors, 's-', linewidth=2, label='Test Error')
axes[0].axvline(3, color='green', linestyle='--', alpha=0.7, label='Optimal Complexity')
axes[0].set_xlabel('Model Complexity (Polynomial Degree)')
axes[0].set_ylabel('MSE')
axes[0].set_title('Train vs Test Error')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Right: Bias² vs Variance
axes[1].plot(model_complexities, bias_squared, 'o-', linewidth=2, label='Bias²', color='red')
axes[1].plot(model_complexities, variance, 's-', linewidth=2, label='Variance', color='blue')
total_error = np.array(bias_squared) + np.array(variance)
axes[1].plot(model_complexities, total_error, '^-', linewidth=2, label='Bias² + Variance', color='purple')
axes[1].axvline(3, color='green', linestyle='--', alpha=0.7, label='Optimal Complexity')
axes[1].set_xlabel('Model Complexity (Polynomial Degree)')
axes[1].set_ylabel('Error')
axes[1].set_title('Bias-Variance Tradeoff')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("Low Complexity (degree 1-2):  High Bias, Low Variance  → Underfitting")
print("Medium Complexity (degree 3-5): Low Bias, Low Variance  → Optimal!")
print("High Complexity (degree 15+):  Low Bias, High Variance → Overfitting")
```

### Tradeoff Summary:
```
Model Complexity ↑
├─→ Bias ↓        (Model može fitovati kompleksnije pattern-e)
└─→ Variance ↑    (Model postaje osetljiviji na training data)

Sweet Spot: Gde je Bias² + Variance minimalno!
```

---

## 4. Underfitting (High Bias)

**Underfitting** = Model je **previše jednostavan** da bi uhvatio pattern u podacima.

### Simptomi Underfitting-a:
```python
# Check for underfitting
def check_underfitting(train_score, test_score, threshold=0.7):
    """
    Detektuj underfitting.
    """
    print("Underfitting Check:")
    print(f"Train Score: {train_score:.3f}")
    print(f"Test Score:  {test_score:.3f}")
    print(f"Gap:         {abs(train_score - test_score):.3f}")
    
    if train_score < threshold and test_score < threshold:
        print("🚨 UNDERFITTING DETECTED!")
        print("   Both train and test scores are LOW")
        print("   Model is too simple!")
        return True
    
    return False

# Example
from sklearn.metrics import r2_score

# Simple model
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

train_r2 = r2_score(y_train, model_simple.predict(X_train))
test_r2 = r2_score(y_test, model_simple.predict(X_test))

check_underfitting(train_r2, test_r2)

# Output:
# Train Score: 0.120  ← LOW!
# Test Score:  0.135  ← LOW!
# Gap:         0.015  ← Small gap
# 🚨 UNDERFITTING DETECTED!
```

### Kako Prepoznati Underfitting:

1. **Train score je loš** (< 0.7 za R², < 0.7 za accuracy)
2. **Test score je takođe loš** (sličan train score-u)
3. **Mali train-test gap** (< 0.05)
4. **Learning curve: Obe krive su niske i paralelne**

### Kako Rešiti Underfitting:
```python
# Solutions for Underfitting

# ✅ 1. Increase Model Complexity
# Bad: Linear model
model_simple = LinearRegression()

# Good: Polynomial model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

model_complex = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

# ✅ 2. Add More Features
# Bad: Only 2 features
X_few = X_train[:, :2]

# Good: All features + interactions
poly = PolynomialFeatures(degree=2, include_bias=False)
X_more = poly.fit_transform(X_train)

# ✅ 3. Reduce Regularization
# Bad: Too much regularization
from sklearn.linear_model import Ridge
model_over_reg = Ridge(alpha=1000)  # Alpha too high!

# Good: Less regularization
model_good_reg = Ridge(alpha=1)

# ✅ 4. Train Longer (for iterative algorithms)
# Bad: Too few iterations
from sklearn.neural_network import MLPRegressor
model_few_iter = MLPRegressor(max_iter=10)  # Not enough!

# Good: More iterations
model_more_iter = MLPRegressor(max_iter=1000)

# ✅ 5. Try Different Algorithm
# Bad: Linear model for non-linear data
from sklearn.linear_model import LinearRegression
model_linear = LinearRegression()

# Good: Non-linear model
from sklearn.ensemble import RandomForestRegressor
model_nonlinear = RandomForestRegressor()
```

---

## 5. Overfitting (High Variance)

**Overfitting** = Model je **previše kompleksan** i "uči" noise umesto pravog pattern-a.

### Simptomi Overfitting-a:
```python
# Check for overfitting
def check_overfitting(train_score, test_score, gap_threshold=0.1):
    """
    Detektuj overfitting.
    """
    print("Overfitting Check:")
    print(f"Train Score: {train_score:.3f}")
    print(f"Test Score:  {test_score:.3f}")
    print(f"Gap:         {train_score - test_score:.3f}")
    
    if train_score > 0.9 and (train_score - test_score) > gap_threshold:
        print("🚨 OVERFITTING DETECTED!")
        print("   Train score is HIGH, but test score is MUCH LOWER")
        print("   Model is too complex or memorizing training data!")
        return True
    
    return False

# Example
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Complex model
model_complex = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('linear', LinearRegression())
])
model_complex.fit(X_train, y_train)

train_r2 = r2_score(y_train, model_complex.predict(X_train))
test_r2 = r2_score(y_test, model_complex.predict(X_test))

check_overfitting(train_r2, test_r2)

# Output:
# Train Score: 0.987  ← VERY HIGH!
# Test Score:  0.245  ← LOW!
# Gap:         0.742  ← HUGE GAP!
# 🚨 OVERFITTING DETECTED!
```

### Kako Prepoznati Overfitting:

1. **Train score je odličan** (> 0.9)
2. **Test score je loš** (mnogo niži od train)
3. **Veliki train-test gap** (> 0.1)
4. **Learning curve: Train kriva visoka, validation kriva niska**

### Kako Rešiti Overfitting:
```python
# Solutions for Overfitting

# ✅ 1. Get More Training Data
# Bad: Only 50 samples
X_train_small = X_train[:50]
y_train_small = y_train[:50]

# Good: More data
# Collect more data if possible!

# ✅ 2. Reduce Model Complexity
# Bad: Polynomial degree 20
model_complex = Pipeline([
    ('poly', PolynomialFeatures(degree=20)),
    ('linear', LinearRegression())
])

# Good: Polynomial degree 3
model_simple = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])

# ✅ 3. Add Regularization
# Bad: No regularization
from sklearn.linear_model import LinearRegression
model_no_reg = LinearRegression()

# Good: L2 regularization (Ridge)
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=10)

# Good: L1 regularization (Lasso)
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=1)

# ✅ 4. Feature Selection
# Bad: All 100 features (many irrelevant)
X_all = X_train

# Good: Select top 20 features
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X_train, y_train)

# ✅ 5. Cross-Validation
# Bad: Single train-test split
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Good: Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
mean_score = scores.mean()

# ✅ 6. Early Stopping (for iterative algorithms)
# Neural Networks, Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

model_early = GradientBoostingRegressor(
    n_estimators=1000,
    validation_fraction=0.2,  # Use 20% for validation
    n_iter_no_change=10,      # Stop if no improvement for 10 iterations
    random_state=42
)

# ✅ 7. Dropout (for Neural Networks)
from sklearn.neural_network import MLPRegressor

model_dropout = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    alpha=0.001,  # L2 regularization
    early_stopping=True,
    random_state=42
)

# ✅ 8. Ensemble Methods
# Bagging reduces variance
from sklearn.ensemble import BaggingRegressor

model_bagging = BaggingRegressor(
    estimator=LinearRegression(),
    n_estimators=50,
    random_state=42
)
```

---

## 6. Learning Curves

**Learning Curves** pokazuju kako **train i validation score** se menjaju sa količinom training data.

### Kako Kreirati Learning Curve:
```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title, cv=5):
    """
    Plot learning curve za estimator.
    """
    # Calculate learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    # Calculate mean and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Train score
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='blue')
    
    # Validation score
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score (R²)')
    plt.title(f'Learning Curve - {title}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    return train_sizes, train_mean, val_mean

# Generate data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)

# High Bias Model (Underfitting)
from sklearn.linear_model import LinearRegression
model_bias = LinearRegression()

plt.subplot(1, 3, 1)
plot_learning_curve(model_bias, X, y, 'High Bias (Underfitting)')

# Good Model
from sklearn.ensemble import RandomForestRegressor
model_good = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

plt.subplot(1, 3, 2)
plot_learning_curve(model_good, X, y, 'Good Fit')

# High Variance Model (Overfitting)
model_variance = RandomForestRegressor(n_estimators=50, max_depth=None, 
                                       min_samples_split=2, random_state=42)

plt.subplot(1, 3, 3)
plot_learning_curve(model_variance, X, y, 'High Variance (Overfitting)')

plt.tight_layout()
plt.show()
```

### Interpretacija Learning Curves:
```
UNDERFITTING (High Bias):
├─ Train score: Low and plateaus early
├─ Validation score: Low and plateaus early
├─ Gap: Small (curves converge)
└─ Solution: More complex model, more features

GOOD FIT:
├─ Train score: High
├─ Validation score: High and close to train
├─ Gap: Small
└─ Perfect!

OVERFITTING (High Variance):
├─ Train score: Very high
├─ Validation score: Low
├─ Gap: Large (curves don't converge)
└─ Solution: More data, regularization, simpler model

MORE DATA HELPS?
├─ If curves are converging → NO (underfitting)
└─ If gap is large → YES! (overfitting)
```

### Learning Curve Patterns:
```python
# Visualize different patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pattern 1: High Bias (Underfitting)
axes[0, 0].plot([10, 50, 100, 200, 500], [0.3, 0.35, 0.37, 0.38, 0.38], 'o-', label='Train')
axes[0, 0].plot([10, 50, 100, 200, 500], [0.28, 0.32, 0.34, 0.35, 0.35], 's-', label='Validation')
axes[0, 0].set_title('High Bias (Underfitting)\nBoth Low, Converged', fontweight='bold')
axes[0, 0].set_xlabel('Training Size')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(250, 0.15, '❌ More data won\'t help\n✅ Need complex model', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Pattern 2: High Variance (Overfitting)
axes[0, 1].plot([10, 50, 100, 200, 500], [0.95, 0.92, 0.90, 0.88, 0.87], 'o-', label='Train')
axes[0, 1].plot([10, 50, 100, 200, 500], [0.50, 0.55, 0.62, 0.68, 0.73], 's-', label='Validation')
axes[0, 1].set_title('High Variance (Overfitting)\nLarge Gap, Converging', fontweight='bold')
axes[0, 1].set_xlabel('Training Size')
axes[0, 1].set_ylabel('Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(250, 0.30, '✅ More data will help!\n✅ Add regularization', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Pattern 3: Good Fit
axes[1, 0].plot([10, 50, 100, 200, 500], [0.85, 0.82, 0.80, 0.79, 0.78], 'o-', label='Train')
axes[1, 0].plot([10, 50, 100, 200, 500], [0.75, 0.76, 0.77, 0.77, 0.77], 's-', label='Validation')
axes[1, 0].set_title('Good Fit\nBoth High, Small Gap', fontweight='bold')
axes[1, 0].set_xlabel('Training Size')
axes[1, 0].set_ylabel('Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(250, 0.50, '✅ Perfect!\n Model is well-balanced', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Pattern 4: Need More Data
axes[1, 1].plot([10, 50, 100, 200, 500], [0.92, 0.89, 0.87, 0.85, 0.84], 'o-', label='Train')
axes[1, 1].plot([10, 50, 100, 200, 500], [0.60, 0.65, 0.70, 0.74, 0.77], 's-', label='Validation')
axes[1, 1].set_title('Still Improving\nGap Closing', fontweight='bold')
axes[1, 1].set_xlabel('Training Size')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].text(250, 0.40, '✅ Get more data!\nCurves still converging', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
```

---

## 7. Validation Curves

**Validation Curves** pokazuju kako **hyperparameter** utiče na train i validation score.

### Kreiranje Validation Curve:
```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, X, y, param_name, param_range, title):
    """
    Plot validation curve za hyperparameter.
    """
    # Calculate validation curve
    train_scores, val_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    # Calculate mean and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Train score
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Train Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='blue')
    
    # Validation score
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='red')
    
    # Optimal point
    best_idx = np.argmax(val_mean)
    best_param = param_range[best_idx]
    best_score = val_mean[best_idx]
    
    plt.axvline(best_param, color='green', linestyle='--', alpha=0.7,
                label=f'Optimal: {best_param}')
    plt.scatter([best_param], [best_score], color='green', s=100, zorder=5)
    
    plt.xlabel(param_name)
    plt.ylabel('Score (R²)')
    plt.title(f'Validation Curve - {title}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    print(f"Optimal {param_name}: {best_param}")
    print(f"Best Validation Score: {best_score:.3f}")
    
    return best_param, best_score

# Example 1: Random Forest max_depth
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=42)
depth_range = range(1, 21)

plot_validation_curve(
    rf, X, y,
    param_name='max_depth',
    param_range=depth_range,
    title='Random Forest max_depth'
)
plt.show()

# Example 2: Ridge regularization (alpha)
from sklearn.linear_model import Ridge

ridge = Ridge()
alpha_range = np.logspace(-3, 3, 20)

plot_validation_curve(
    ridge, X, y,
    param_name='alpha',
    param_range=alpha_range,
    title='Ridge Regularization (alpha)'
)
plt.xscale('log')
plt.show()

# Example 3: Polynomial degree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('linear', LinearRegression())
])

degree_range = range(1, 11)

plot_validation_curve(
    poly_pipeline, X[:, :2], y,  # Use only 2 features to see effect
    param_name='poly__degree',
    param_range=degree_range,
    title='Polynomial Degree'
)
plt.show()
```

### Interpretacija Validation Curves:
```
Left Side (Low parameter value):
├─ Both train and validation LOW → Underfitting
└─ Increase parameter

Sweet Spot (Middle):
├─ Both train and validation HIGH, small gap → Optimal!
└─ This is your best parameter!

Right Side (High parameter value):
├─ Train HIGH, Validation drops → Overfitting
└─ Decrease parameter
```

---

## 8. Model Complexity Spectrum
```
MODEL COMPLEXITY SPECTRUM:

Simple                                           Complex
│                                                      │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│  Linear  │  Poly 2  │  Poly 5  │  Poly 10 │ Poly 20 │
│          │          │          │          │         │
│ HIGH     │  Medium  │   Low    │  Medium  │  HIGH   │
│ BIAS     │   Bias   │   Bias   │   Bias   │  BIAS   │
│          │          │          │          │         │
│  LOW     │  Medium  │   Low    │  Medium  │  HIGH   │
│ VARIANCE │ Variance │ Variance │ Variance │VARIANCE │
│          │          │          │          │         │
│Underfit  │          │ OPTIMAL! │          │Overfit  │
└──────────┴──────────┴──────────┴──────────┴─────────┘
```

### Different Model Types - Complexity:
```python
# Complexity ranking (for same data)

models_by_complexity = [
    # Simple (High Bias, Low Variance)
    ('Linear Regression', LinearRegression()),
    ('Logistic Regression (C=0.01)', LogisticRegression(C=0.01, max_iter=1000)),
    
    # Medium-Low
    ('Ridge (alpha=10)', Ridge(alpha=10)),
    ('Decision Tree (max_depth=3)', DecisionTreeRegressor(max_depth=3)),
    
    # Medium
    ('Random Forest (max_depth=10)', RandomForestRegressor(n_estimators=50, max_depth=10)),
    ('SVM (RBF, C=1)', SVR(kernel='rbf', C=1)),
    
    # Medium-High
    ('Random Forest (max_depth=20)', RandomForestRegressor(n_estimators=100, max_depth=20)),
    ('XGBoost (default)', XGBRegressor()),
    
    # Complex (Low Bias, High Variance)
    ('Random Forest (no limit)', RandomForestRegressor(n_estimators=100)),
    ('Neural Network (deep)', MLPRegressor(hidden_layers=(100, 100, 100))),
    ('Polynomial (degree=20)', Pipeline([
        ('poly', PolynomialFeatures(degree=20)),
        ('linear', LinearRegression())
    ]))
]

# Controlling complexity:
# - Linear Models: Regularization (alpha, C)
# - Tree Models: max_depth, min_samples_split
# - Ensemble: n_estimators, max_depth
# - Neural Networks: layers, neurons, dropout, L2
```

---

## Decision Framework - Dijagnoza Problema
```python
def diagnose_model(train_score, val_score, threshold_low=0.7, threshold_gap=0.1):
    """
    Dijagnostikuj model problem.
    
    Parameters:
    - train_score: Score na train set-u
    - val_score: Score na validation set-u
    - threshold_low: Threshold za "low" score
    - threshold_gap: Threshold za "large" gap
    """
    gap = train_score - val_score
    
    print("="*60)
    print("MODEL DIAGNOSIS")
    print("="*60)
    print(f"Train Score:      {train_score:.3f}")
    print(f"Validation Score: {val_score:.3f}")
    print(f"Gap:              {gap:.3f}")
    print("")
    
    # Underfitting (High Bias)
    if train_score < threshold_low and val_score < threshold_low and gap < threshold_gap:
        print("🚨 DIAGNOSIS: UNDERFITTING (High Bias)")
        print("")
        print("SYMPTOMS:")
        print("  • Both train and validation scores are LOW")
        print("  • Small gap between train and validation")
        print("  • Model is too simple for the data")
        print("")
        print("SOLUTIONS:")
        print("  ✅ Increase model complexity")
        print("     - Higher polynomial degree")
        print("     - Deeper trees (increase max_depth)")
        print("     - More layers/neurons in Neural Network")
        print("  ✅ Add more features")
        print("     - Feature engineering")
        print("     - Polynomial features")
        print("     - Interaction terms")
        print("  ✅ Reduce regularization")
        print("     - Decrease alpha (Ridge/Lasso)")
        print("     - Increase C (SVM/Logistic)")
        print("  ✅ Train longer")
        print("     - More iterations/epochs")
        print("  ✅ Try different algorithm")
        print("     - Non-linear models for non-linear data")
    
    # Overfitting (High Variance)
    elif train_score > 0.9 and gap > threshold_gap:
        print("🚨 DIAGNOSIS: OVERFITTING (High Variance)")
        print("")
        print("SYMPTOMS:")
        print("  • Train score is VERY HIGH")
        print("  • Validation score is MUCH LOWER")
        print("  • Large gap between train and validation")
        print("  • Model is memorizing training data")
        print("")
        print("SOLUTIONS:")
        print("  ✅ Get more training data")
        print("     - Collect more samples")
        print("     - Data augmentation (for images/text)")
        print("  ✅ Reduce model complexity")
        print("     - Lower polynomial degree")
        print("     - Shallower trees (decrease max_depth)")
        print("     - Fewer layers/neurons")
        print("  ✅ Add regularization")
        print("     - Increase alpha (Ridge/Lasso)")
        print("     - Decrease C (SVM/Logistic)")
        print("     - Add dropout (Neural Networks)")
        print("  ✅ Feature selection")
        print("     - Remove irrelevant features")
        print("     - SelectKBest, RFE")
        print("  ✅ Cross-validation")
        print("     - Use K-Fold CV for robust evaluation")
        print("  ✅ Early stopping")
        print("     - Stop training before memorization")
        print("  ✅ Ensemble methods")
        print("     - Bagging to reduce variance")
    
    # Good Fit
    elif train_score >= threshold_low and val_score >= threshold_low and gap < threshold_gap:
        print("✅ DIAGNOSIS: GOOD FIT!")
        print("")
        print("GREAT NEWS:")
        print("  • Both train and validation scores are HIGH")
        print("  • Small gap between train and validation")
        print("  • Model generalizes well")
        print("")
        print("NEXT STEPS:")
        print("  • Test on final test set")
        print("  • Consider hyperparameter tuning for minor improvements")
        print("  • Deploy to production!")
    
    # Moderate Overfitting
    elif gap > threshold_gap:
        print("⚠️  DIAGNOSIS: MODERATE OVERFITTING")
        print("")
        print("SYMPTOMS:")
        print("  • Noticeable gap between train and validation")
        print("  • Model could generalize better")
        print("")
        print("SOLUTIONS:")
        print("  • Try regularization (mild)")
        print("  • Cross-validation")
        print("  • Monitor learning curves")
    
    else:
        print("📊 DIAGNOSIS: UNCLEAR")
        print("")
        print("SUGGESTIONS:")
        print("  • Check learning curves")
        print("  • Try cross-validation")
        print("  • Inspect residuals/errors")
    
    print("="*60)

# Example usage
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=500, n_features=20, noise=10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Model 1: Too simple (underfitting)
model1 = LinearRegression()
model1.fit(X_train, y_train)
diagnose_model(
    model1.score(X_train, y_train),
    model1.score(X_val, y_val)
)

# Model 2: Too complex (overfitting)
model2 = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2)
model2.fit(X_train, y_train)
diagnose_model(
    model2.score(X_train, y_train),
    model2.score(X_val, y_val)
)

# Model 3: Good fit
model3 = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=10)
model3.fit(X_train, y_train)
diagnose_model(
    model3.score(X_train, y_train),
    model3.score(X_val, y_val)
)
```

---

## Complete Example - Bias-Variance Analysis
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 1. GENERATE DATA ====================
np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

# Add some non-linearity
X = X.ravel()
y = y + 0.5 * X**2

X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y, test_size=0.3, random_state=42
)

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")

# ==================== 2. MODELS WITH DIFFERENT COMPLEXITY ====================
models = {
    'Linear (High Bias)': Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('linear', LinearRegression())
    ]),
    'Polynomial deg=2 (Good)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    'Polynomial deg=3': Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ]),
    'Polynomial deg=5': Pipeline([
        ('poly', PolynomialFeatures(degree=5)),
        ('linear', LinearRegression())
    ]),
    'Polynomial deg=10 (High Variance)': Pipeline([
        ('poly', PolynomialFeatures(degree=10)),
        ('linear', LinearRegression())
    ])
}

# ==================== 3. TRAIN AND EVALUATE ====================
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Gap': train_r2 - test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })

# Results DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# ==================== 4. VISUALIZE PREDICTIONS ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    # Data
    ax.scatter(X_train, y_train, alpha=0.5, s=30, label='Train')
    ax.scatter(X_test, y_test, alpha=0.5, s=30, label='Test')
    
    # Prediction
    y_plot = model.predict(X_plot)
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    
    # Scores
    train_r2 = results_df.loc[results_df['Model'] == name, 'Train R²'].values[0]
    test_r2 = results_df.loc[results_df['Model'] == name, 'Test R²'].values[0]
    gap = results_df.loc[results_df['Model'] == name, 'Gap'].values[0]
    
    ax.set_title(f'{name}\nTrain R²={train_r2:.2f}, Test R²={test_r2:.2f}, Gap={gap:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bias_variance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 5. LEARNING CURVES ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx]
    
    # Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    ax.plot(train_sizes, train_mean, 'o-', label='Train')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    ax.plot(train_sizes, val_mean, 's-', label='Validation')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    ax.set_title(f'Learning Curve - {name}')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 6. VALIDATION CURVE (POLYNOMIAL DEGREE) ====================
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('linear', LinearRegression())
])

degree_range = range(1, 16)

train_scores, val_scores = validation_curve(
    poly_pipeline, X_train, y_train,
    param_name='poly__degree',
    param_range=degree_range,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Plot
plt.figure(figsize=(12, 6))

plt.plot(degree_range, train_mean, 'o-', label='Train Score', linewidth=2)
plt.fill_between(degree_range, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(degree_range, val_mean, 's-', label='Validation Score', linewidth=2)
plt.fill_between(degree_range, val_mean - val_std, val_mean + val_std, alpha=0.2)

# Optimal point
best_degree = degree_range[np.argmax(val_mean)]
best_score = val_mean[np.argmax(val_mean)]

plt.axvline(best_degree, color='green', linestyle='--', alpha=0.7,
            label=f'Optimal Degree = {best_degree}')
plt.scatter([best_degree], [best_score], color='green', s=150, zorder=5)

# Regions
plt.axvspan(1, 2, alpha=0.1, color='red', label='Underfitting Region')
plt.axvspan(8, 15, alpha=0.1, color='orange', label='Overfitting Region')

plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Validation Curve - Polynomial Degree\n(Bias-Variance Tradeoff)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(degree_range)

plt.savefig('validation_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Optimal Polynomial Degree: {best_degree}")
print(f"✅ Best Validation R²: {best_score:.3f}")

# ==================== 7. FINAL DIAGNOSIS ====================
print("\n" + "="*80)
print("FINAL MODEL DIAGNOSIS")
print("="*80)

for _, row in results_df.iterrows():
    print(f"\n{row['Model']}:")
    
    if row['Train R²'] < 0.7 and row['Test R²'] < 0.7:
        print("  🚨 UNDERFITTING - Model too simple!")
    elif row['Train R²'] > 0.9 and row['Gap'] > 0.2:
        print("  🚨 OVERFITTING - Model too complex!")
    elif row['Gap'] < 0.1 and row['Test R²'] > 0.7:
        print("  ✅ GOOD FIT - Well balanced!")
    else:
        print("  ⚠️  MODERATE FIT - Could be improved")
```

---

## Rezime - Bias-Variance Tradeoff

### Quick Reference:

| Problem | Symptoms | Solutions |
|---------|----------|-----------|
| **UNDERFITTING (High Bias)** | Train LOW, Test LOW, Gap SMALL | ↑ Complexity, ↑ Features, ↓ Regularization |
| **OVERFITTING (High Variance)** | Train HIGH, Test LOW, Gap LARGE | ↓ Complexity, ↑ Data, ↑ Regularization |
| **GOOD FIT** | Train HIGH, Test HIGH, Gap SMALL | Deploy! Minor tuning. |

### Decision Flowchart:
```
Check train and test scores
│
├─→ Both LOW, small gap?
│   └─→ UNDERFITTING
│       ├─ Increase complexity
│       ├─ Add features
│       └─ Reduce regularization
│
├─→ Train HIGH, Test LOW, large gap?
│   └─→ OVERFITTING
│       ├─ Get more data
│       ├─ Reduce complexity
│       ├─ Add regularization
│       └─ Feature selection
│
└─→ Both HIGH, small gap?
    └─→ GOOD FIT ✅
        └─ Deploy!
```

### Key Metrics:
```python
# Check these metrics:
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
gap = train_score - test_score

# Decision thresholds:
LOW_THRESHOLD = 0.7     # Scores below this are "low"
GAP_THRESHOLD = 0.1     # Gap above this is "large"

# Diagnosis:
if train_score < LOW_THRESHOLD and test_score < LOW_THRESHOLD:
    print("UNDERFITTING")
elif train_score > 0.9 and gap > GAP_THRESHOLD:
    print("OVERFITTING")
elif gap < GAP_THRESHOLD and test_score > LOW_THRESHOLD:
    print("GOOD FIT")
```

**Key Takeaway:** Bias-Variance Tradeoff je srce machine learning-a! Cilj nije savršen train score - cilj je balans između bias-a i variance-a za najbolju generalizaciju. Learning curves i validation curves su tvoji najbolji alati za dijagnozu. Ne pokušavaj postići 100% train accuracy - to je put ka overfitting-u! 🎯