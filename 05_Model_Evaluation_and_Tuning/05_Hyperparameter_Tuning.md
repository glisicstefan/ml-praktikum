# Hyperparameter Tuning

Hyperparameter Tuning je proces **pronalaženja optimalnih vrednosti hiperparametara** modela za najbolje performanse. Različiti hiperparametri drastično utiču na bias-variance tradeoff i generalizaciju modela.

**Zašto je hyperparameter tuning kritičan?**
- **Default parametri retko optimalni** - sklearn defaults su safe, ali ne najbolji
- **Ogromna razlika u performansama** - Tuned model može biti 10-20% bolji!
- **Model selection** - Isti algoritam sa različitim parametrima = različit model
- **Systematičan pristup** - Umesto nasumičnih pokušaja
- **Production readiness** - Squeezovati svaki procenat performanse

**VAŽNO:** Hyperparameter tuning mora biti na TRAIN setu (ili sa nested CV) - nikad direktno na test setu!

---

## Hyperparameters vs Parameters

### Razlika:
```
PARAMETERS (Model Parameters):
├─ Naučeni tokom treniranja
├─ Optimizovani algoritmom (gradient descent, itd.)
├─ Primer: Weights i biases u Neural Network, coeficijenti u Linear Regression
└─ Ne postavljaš ih ručno!

HYPERPARAMETERS:
├─ Postavljeni PRE treniranja
├─ NE naučeni tokom treniranja
├─ Kontrolišu proces učenja i model kompleksnost
├─ Primer: learning_rate, n_estimators, max_depth
└─ TI ih postavljaš!
```

### Primeri:
```python
# Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# PARAMETERS (naučeni):
print(f"Coefficients: {model.coef_}")      # Learned weights
print(f"Intercept: {model.intercept_}")    # Learned bias

# HYPERPARAMETERS: Nema (ili fit_intercept=True/False)

# -----------------------------------------------------------

# Ridge Regression
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # ← HYPERPARAMETER (postavljaš TI)
model.fit(X_train, y_train)

# PARAMETERS (naučeni):
print(f"Coefficients: {model.coef_}")      # Learned weights

# -----------------------------------------------------------

# Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # ← HYPERPARAMETER
    max_depth=10,        # ← HYPERPARAMETER
    min_samples_split=5, # ← HYPERPARAMETER
    random_state=42
)
model.fit(X_train, y_train)

# PARAMETERS (naučeni):
# Tree structure, split points, leaf values - sve automatski naučeno!

# -----------------------------------------------------------

# Neural Network
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # ← HYPERPARAMETER
    learning_rate_init=0.001,      # ← HYPERPARAMETER
    alpha=0.0001,                  # ← HYPERPARAMETER (L2 penalty)
    random_state=42
)
model.fit(X_train, y_train)

# PARAMETERS (naučeni):
# Weights and biases between all layers
```

---

## 1. Grid Search (Exhaustive Search)

**Grid Search** testira **SVE kombinacije** hiperparametara iz definisanog grid-a.

### Kako Radi:
```
Hyperparameters:
├─ max_depth: [5, 10, 20]
└─ n_estimators: [50, 100, 200]

Grid = Kartezijanski proizvod:
(5, 50), (5, 100), (5, 200),
(10, 50), (10, 100), (10, 200),
(20, 50), (20, 100), (20, 200)

= 3 × 3 = 9 kombinacija

Za svaku kombinaciju:
├─ 5-fold CV → 5 treniranja
└─ Total: 9 × 5 = 45 treniranja!
```

### Python Implementacija:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42)

# Model
rf = RandomForestClassifier(random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                    # 5-fold CV
    scoring='accuracy',      # Metric za optimizaciju
    n_jobs=-1,               # Use all cores
    verbose=2,               # Progress updates
    return_train_score=True  # Track overfitting
)

# Fit (ovo može trajati!)
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

# Results
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")

# Test set performance
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")

# Total combinations tested
print(f"\nTotal combinations: {len(grid_search.cv_results_['params'])}")
print(f"Total fits: {len(grid_search.cv_results_['params']) * 5}")  # × CV folds
```

### Analyzing Results:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Results DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Top 10 combinations
top_results = results_df.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
print("\nTop 10 Parameter Combinations:")
print(top_results)

# Visualize: max_depth vs n_estimators (fixing other params)
# Filter results for specific values
filtered = results_df[
    (results_df['param_min_samples_split'] == 2) &
    (results_df['param_min_samples_leaf'] == 1)
]

# Pivot table
pivot = filtered.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators'
)

# Heatmap
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu')
plt.title('Grid Search Results\nAccuracy by max_depth and n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.show()
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Garantovano nalazi najbolju kombinaciju** (od testiranih)
- ✅ **Jednostavan za razumevanje**
- ✅ **Comprehensive** - testira sve

**Mane:**
- ❌ **VEOMA SPOR** - Broj kombinacija eksponencijalno raste!
- ❌ **Curse of dimensionality** - 5 parametara × 5 vrednosti = 3,125 kombinacija!
- ❌ **Neefektivno** - Testira i loše regione parameter prostora

---

## 2. Random Search

**Random Search** **random-izovano sampla** kombinacije iz definisanih distribucija.

### Kako Radi:
```
Umesto testiranja SVE kombinacije:
├─ Sample random kombinacije
├─ Test njih (npr. 50 random kombinacija)
└─ Vrati najbolju

Prednost:
Bolje pokriva prostor parametara sa manje testova!
```

### Zašto je Random Search Bolji?
```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstracija: 1D optimization (jedna "prava" optimalna vrednost)
np.random.seed(42)

# True optimum at x=0.7
def objective(x):
    return -((x - 0.7) ** 2) + 1.0

# Grid Search: 9 equally spaced points
grid_points = np.linspace(0, 1, 9)
grid_scores = [objective(x) for x in grid_points]

# Random Search: 9 random points
random_points = np.random.uniform(0, 1, 9)
random_scores = [objective(x) for x in random_points]

# Plot
x_plot = np.linspace(0, 1, 100)
y_plot = [objective(x) for x in x_plot]

plt.figure(figsize=(14, 5))

# Grid Search
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='True Function')
plt.scatter(grid_points, grid_scores, color='red', s=100, zorder=5, label='Grid Points')
plt.axvline(0.7, color='green', linestyle='--', label='True Optimum')
best_grid = grid_points[np.argmax(grid_scores)]
plt.axvline(best_grid, color='orange', linestyle='--', label=f'Best Found: {best_grid:.2f}')
plt.xlabel('Parameter Value')
plt.ylabel('Score')
plt.title('Grid Search (9 points)\nRegularly Spaced')
plt.legend()
plt.grid(True, alpha=0.3)

# Random Search
plt.subplot(1, 2, 2)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='True Function')
plt.scatter(random_points, random_scores, color='red', s=100, zorder=5, label='Random Points')
plt.axvline(0.7, color='green', linestyle='--', label='True Optimum')
best_random = random_points[np.argmax(random_scores)]
plt.axvline(best_random, color='orange', linestyle='--', label=f'Best Found: {best_random:.2f}')
plt.xlabel('Parameter Value')
plt.ylabel('Score')
plt.title('Random Search (9 points)\nRandomly Sampled')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Grid Search error: {abs(best_grid - 0.7):.3f}")
print(f"Random Search error: {abs(best_random - 0.7):.3f}")
# Random Search često bliže optimumu!
```

### Python Implementacija:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Parameter distributions (not fixed grid!)
param_distributions = {
    'n_estimators': randint(50, 300),        # Uniform int between 50-300
    'max_depth': randint(5, 50),             # Uniform int between 5-50
    'min_samples_split': randint(2, 20),     # 2-20
    'min_samples_leaf': randint(1, 10),      # 1-10
    'max_features': uniform(0.1, 0.9)        # Uniform float 0.1-1.0
}

# Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,           # Number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

# Fit
print("Starting Random Search...")
random_search.fit(X_train, y_train)

# Results
print(f"\nBest Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.3f}")

# Test
test_score = random_search.best_estimator_.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")

print(f"\nTotal combinations tested: {random_search.n_iter}")
print(f"Total fits: {random_search.n_iter * 5}")  # Much less than Grid!
```

### Distributions:
```python
from scipy.stats import randint, uniform, loguniform

# Common distributions:

# 1. Uniform Integer (discrete)
param_int = randint(10, 100)  # Uniform between 10-100
# Samples: 10, 23, 45, 67, 89, ...

# 2. Uniform Float (continuous)
param_float = uniform(0.0, 1.0)  # Uniform between 0.0-1.0
# Samples: 0.234, 0.567, 0.891, ...

# 3. Log-Uniform (for learning rates, regularization)
param_log = loguniform(1e-4, 1e-1)  # Log-scale between 0.0001-0.1
# Samples: 0.0001, 0.001, 0.01, 0.05, ... (more dense at lower values)

# 4. Custom choices
param_choice = ['auto', 'sqrt', 'log2']
# Samples: random choice from list

# Example: Full parameter space
param_distributions = {
    # Tree structure
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    
    # Features
    'max_features': ['sqrt', 'log2', None],
    
    # Regularization (if applicable)
    'alpha': loguniform(1e-4, 1e0),  # Log-scale for regularization
    
    # Learning (for boosting)
    'learning_rate': loguniform(1e-3, 1e0)  # 0.001 to 1.0 on log scale
}
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Mnogo brži** od Grid Search (isti broj testova, bolji coverage)
- ✅ **Bolje pokriva prostor** - Testira više vrednosti po parametru
- ✅ **Scalable** - Jednostavno dodaj više iterations

**Mane:**
- ❌ **Ne garantuje optimum** - Random sampling
- ❌ **Može propustiti dobar region** - Zavisi od sreće

---

## 3. Bayesian Optimization

**Bayesian Optimization** koristi **prethodne rezultate** da inteligentno bira sledeće kombinacije za testiranje.

### Kako Radi:
```
1. Kreiraj probabilistički model (Gaussian Process) performance-a
2. Sample nekoliko random tačaka (initial exploration)
3. FIT model na dosadašnjim rezultatima
4. Koristi "acquisition function" da izabereš sledeću najbolju tačku:
   - High expected performance (exploitation)
   - High uncertainty (exploration)
5. Test tu kombinaciju
6. Update model
7. Repeat 3-6
```

### Python - BayesSearchCV (scikit-optimize):
```python
# pip install scikit-optimize

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Define search space
search_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['sqrt', 'log2', None]),
    'learning_rate': Real(0.01, 1.0, prior='log-uniform')  # For GradientBoosting
}

# Bayesian Search
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=50,           # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit
print("Starting Bayesian Optimization...")
bayes_search.fit(X_train, y_train)

# Results
print(f"\nBest Parameters: {bayes_search.best_params_}")
print(f"Best CV Score: {bayes_search.best_score_:.3f}")

# Test
test_score = bayes_search.best_estimator_.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Inteligentna pretraga** - Uči iz prethodnih rezultata
- ✅ **Brži od Random** - Fokusira se na obećavajuće regione
- ✅ **Exploration + Exploitation** - Balans između poznatog i nepoznatog

**Mane:**
- ❌ **Složeniji setup** - Zahteva dodatne biblioteke
- ❌ **Overhead** - Model fitting između testova
- ❌ **Može zaglaviti** u local optima

---

## 4. Optuna (Moderna Biblioteka)

**Optuna** je **state-of-the-art** framework za hyperparameter optimization.

### Instalacija:
```bash
pip install optuna
```

### Osnovni Primer:
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define objective function
def objective(trial):
    """
    Optuna poziva ovu funkciju za svaki trial.
    Trial sugeriše parametre, mi vraćamo score.
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    
    # Create model
    model = RandomForestClassifier(**params)
    
    # Cross-validation score
    score = cross_val_score(model, X_train, y_train, cv=5, 
                           scoring='accuracy', n_jobs=-1).mean()
    
    return score

# Create study
study = optuna.create_study(
    direction='maximize',  # Maximize accuracy
    sampler=optuna.samplers.TPESampler(seed=42)  # Tree-structured Parzen Estimator
)

# Optimize
print("Starting Optuna Optimization...")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Best results
print(f"\nBest Parameters: {study.best_params}")
print(f"Best CV Score: {study.best_value:.3f}")

# Train final model with best params
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

# Test
test_score = best_model.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")
```

### Optuna Visualizations:
```python
# Visualization (requires plotly)
# pip install plotly

import optuna.visualization as vis

# 1. Optimization History
fig1 = vis.plot_optimization_history(study)
fig1.show()

# 2. Parameter Importances
fig2 = vis.plot_param_importances(study)
fig2.show()

# 3. Parallel Coordinate Plot
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()

# 4. Contour Plot (pairwise interactions)
fig4 = vis.plot_contour(study, params=['n_estimators', 'max_depth'])
fig4.show()

# 5. Slice Plot (effect of single parameter)
fig5 = vis.plot_slice(study)
fig5.show()
```

### Advanced Optuna Features:
```python
# 1. Pruning - Stop unpromising trials early
import optuna
from optuna.pruners import MedianPruner

study_pruned = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

def objective_with_pruning(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
    }
    
    model = RandomForestClassifier(**params, random_state=42)
    
    # Report intermediate values (for pruning)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for step, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_val, y_fold_val)
        
        # Report intermediate score
        trial.report(score, step)
        
        # Prune if unpromising
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return score

study_pruned.optimize(objective_with_pruning, n_trials=50)

# 2. Multi-objective Optimization
study_multi = optuna.create_study(
    directions=['maximize', 'minimize']  # Maximize accuracy, minimize model size
)

def multi_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
    }
    
    model = RandomForestClassifier(**params, random_state=42)
    
    # Objective 1: Accuracy
    accuracy = cross_val_score(model, X_train, y_train, cv=3).mean()
    
    # Objective 2: Model complexity (number of parameters)
    complexity = params['n_estimators'] * (2 ** params['max_depth'])
    
    return accuracy, complexity

# Optimize
study_multi.optimize(multi_objective, n_trials=30)

# Pareto front
print("\nPareto Front (non-dominated solutions):")
for trial in study_multi.best_trials:
    print(f"Accuracy: {trial.values[0]:.3f}, Complexity: {trial.values[1]:.0f}")
    print(f"  Params: {trial.params}")
```

### Prednosti Optuna:

- ✅ **Modern i brz** - State-of-the-art algoritmi (TPE, CMA-ES)
- ✅ **Pruning** - Zaustavlja loše trial-e rano
- ✅ **Multi-objective** - Optimizuj više metrika odjednom
- ✅ **Excellent visualizations** - Built-in plotly grafikoni
- ✅ **Easy to use** - Jednostavniji API od drugih
- ✅ **Parallelization** - Multi-process/distributed optimization

---

## 5. Halving Grid/Random Search (sklearn 1.0+)

**Successive Halving** - iterativno eliminiše loše kombinacije sa više podataka.

### Kako Radi:
```
Start: Testira SVE kombinacije sa MALO data
│
├─→ Eliminiši najgorih 50%
├─→ Ostale testira sa 2× više data
│
├─→ Eliminiši najgorih 50%
├─→ Ostale testira sa 2× više data
│
└─→ Repeat dok ne ostane najbolja kombinacija
```

### Python:
```python
from sklearn.experimental import enable_halving_search_cv  # Explicitly import
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

# Halving Grid Search
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20]
}

halving_grid = HalvingGridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    factor=2,            # Eliminate half each iteration
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit
halving_grid.fit(X_train, y_train)

print(f"Best Parameters: {halving_grid.best_params_}")
print(f"Best Score: {halving_grid.best_score_:.3f}")

# Halving Random Search
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
}

halving_random = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_candidates=50,     # Initial number of candidates
    factor=2,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

halving_random.fit(X_train, y_train)

print(f"Best Parameters: {halving_random.best_params_}")
print(f"Best Score: {halving_random.best_score_:.3f}")
```

### Prednosti:

- ✅ **Mnogo brži** - Eliminiše loše rano sa manje podataka
- ✅ **Resource efficient** - Ne troši vreme na očigledno loše
- ✅ **Scalable** - Dobro radi sa velikim grid-ovima

---

## 6. Hyperparameters za Različite Algoritme

### Random Forest:
```python
# Important parameters (uticaj na performance)
param_grid_rf = {
    # TREE STRUCTURE (najveći uticaj!)
    'n_estimators': [50, 100, 200, 300],      # More trees = better (but slower)
    'max_depth': [10, 20, 30, None],          # None = unlimited (might overfit)
    'min_samples_split': [2, 5, 10],          # Higher = less overfitting
    'min_samples_leaf': [1, 2, 4],            # Higher = smoother decision boundary
    
    # FEATURES
    'max_features': ['sqrt', 'log2', None],   # 'sqrt' usually best for classification
    
    # SAMPLING
    'bootstrap': [True, False],               # True = bagging (variance ↓)
    'max_samples': [0.5, 0.7, 1.0]           # If bootstrap=True, sample size
}

# Priority tuning order:
# 1. n_estimators (set high, 100-300)
# 2. max_depth (controls complexity)
# 3. min_samples_split (regularization)
# 4. max_features ('sqrt' for classification, None for regression)
```

### XGBoost:
```python
# Important parameters
param_grid_xgb = {
    # BOOSTING
    'n_estimators': [100, 200, 300],          # More = better (with early stopping)
    'learning_rate': [0.01, 0.05, 0.1, 0.3],  # Lower = better (but slower)
    
    # TREE STRUCTURE
    'max_depth': [3, 5, 7, 9],                # Shallow trees usually better (3-7)
    'min_child_weight': [1, 3, 5],            # Higher = less overfitting
    
    # SAMPLING
    'subsample': [0.6, 0.8, 1.0],             # < 1.0 = stochastic boosting (variance ↓)
    'colsample_bytree': [0.6, 0.8, 1.0],      # Feature sampling per tree
    
    # REGULARIZATION
    'gamma': [0, 0.1, 0.3],                   # Minimum loss reduction (pruning)
    'reg_alpha': [0, 0.1, 1],                 # L1 regularization
    'reg_lambda': [1, 2, 5]                   # L2 regularization
}

# Priority tuning order:
# 1. learning_rate + n_estimators (together!)
# 2. max_depth
# 3. subsample + colsample_bytree
# 4. reg_alpha + reg_lambda (if overfitting)
```

### LightGBM:
```python
param_grid_lgb = {
    # BOOSTING
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    
    # TREE STRUCTURE
    'num_leaves': [31, 63, 127],              # 2^max_depth - 1 (usually)
    'max_depth': [-1, 10, 20],                # -1 = no limit
    'min_child_samples': [20, 30, 50],        # Like min_samples_leaf
    
    # SAMPLING
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    
    # REGULARIZATION
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}
```

### SVM:
```python
param_grid_svm = {
    # KERNEL
    'kernel': ['linear', 'rbf', 'poly'],
    
    # REGULARIZATION
    'C': [0.1, 1, 10, 100],                   # Inverse regularization (higher = less reg)
    
    # KERNEL PARAMETERS
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # RBF kernel width
    'degree': [2, 3, 4],                      # Polynomial degree
    
    # For imbalanced
    'class_weight': ['balanced', None]
}

# Note: SVM je SPOR za velike dataset-e!
# Preporuka: GridSearch samo za mali dataset (< 10k samples)
```

### Logistic Regression:
```python
param_grid_lr = {
    # REGULARIZATION
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],      # Inverse regularization
    
    # SOLVER (depends on penalty)
    'solver': ['liblinear', 'saga'],          # 'liblinear' for L1, 'saga' for all
    
    # ELASTICNET (only if penalty='elasticnet')
    'l1_ratio': [0.1, 0.5, 0.9],              # 0 = L2, 1 = L1
    
    # For imbalanced
    'class_weight': ['balanced', None]
}
```

### Neural Network (MLPClassifier):
```python
param_grid_nn = {
    # ARCHITECTURE
    'hidden_layer_sizes': [
        (50,), (100,), (50, 50), (100, 50), (100, 100, 50)
    ],
    
    # LEARNING
    'learning_rate_init': [0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    
    # REGULARIZATION
    'alpha': [0.0001, 0.001, 0.01],           # L2 penalty
    
    # OPTIMIZATION
    'solver': ['adam', 'sgd'],
    'batch_size': ['auto', 32, 64, 128],
    'max_iter': [200, 500, 1000]
}
```

---

## 7. Best Practices

### ✅ DO:

**1. Start with Random Search (Broad Exploration):**
```python
# Step 1: Random Search (broad)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=broad_distributions,
    n_iter=50,
    cv=5
)
random_search.fit(X_train, y_train)

# Step 2: Grid Search (narrow around best)
best_params = random_search.best_params_

# Narrow grid around best params
narrow_grid = {
    'n_estimators': [best_params['n_estimators'] - 20, 
                     best_params['n_estimators'],
                     best_params['n_estimators'] + 20],
    'max_depth': [best_params['max_depth'] - 2,
                  best_params['max_depth'],
                  best_params['max_depth'] + 2]
}

grid_search = GridSearchCV(estimator=model, param_grid=narrow_grid, cv=5)
grid_search.fit(X_train, y_train)
```

**2. Use Log Scale za Learning Rates/Regularization:**
```python
# ❌ LOŠE - Linear scale
learning_rates_bad = [0.0001, 0.0002, 0.0003, ..., 0.1]  # Većina između 0.0001-0.001

# ✅ DOBRO - Log scale
from scipy.stats import loguniform
learning_rates_good = loguniform(1e-4, 1e-1)
# Samples: 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1
# Bolja distribucija preko range-a!
```

**3. Use Nested CV za Unbiased Evaluation:**
```python
# ✅ DOBRO - Nested CV
from sklearn.model_selection import cross_val_score

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Inner CV: Hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=inner_cv)

# Outer CV: Unbiased performance estimation
scores = cross_val_score(grid_search, X_train, y_train, cv=outer_cv)
print(f"Nested CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

**4. Monitor Train-Test Gap (Overfitting):**
```python
grid_search = GridSearchCV(
    model, param_grid, cv=5,
    return_train_score=True  # ← Track overfitting!
)

grid_search.fit(X_train, y_train)

# Check overfitting
results = pd.DataFrame(grid_search.cv_results_)
results['gap'] = results['mean_train_score'] - results['mean_test_score']

# Flag overfitting
overfitting = results[results['gap'] > 0.1]
print(f"Combinations with overfitting (gap > 0.1): {len(overfitting)}")
```

**5. Set Budget (n_iter) Wisely:**
```python
# Rule of thumb for RandomizedSearchCV:
# n_iter = 10 × number_of_parameters

n_params = len(param_distributions)
n_iter = 10 * n_params

random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=n_iter,  # Adaptive budget
    cv=5
)
```

**6. Save Best Model:**
```python
import joblib

# After tuning
best_model = grid_search.best_estimator_

# Save
joblib.dump(best_model, 'best_model_tuned.pkl')
joblib.dump(grid_search.best_params_, 'best_params.pkl')

# Metadata
metadata = {
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'n_combinations_tested': len(grid_search.cv_results_['params']),
    'tuning_method': 'GridSearchCV'
}

import json
with open('tuning_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### ❌ DON'T:

**1. Ne Tune Direktno na Test Set:**
```python
# ❌ LOŠE - Tuning na test set!
param_grid = {'max_depth': [5, 10, 20]}
best_score = 0
best_params = None

for depth in [5, 10, 20]:
    model = RandomForestClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ← TEST SET!
    
    if score > best_score:
        best_score = score
        best_params = depth

# Problem: Indirektno overfitting na test set!

# ✅ DOBRO - Use validation set ili CV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # Only train!

# Test POSLE tuning-a
final_score = grid_search.best_estimator_.score(X_test, y_test)
```

**2. Ne Zaboravi Scale Podatke PRE Tuning-a:**
```python
# ❌ LOŠE - Tuning bez scaling-a (za SVM, NN, etc.)
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)  # X_train nije scaled!

# ✅ DOBRO - Pipeline sa scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # Scaling happens inside CV!
```

**3. Ne Prenaglasi Fine-Tuning:**
```python
# Diminishing returns sa tuning-om

# Default parameters
model_default = RandomForestClassifier()
model_default.fit(X_train, y_train)
score_default = model_default.score(X_test, y_test)
print(f"Default: {score_default:.3f}")  # 0.850

# After extensive tuning (10 hours)
# ... Grid Search with 1000 combinations ...
score_tuned = best_model.score(X_test, y_test)
print(f"Tuned: {score_tuned:.3f}")  # 0.862

# Improvement: 0.012 (1.2%) - Da li vredi 10 sati?

# Često bolje:
# - Bolja feature engineering (+5-10%)
# - Više podataka (+10-20%)
# - Bolji algoritam (+5-15%)
```

---

## Complete Example - End-to-End Tuning
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 1. LOAD DATA ====================
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y, return_counts=True)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==================== 2. BASELINE MODEL ====================
print("\n" + "="*60)
print("BASELINE MODEL (Default Parameters)")
print("="*60)

baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train, y_train)

baseline_score = baseline_rf.score(X_test, y_test)
print(f"Baseline Test Accuracy: {baseline_score:.3f}")

# ==================== 3. RANDOM SEARCH (BROAD EXPLORATION) ====================
print("\n" + "="*60)
print("STEP 1: RANDOM SEARCH (Broad Exploration)")
print("="*60)

from scipy.stats import randint, uniform

param_distributions_broad = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions_broad,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

random_search.fit(X_train, y_train)

print(f"\nBest Parameters (Random Search): {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.3f}")

random_test_score = random_search.best_estimator_.score(X_test, y_test)
print(f"Test Accuracy: {random_test_score:.3f}")
print(f"Improvement over baseline: {random_test_score - baseline_score:+.3f}")

# ==================== 4. GRID SEARCH (NARROW REFINEMENT) ====================
print("\n" + "="*60)
print("STEP 2: GRID SEARCH (Narrow Refinement)")
print("="*60)

# Build narrow grid around Random Search best params
best_random = random_search.best_params_

param_grid_narrow = {
    'n_estimators': [
        max(50, best_random['n_estimators'] - 30),
        best_random['n_estimators'],
        best_random['n_estimators'] + 30
    ],
    'max_depth': [
        max(5, best_random['max_depth'] - 3),
        best_random['max_depth'],
        min(50, best_random['max_depth'] + 3)
    ],
    'min_samples_split': [
        max(2, best_random['min_samples_split'] - 2),
        best_random['min_samples_split'],
        best_random['min_samples_split'] + 2
    ],
    'min_samples_leaf': [
        max(1, best_random['min_samples_leaf'] - 1),
        best_random['min_samples_leaf'],
        best_random['min_samples_leaf'] + 1
    ],
    'max_features': [best_random['max_features']]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_narrow,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters (Grid Search): {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")

grid_test_score = grid_search.best_estimator_.score(X_test, y_test)
print(f"Test Accuracy: {grid_test_score:.3f}")
print(f"Improvement over Random Search: {grid_test_score - random_test_score:+.3f}")

# ==================== 5. COMPARE WITH XGBOOST ====================
print("\n" + "="*60)
print("STEP 3: TRY DIFFERENT ALGORITHM (XGBoost)")
print("="*60)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_search = GridSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

print(f"\nBest Parameters (XGBoost): {xgb_search.best_params_}")
print(f"Best CV Score: {xgb_search.best_score_:.3f}")

xgb_test_score = xgb_search.best_estimator_.score(X_test, y_test)
print(f"Test Accuracy: {xgb_test_score:.3f}")

# ==================== 6. FINAL COMPARISON ====================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

results_comparison = pd.DataFrame({
    'Model': ['Baseline RF', 'Random Search RF', 'Grid Search RF', 'XGBoost Tuned'],
    'Test Accuracy': [baseline_score, random_test_score, grid_test_score, xgb_test_score],
    'Improvement': [0, random_test_score - baseline_score, 
                    grid_test_score - baseline_score, xgb_test_score - baseline_score]
})

print(results_comparison.to_string(index=False))

# Best model
best_overall = max([
    ('Baseline RF', baseline_score, baseline_rf),
    ('Random Search RF', random_test_score, random_search.best_estimator_),
    ('Grid Search RF', grid_test_score, grid_search.best_estimator_),
    ('XGBoost', xgb_test_score, xgb_search.best_estimator_)
], key=lambda x: x[1])

print(f"\n🏆 BEST MODEL: {best_overall[0]}")
print(f"   Test Accuracy: {best_overall[1]:.3f}")

# ==================== 7. DETAILED EVALUATION ====================
best_model = best_overall[2]
y_pred = best_model.predict(X_test)

print("\n" + "="*60)
print("DETAILED EVALUATION - BEST MODEL")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.title(f'Confusion Matrix - {best_overall[0]}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('tuning_confusion_matrix.png', dpi=300)
plt.show()

# ==================== 8. TUNING VISUALIZATION ====================

# Grid Search results heatmap
results_df = pd.DataFrame(grid_search.cv_results_)

# Create pivot for n_estimators vs max_depth
if 'param_n_estimators' in results_df.columns and 'param_max_depth' in results_df.columns:
    pivot = results_df.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Grid Search Results\nAccuracy by max_depth and n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.tight_layout()
    plt.savefig('tuning_heatmap.png', dpi=300)
    plt.show()

# ==================== 9. SAVE BEST MODEL ====================
import joblib

joblib.dump(best_model, 'best_model_final.pkl')

metadata = {
    'model_name': best_overall[0],
    'test_accuracy': float(best_overall[1]),
    'improvement_over_baseline': float(best_overall[1] - baseline_score),
    'best_params': str(best_model.get_params()),
    'tuning_steps': ['Random Search (50 iter)', 'Grid Search (narrow)', 'XGBoost comparison']
}

with open('tuning_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Best model saved: best_model_final.pkl")
print("✅ Metadata saved: tuning_metadata.json")
```

---

## Rezime - Hyperparameter Tuning

### Quick Comparison:

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **Manual** | Fast | Poor | Quick experiments |
| **Grid Search** | Slow | Good | Small param space (< 100 combos) |
| **Random Search** | Medium | Good | Medium param space, first pass |
| **Bayesian (BayesSearchCV)** | Medium | Very Good | Smart exploration |
| **Optuna** | Medium-Fast | Very Good | **Recommended!** Modern, flexible |
| **Halving Search** | Fast | Good | Large param space, resource-efficient |

### Default Strategy:
```
Step 1: Baseline (default params)
  ↓
Step 2: Random Search (broad, 50-100 iterations)
  ↓
Step 3: Grid Search (narrow around best from Step 2)
  ↓
Step 4: Evaluate on test set (ONCE!)
  ↓
Step 5: Deploy best model
```

### Budget Guidelines:
```
Small dataset (< 1k samples):
├─ Grid Search: 50-100 combinations
└─ Time: 5-10 minutes

Medium dataset (1k-100k):
├─ Random Search: 50-100 iterations
├─ Grid Search: 20-50 combinations (narrow)
└─ Time: 30 minutes - 2 hours

Large dataset (> 100k):
├─ Random Search: 20-50 iterations
├─ Halving Search: 50-100 initial candidates
└─ Time: 2-24 hours

Use Optuna for all cases!
```

### Key Hyperparameters by Algorithm:
```
Random Forest:
✅ n_estimators, max_depth, min_samples_split

XGBoost:
✅ learning_rate, n_estimators, max_depth, subsample

SVM:
✅ C, gamma, kernel

Neural Networks:
✅ hidden_layer_sizes, learning_rate_init, alpha

Logistic Regression:
✅ C, penalty, solver
```

**Key Takeaway:** Hyperparameter tuning može značajno poboljšati performanse (5-15%), ali ima diminishing returns. Start sa Random Search za broad exploration, refine sa Grid Search, i razmotri Bayesian/Optuna za inteligentnu pretragu. UVEK koristi nested CV za unbiased evaluation! Ne tune direktno na test set! 🎯