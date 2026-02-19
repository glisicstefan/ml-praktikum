# Model Interpretation

Model Interpretation je proces **razumevanja kako model donosi odluke** i **koji features utiču na predikcije**. Omogućava nam da objasnimo model stakeholderima, otkrimo greške, i izgradimo poverenje u sistem.

**Zašto je interpretabilnost kritična?**
- **Trust & Transparency** - Korisnici moraju razumeti zašto model donosi odluke
- **Debugging** - Otkrivanje grešaka u modelu ili podacima
- **Compliance** - Regulativa često zahteva objašnjenje (GDPR, finansije, medicina)
- **Feature engineering** - Razumevanje važnosti features vodi ka boljim features
- **Business insights** - Actionable insights iz model behavior-a

**VAŽNO:** "Black-box" modeli (Neural Networks, XGBoost) mogu biti moćni, ali moramo moći da ih interpretiramo!

---

## Interpretability Spectrum
```
INTERPRETABLE                                    BLACK-BOX
│                                                        │
├──────────┬──────────┬──────────┬──────────┬───────────┤
│  Linear  │ Decision │  Random  │ Gradient │   Deep    │
│  Reg/Log │   Tree   │  Forest  │ Boosting │  Neural   │
│          │          │          │          │  Networks │
│          │          │          │          │           │
│ Directly │ Visualize│ Feature  │ Feature  │  SHAP     │
│ readable │   tree   │importance│importance│   LIME    │
│   coefs  │          │   SHAP   │   SHAP   │    PDP    │
│          │          │   LIME   │   LIME   │           │
└──────────┴──────────┴──────────┴──────────┴───────────┘

Left: Built-in interpretability
Right: Need external interpretation methods
```

---

## 1. Feature Importance (Tree-Based Models)

**Feature Importance** za tree-based modele meri **koliko svaki feature doprinosi predikcijama** na osnovu split-ova u stablima.

### Kako Se Računa (Random Forest):
```
Za svaki feature:
├─ Za svako stablo u forest-u:
│   ├─ Za svaki split koji koristi taj feature:
│   │   └─ Izračunaj koliko taj split poboljšava criterion (Gini/Entropy)
│   └─ Saberi sve improvements za taj feature
└─ Average preko svih stabala

Result: Feature importance score (normalizovan na sum = 1.0)
```

### Python - Random Forest:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Importance (Random Forest):")
for i in range(10):  # Top 10
    print(f"{i+1}. {feature_names[indices[i]]:30s}: {importances[indices[i]]:.4f}")

# Visualization - Top 15
plt.figure(figsize=(10, 8))
top_n = 15
top_indices = indices[:top_n]

plt.barh(range(top_n), importances[top_indices], align='center')
plt.yticks(range(top_n), feature_names[top_indices])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features\n(Random Forest)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

### XGBoost Feature Importance (Multiple Types):
```python
import xgboost as xgb

# Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# XGBoost has 3 importance types!

# 1. Weight (default) - Number of times feature appears in splits
importance_weight = xgb_model.get_booster().get_score(importance_type='weight')

# 2. Gain - Average gain of splits which use the feature (BEST!)
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')

# 3. Cover - Average coverage of splits which use the feature
importance_cover = xgb_model.get_booster().get_score(importance_type='cover')

# Plot all three
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, importance, title in zip(axes, 
                                  [importance_weight, importance_gain, importance_cover],
                                  ['Weight', 'Gain (BEST)', 'Cover']):
    # Sort
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_importance[:15]]
    values = [x[1] for x in sorted_importance[:15]]
    
    ax.barh(range(len(features)), values)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title(f'XGBoost Feature Importance\n({title})')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# PREPORUKA: Use 'gain' for best quality measure!
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=15)
plt.title('XGBoost Feature Importance (Gain)')
plt.tight_layout()
plt.show()
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Brzo** - Built-in, nema dodatnih računanja
- ✅ **Globalno** - Overall feature importance
- ✅ **Jednostavno** - Jedna vrednost po feature-u

**Mane:**
- ❌ **Bias prema high-cardinality features** - Više unique vrednosti = veća importance
- ❌ **Ignoriše korelacije** - Ne vidi redundant features
- ❌ **Model-specific** - Samo za tree-based
- ❌ **Ne pokazuje direction** - Pozitivan ili negativan uticaj?

---

## 2. Permutation Importance (Model-Agnostic)

**Permutation Importance** meri koliko **performance pada** kada permutujemo (random shuffle) vrednosti jednog feature-a.

### Kako Radi:
```
Za svaki feature:
1. Izračunaj baseline performance (npr. accuracy)
2. Permutuj (shuffle) vrednosti tog feature-a u test setu
3. Izračunaj performance sa permutovanim feature-om
4. Importance = baseline - permuted_performance

Ako je feature važan → Performance će drastično pasti
Ako je feature nevažan → Performance ostaje ista
```

### Python:
```python
from sklearn.inspection import permutation_importance

# Train model (bilo koji!)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Permutation Importance
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,        # Repeat 10 times (variance estimate)
    random_state=42,
    n_jobs=-1
)

# Results
importances_mean = perm_importance.importances_mean
importances_std = perm_importance.importances_std

# Sort
indices = np.argsort(importances_mean)[::-1]

print("Permutation Importance:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]:30s}: "
          f"{importances_mean[indices[i]]:.4f} ± {importances_std[indices[i]]:.4f}")

# Visualization
plt.figure(figsize=(10, 8))
top_n = 15
top_indices = indices[:top_n]

plt.barh(range(top_n), importances_mean[top_indices], 
         xerr=importances_std[top_indices], align='center')
plt.yticks(range(top_n), feature_names[top_indices])
plt.xlabel('Permutation Importance\n(Decrease in Accuracy)')
plt.title('Top 15 Most Important Features\n(Permutation Importance)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Model-agnostic** - Radi sa bilo kojim modelom!
- ✅ **Intuitivno** - "Koliko je važan ovaj feature?"
- ✅ **Realniji** - Meri stvarni uticaj na performance

**Mane:**
- ❌ **Sporije** - Mora re-evaluate za svaki feature
- ❌ **Može varirati** - Osetljivo na random shuffle
- ❌ **Korelacije** - Može biti misleading sa korelisanim features

---

## 3. Partial Dependence Plots (PDP)

**PDP** pokazuje **marginalni efekat** jednog ili dva feature-a na predikciju modela.

### Kako Radi:
```
Za feature X1:
1. Izaberi grid vrednosti za X1 (npr. 10-100 tačaka)
2. Za svaku grid vrednost:
   ├─ Postavi X1 na tu vrednost za SVE samples
   ├─ Ostavi ostale features iste
   ├─ Predvidi
   └─ Average predikcija
3. Plot: X1 vs Average Prediction
```

### Python - 1D PDP:
```python
from sklearn.inspection import PartialDependenceDisplay

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# PDP za top 4 features
top_features = indices[:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    PartialDependenceDisplay.from_estimator(
        rf, X_train, [feature],
        feature_names=feature_names,
        ax=axes[idx],
        grid_resolution=50
    )
    axes[idx].set_title(f'PDP: {feature_names[feature]}')

plt.tight_layout()
plt.show()
```

### 2D PDP (Interactions):
```python
# 2D PDP za dva features (interaction)
fig, ax = plt.subplots(figsize=(10, 8))

feature_pair = (indices[0], indices[1])  # Top 2 features

PartialDependenceDisplay.from_estimator(
    rf, X_train, [feature_pair],
    feature_names=feature_names,
    ax=ax,
    grid_resolution=30
)

plt.title(f'2D PDP: {feature_names[feature_pair[0]]} vs {feature_names[feature_pair[1]]}')
plt.tight_layout()
plt.show()
```

### Manual PDP Computation:
```python
def manual_pdp(model, X, feature_idx, feature_name, grid_resolution=50):
    """
    Manually compute Partial Dependence Plot.
    """
    # Create grid
    feature_values = np.linspace(X[:, feature_idx].min(), 
                                 X[:, feature_idx].max(), 
                                 grid_resolution)
    
    pdp_values = []
    
    for value in feature_values:
        # Copy X
        X_temp = X.copy()
        
        # Set feature to grid value for ALL samples
        X_temp[:, feature_idx] = value
        
        # Predict
        predictions = model.predict_proba(X_temp)[:, 1]  # Class 1 probability
        
        # Average
        pdp_values.append(predictions.mean())
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, pdp_values, linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('Average Predicted Probability (Class 1)')
    plt.title(f'Partial Dependence Plot\n{feature_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return feature_values, pdp_values

# Example
feature_idx = indices[0]
manual_pdp(rf, X_train, feature_idx, feature_names[feature_idx])
```

### Interpretacija PDP:
```
PDP PATTERNS:

Linear positive slope:
  → Feature ↑ → Prediction ↑

Linear negative slope:
  → Feature ↑ → Prediction ↓

Flat line:
  → Feature has NO effect (not important)

Non-linear (curve):
  → Complex relationship (threshold, polynomial)

2D PDP with interaction:
  → Shows how two features interact
```

### Prednosti i Mane:

**Prednosti:**
- ✅ **Intuitivno** - Lako razumeti relationship
- ✅ **Model-agnostic** - Radi sa bilo kojim modelom
- ✅ **Shows direction** - Pozitivan ili negativan uticaj

**Mane:**
- ❌ **Assumes independence** - Ignoriše korelacije između features
- ❌ **Averages over all samples** - Ne vidi individualne razlike
- ❌ **Može biti misleading** - Sa korelisanim features

---

## 4. Individual Conditional Expectation (ICE)

**ICE** je kao PDP, ali **za svaki sample pojedinačno** - ne average!

### Kako Radi:
```
Za feature X1:
1. Izaberi grid vrednosti za X1
2. Za SVAKI sample:
   ├─ Postavi X1 na grid vrednosti
   ├─ Predict za svaku grid vrednost
   └─ Plot liniju za taj sample
3. Result: Mnoštvo linija (jedna po sample-u)

PDP = Average svih ICE linija
```

### Python:
```python
from sklearn.inspection import PartialDependenceDisplay

# ICE Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_feature = indices[0]

# PDP (average)
PartialDependenceDisplay.from_estimator(
    rf, X_train[:100], [top_feature],  # Use subset for clarity
    feature_names=feature_names,
    kind='average',
    ax=axes[0]
)
axes[0].set_title(f'PDP (Average)\n{feature_names[top_feature]}')

# ICE (individual lines)
PartialDependenceDisplay.from_estimator(
    rf, X_train[:100], [top_feature],
    feature_names=feature_names,
    kind='individual',  # Show individual lines
    ax=axes[1]
)
axes[1].set_title(f'ICE (Individual)\n{feature_names[top_feature]}')

plt.tight_layout()
plt.show()
```

### ICE + PDP Together:
```python
# Both together (ICE + PDP overlay)
fig, ax = plt.subplots(figsize=(10, 6))

PartialDependenceDisplay.from_estimator(
    rf, X_train[:200], [top_feature],
    feature_names=feature_names,
    kind='both',  # Show both ICE and PDP
    ax=ax
)

plt.title(f'ICE + PDP\n{feature_names[top_feature]}')
plt.tight_layout()
plt.show()
```

### Prednosti ICE:

- ✅ **Shows heterogeneity** - Vidi razlike između samples
- ✅ **Detects interactions** - Ako ICE linije nisu paralelne → Interactions postoje
- ✅ **More informative than PDP** - Ne gubi informaciju averaging-om

---

## 5. SHAP (SHapley Additive exPlanations)

**SHAP** je **trenutno najmoćnija metoda** za interpretaciju - zasnovana na game theory!

### Koncept:
```
SHAP vrednost za feature = Prosečan marginalni doprinos tog feature-a
                           preko svih mogućih kombinacija drugih features

Interpretacija: Koliko ovaj feature dodaje ili oduzima od baseline predikcije?

baseline_prediction + Σ(SHAP values) = final_prediction
```

### Python - SHAP Setup:
```bash
pip install shap
```

### SHAP - TreeExplainer (Brzo za tree-based):
```python
import shap

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values (for test set)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is list [class_0, class_1]
# We usually use class 1 (positive class)
shap_values_class1 = shap_values[1]

print(f"SHAP values shape: {shap_values_class1.shape}")
# (n_samples, n_features)
```

### SHAP Summary Plot (Global Importance):
```python
# Summary plot - Shows feature importance + impact
shap.summary_plot(shap_values_class1, X_test, 
                  feature_names=feature_names,
                  plot_type='bar')
plt.title('SHAP Feature Importance (Mean |SHAP value|)')
plt.tight_layout()
plt.show()

# Detailed summary plot (beeswarm)
shap.summary_plot(shap_values_class1, X_test, 
                  feature_names=feature_names)
plt.title('SHAP Summary Plot\n(Feature Impact on Model Output)')
plt.tight_layout()
plt.show()

# Interpretacija beeswarm plot:
# - X-axis: SHAP value (impact na output)
# - Y-axis: Features (sorted by importance)
# - Color: Feature value (red=high, blue=low)
# - Dots: Individual samples
```

### SHAP Waterfall Plot (Single Prediction):
```python
# Explain single prediction
sample_idx = 0

# Waterfall plot
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_class1[sample_idx],
        base_values=explainer.expected_value[1],
        data=X_test[sample_idx],
        feature_names=feature_names
    )
)
plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}')
plt.tight_layout()
plt.show()

# Interpretacija:
# Počinje od baseline (expected value)
# Svaki feature dodaje ili oduzima od predikcije
# Final = baseline + sum of SHAP values
```

### SHAP Force Plot (Single Prediction):
```python
# Force plot za single prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values_class1[sample_idx],
    X_test[sample_idx],
    feature_names=feature_names,
    matplotlib=True
)
plt.title(f'SHAP Force Plot - Sample {sample_idx}')
plt.tight_layout()
plt.show()

# Red features = Push prediction higher
# Blue features = Push prediction lower
```

### SHAP Dependence Plot (Feature Relationships):
```python
# Dependence plot - Like PDP but with SHAP
top_feature_idx = 0

shap.dependence_plot(
    top_feature_idx,
    shap_values_class1,
    X_test,
    feature_names=feature_names
)
plt.title(f'SHAP Dependence Plot\n{feature_names[top_feature_idx]}')
plt.tight_layout()
plt.show()

# X-axis: Feature value
# Y-axis: SHAP value (impact)
# Color: Interaction feature (auto-detected)
```

### SHAP - KernelExplainer (Model-Agnostic, Ali Spor):
```python
# KernelExplainer - Works with ANY model (but slow!)
from sklearn.neural_network import MLPClassifier

# Train NN
nn = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# Create explainer
# Use background data (sample from training set)
background = shap.sample(X_train, 100)  # Sample 100 for speed

explainer_kernel = shap.KernelExplainer(
    nn.predict_proba,
    background
)

# Calculate SHAP values (SLOW!)
# Use subset za speed
shap_values_nn = explainer_kernel.shap_values(X_test[:100])

# Plot
shap.summary_plot(shap_values_nn[1], X_test[:100], 
                  feature_names=feature_names)
plt.title('SHAP Summary - Neural Network\n(KernelExplainer)')
plt.tight_layout()
plt.show()
```

### SHAP Prednosti:

- ✅ **Theoretically sound** - Zasnovano na game theory
- ✅ **Additive** - Vrednosti se sabiru do final prediction
- ✅ **Consistent** - Ako feature više doprinosi, SHAP value je veći
- ✅ **Local + Global** - Može objasniti pojedinačne i sve predikcije
- ✅ **Direction** - Pozitivan ili negativan uticaj
- ✅ **Model-agnostic** - KernelExplainer radi sa bilo čim

---

## 6. LIME (Local Interpretable Model-agnostic Explanations)

**LIME** objašnjava **pojedinačne predikcije** fitovanjem **lokalno linearnog modela** oko te tačke.

### Kako Radi:
```
Za single prediction:
1. Generiši perturbacije (varijacije) oko tog sample-a
2. Predict sa original modelom za sve perturbacije
3. Weight perturbacije po blizini original sample-a
4. Fit simple linear model (interpretable) na perturbacije
5. Linear model objašnjava kako original model radi lokalno
```

### Python - LIME:
```bash
pip install lime
```
```python
import lime
import lime.lime_tabular

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Create LIME explainer
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Malignant', 'Benign'],
    mode='classification',
    random_state=42
)

# Explain single prediction
sample_idx = 0

explanation = explainer_lime.explain_instance(
    X_test[sample_idx],
    rf.predict_proba,
    num_features=10  # Show top 10 features
)

# Show explanation
explanation.show_in_notebook(show_table=True)

# Or plot
fig = explanation.as_pyplot_figure()
plt.title(f'LIME Explanation - Sample {sample_idx}')
plt.tight_layout()
plt.show()

# Interpretacija:
# Orange bars = Features pushing towards class 1
# Blue bars = Features pushing towards class 0
```

### LIME - Multiple Explanations:
```python
# Explain multiple samples
n_samples = 5

fig, axes = plt.subplots(n_samples, 1, figsize=(10, 15))

for i in range(n_samples):
    explanation = explainer_lime.explain_instance(
        X_test[i],
        rf.predict_proba,
        num_features=10
    )
    
    # Plot on axis
    explanation.as_pyplot_figure(ax=axes[i])
    axes[i].set_title(f'Sample {i} - True: {y_test[i]}, '
                     f'Pred: {rf.predict([X_test[i]])[0]}')

plt.tight_layout()
plt.show()
```

### LIME vs SHAP:
```python
import pandas as pd

comparison = pd.DataFrame({
    'Property': [
        'Scope',
        'Speed',
        'Theory',
        'Consistency',
        'Model Support',
        'Use Case'
    ],
    'LIME': [
        'Local (single prediction)',
        'Fast',
        'Heuristic (no theory)',
        'Can vary between runs',
        'Any model (model-agnostic)',
        'Quick local explanations'
    ],
    'SHAP': [
        'Local + Global',
        'Slower (but TreeExplainer fast)',
        'Game theory (solid foundation)',
        'Consistent',
        'Any model (KernelExplainer)',
        'Comprehensive analysis'
    ]
})

print(comparison.to_string(index=False))
```

---

## 7. Coefficients (Linear Models)

**Za linear modele**, koeficijenti **direktno pokazuju uticaj** svakog feature-a!

### Logistic Regression:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# MUST scale for interpretable coefficients!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# Coefficients
coefficients = logreg.coef_[0]

# Sort by absolute value
indices_coef = np.argsort(np.abs(coefficients))[::-1]

print("Logistic Regression Coefficients:")
for i in range(15):
    idx = indices_coef[i]
    print(f"{i+1}. {feature_names[idx]:30s}: {coefficients[idx]:+.4f}")

# Visualization
plt.figure(figsize=(10, 8))
top_n = 15
top_indices = indices_coef[:top_n]

colors = ['red' if c < 0 else 'green' for c in coefficients[top_indices]]

plt.barh(range(top_n), coefficients[top_indices], color=colors, alpha=0.7)
plt.yticks(range(top_n), feature_names[top_indices])
plt.xlabel('Coefficient Value\n(Positive = Higher probability of class 1)')
plt.title('Logistic Regression - Top 15 Features by |Coefficient|')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

### Interpretacija Coefficients:
```
Logistic Regression (scaled features):

Coefficient = +0.5
→ 1 std increase in feature → log-odds increase by 0.5
→ Odds multiply by exp(0.5) ≈ 1.65
→ ~65% increase in odds

Coefficient = -0.5
→ 1 std increase in feature → log-odds decrease by 0.5
→ Odds multiply by exp(-0.5) ≈ 0.61
→ ~39% decrease in odds

Magnitude |coefficient|:
→ Larger magnitude = stronger effect
```

### Linear Regression Coefficients:
```python
from sklearn.linear_model import LinearRegression

# For regression
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

coefficients_reg = linreg.coef_

print("Linear Regression Coefficients:")
for i in range(10):
    idx = indices_coef[i]
    print(f"{feature_names[idx]:30s}: {coefficients_reg[idx]:+.4f}")

# Interpretacija:
# Coefficient = 2.5 → 1 unit increase in feature → prediction increases by 2.5
```

---

## Decision Framework - Koja Metoda Kada?
```python
def recommend_interpretation_method(model_type, goal, data_size):
    """
    Preporuči interpretation method.
    
    Parameters:
    - model_type: 'linear', 'tree', 'ensemble', 'neural', 'other'
    - goal: 'global' (overall), 'local' (single prediction), 'both'
    - data_size: 'small' (<1k), 'medium' (1k-100k), 'large' (>100k)
    """
    print("="*60)
    print("INTERPRETATION METHOD RECOMMENDATION")
    print("="*60)
    print(f"Model Type: {model_type}")
    print(f"Goal: {goal}")
    print(f"Data Size: {data_size}")
    print("")
    
    recommendations = []
    
    # Linear Models
    if model_type == 'linear':
        print("📊 LINEAR MODEL - Inherently Interpretable!")
        recommendations.append("✅ Coefficients (built-in)")
        if goal in ['global', 'both']:
            recommendations.append("✅ Permutation Importance (global)")
        if goal in ['local', 'both']:
            recommendations.append("✅ LIME (local explanations)")
    
    # Tree-based
    elif model_type == 'tree':
        print("🌲 TREE-BASED MODEL")
        if goal in ['global', 'both']:
            recommendations.append("✅ Feature Importance (built-in, fast)")
            recommendations.append("✅ Permutation Importance (more reliable)")
            recommendations.append("✅ SHAP TreeExplainer (best, comprehensive)")
            recommendations.append("✅ PDP (global relationships)")
        if goal in ['local', 'both']:
            recommendations.append("✅ SHAP Force Plot (individual)")
            recommendations.append("✅ LIME (alternative)")
    
    # Ensemble
    elif model_type == 'ensemble':
        print("🎯 ENSEMBLE MODEL (RF, XGBoost, etc.)")
        if goal in ['global', 'both']:
            recommendations.append("✅ SHAP Summary Plot (BEST!)")
            recommendations.append("✅ Permutation Importance")
            recommendations.append("✅ PDP / ICE")
        if goal in ['local', 'both']:
            recommendations.append("✅ SHAP Waterfall/Force Plot")
            recommendations.append("✅ LIME")
    
    # Neural Networks
    elif model_type == 'neural':
        print("🧠 NEURAL NETWORK")
        if goal in ['global', 'both']:
            recommendations.append("✅ Permutation Importance")
            recommendations.append("✅ SHAP (KernelExplainer - slow!)")
            recommendations.append("✅ PDP")
        if goal in ['local', 'both']:
            recommendations.append("✅ LIME (faster than SHAP)")
            recommendations.append("✅ SHAP (if time permits)")
    
    # Other
    else:
        print("❓ OTHER MODEL TYPE")
        recommendations.append("✅ Permutation Importance (model-agnostic)")
        recommendations.append("✅ PDP (global)")
        recommendations.append("✅ LIME (local)")
        if data_size != 'large':
            recommendations.append("✅ SHAP KernelExplainer (if time permits)")
    
    print("RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # Speed considerations
    print("\n⏱️  SPEED CONSIDERATIONS:")
    if data_size == 'large':
        print("  • Large dataset - Avoid SHAP KernelExplainer")
        print("  • Use Feature Importance, Permutation Importance")
        print("  • Sample data for LIME/SHAP")
    elif data_size == 'medium':
        print("  • SHAP TreeExplainer: Fast")
        print("  • SHAP KernelExplainer: Slow (use sampling)")
    else:
        print("  • All methods feasible")
    
    print("="*60)

# Examples
recommend_interpretation_method('tree', 'global', 'medium')
print("\n")
recommend_interpretation_method('neural', 'local', 'large')
print("\n")
recommend_interpretation_method('linear', 'both', 'small')
```

---

## Complete Example - Comprehensive Interpretation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap

# ==================== 1. LOAD & PREPARE DATA ====================
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target: {cancer.target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 2. TRAIN MODEL ====================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Performance
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ==================== 3. FEATURE IMPORTANCE (Built-in) ====================
print("\n" + "="*60)
print("1. FEATURE IMPORTANCE (Built-in)")
print("="*60)

importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

print("\nTop 10 Features:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices_rf[i]]:30s}: {importances_rf[indices_rf[i]]:.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 Feature Importance
ax = axes[0, 0]
top_n = 15
top_indices = indices_rf[:top_n]
ax.barh(range(top_n), importances_rf[top_indices])
ax.set_yticks(range(top_n))
ax.set_yticklabels(feature_names[top_indices])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (Random Forest)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# ==================== 4. PERMUTATION IMPORTANCE ====================
print("\n" + "="*60)
print("2. PERMUTATION IMPORTANCE")
print("="*60)

from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf, X_test_scaled, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importances_perm = perm_importance.importances_mean
indices_perm = np.argsort(importances_perm)[::-1]

print("\nTop 10 Features:")
for i in range(10):
    print(f"{i+1}. {feature_names[indices_perm[i]]:30s}: "
          f"{importances_perm[indices_perm[i]]:.4f}")

# Plot
ax = axes[0, 1]
top_indices_perm = indices_perm[:top_n]
ax.barh(range(top_n), importances_perm[top_indices_perm],
        xerr=perm_importance.importances_std[top_indices_perm])
ax.set_yticks(range(top_n))
ax.set_yticklabels(feature_names[top_indices_perm])
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# ==================== 5. PDP ====================
print("\n" + "="*60)
print("3. PARTIAL DEPENDENCE PLOTS")
print("="*60)

from sklearn.inspection import PartialDependenceDisplay

# PDP za top 2 features
top_2_features = indices_rf[:2]

# Plot
ax = axes[1, 0]
PartialDependenceDisplay.from_estimator(
    rf, X_train_scaled, [top_2_features[0]],
    feature_names=feature_names,
    ax=ax,
    grid_resolution=50
)
ax.set_title(f'PDP: {feature_names[top_2_features[0]]}')

# ==================== 6. SHAP ====================
print("\n" + "="*60)
print("4. SHAP VALUES")
print("="*60)

# Create explainer
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)
shap_values_class1 = shap_values[1]  # Class 1 (malignant)

print("SHAP values computed!")

# Plot - Bar plot (global importance)
ax = axes[1, 1]
shap_importance = np.abs(shap_values_class1).mean(axis=0)
indices_shap = np.argsort(shap_importance)[::-1]
top_indices_shap = indices_shap[:top_n]

ax.barh(range(top_n), shap_importance[top_indices_shap])
ax.set_yticks(range(top_n))
ax.set_yticklabels(feature_names[top_indices_shap])
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('interpretation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 7. SHAP DETAILED PLOTS ====================

# SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class1, X_test_scaled, 
                  feature_names=feature_names,
                  max_display=15)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# SHAP Waterfall for single prediction
sample_idx = 0

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_class1[sample_idx],
        base_values=explainer.expected_value[1],
        data=X_test_scaled[sample_idx],
        feature_names=feature_names
    )
)
plt.title(f'SHAP Waterfall - Sample {sample_idx}\n'
         f'True: {cancer.target_names[y_test[sample_idx]]}, '
         f'Pred: {cancer.target_names[y_pred[sample_idx]]}')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.show()

# SHAP Dependence Plot (top feature)
top_feature_idx = indices_shap[0]

shap.dependence_plot(
    top_feature_idx,
    shap_values_class1,
    X_test_scaled,
    feature_names=feature_names
)
plt.title(f'SHAP Dependence: {feature_names[top_feature_idx]}')
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 8. COMPARISON TABLE ====================
print("\n" + "="*60)
print("FEATURE RANKING COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'RF Importance': [feature_names[i] for i in indices_rf[:10]],
    'Permutation': [feature_names[i] for i in indices_perm[:10]],
    'SHAP': [feature_names[i] for i in indices_shap[:10]]
})

print(comparison_df.to_string(index=True))

# ==================== 9. INSIGHTS ====================
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

# Top feature across all methods
top_feature_name = feature_names[indices_shap[0]]
print(f"\n🏆 Most Important Feature: {top_feature_name}")

# Check consistency
top_3_rf = set([feature_names[i] for i in indices_rf[:3]])
top_3_perm = set([feature_names[i] for i in indices_perm[:3]])
top_3_shap = set([feature_names[i] for i in indices_shap[:3]])

consistent_top = top_3_rf & top_3_perm & top_3_shap

if consistent_top:
    print(f"\n✅ Consistent Top Features Across All Methods:")
    for feat in consistent_top:
        print(f"   • {feat}")
else:
    print("\n⚠️  Methods disagree on top features - investigate further!")

# Model complexity
n_features_80 = np.sum(np.cumsum(np.sort(importances_rf)[::-1]) <= 0.8) + 1
print(f"\n📊 Features needed for 80% importance: {n_features_80}/{len(feature_names)}")

print("\n" + "="*60)
```

---

## Rezime - Model Interpretation

### Quick Reference:

| Method | Scope | Speed | Model Support | Best For |
|--------|-------|-------|---------------|----------|
| **Feature Importance** | Global | ⚡⚡⚡ | Tree-based | Quick overview |
| **Permutation Importance** | Global | ⚡⚡ | Any | Reliable importance |
| **PDP** | Global | ⚡⚡ | Any | Feature relationships |
| **ICE** | Individual | ⚡ | Any | Heterogeneity |
| **SHAP** | Both | ⚡ (Tree) 🐌 (Kernel) | Any | **Comprehensive** |
| **LIME** | Local | ⚡⚡ | Any | Quick local explanations |
| **Coefficients** | Global | ⚡⚡⚡ | Linear only | Direct interpretation |

### Default Strategy:
```
Step 1: Quick overview
  └─→ Feature Importance (if tree-based)
  └─→ Permutation Importance (any model)

Step 2: Detailed analysis
  └─→ SHAP (TreeExplainer for tree-based, comprehensive)
  └─→ PDP for top features (global relationships)

Step 3: Individual predictions
  └─→ SHAP Waterfall/Force Plot
  └─→ LIME (if SHAP too slow)

Step 4: Business insights
  └─→ Present findings to stakeholders
  └─→ Actionable recommendations
```

### Kada Koristiti Šta:
```
Need FAST overview?
  → Feature Importance (tree-based)

Need RELIABLE importance?
  → Permutation Importance

Want to understand relationships?
  → PDP / ICE

Need BEST overall method?
  → SHAP (TreeExplainer for speed)

Need to explain ONE prediction quickly?
  → LIME

Linear model?
  → Coefficients (direct)

Have TIME for comprehensive analysis?
  → SHAP Summary + PDP + Dependence Plots
```

**Key Takeaway:** Interpretacija nije opciona - MORA biti deo svakog ML projekta! SHAP je trenutno gold standard (game theory foundation, comprehensive). Za brzi insight koristi Feature/Permutation Importance. Za individual predictions koristi SHAP Waterfall ili LIME. Kombinuj više metoda za robustne insights! 🎯