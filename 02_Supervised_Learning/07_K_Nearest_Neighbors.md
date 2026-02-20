# K-Nearest Neighbors (KNN)

K-Nearest Neighbors je **najjednostavniji ML algoritam**. Uopšte ne "uči" tokom treninga - samo pamti training data i pravi predictions na osnovu **najbližih suseda**.

**Zašto je KNN važan?**
- **Najlakši za razumevanje** - "Reci mi ko ti je drug, reći ću ti ko si ti"
- **Non-parametric** - Nema pretpostavke o distribuciji data
- **Lazy learning** - Nema trening faze (instant "fit")
- **Versatile** - Classification i regression
- **Baseline model** - Brz test da li ML uopšte ima smisla

**Kada koristiti KNN?**
- ✅ Small datasets (<10k samples)
- ✅ Low-dimensional data (<20 features)
- ✅ Non-linear decision boundaries
- ✅ Brz baseline model
- ✅ Pattern recognition (images, signals)

**Kada NE koristiti:**
- ❌ Large datasets (spor na prediction!)
- ❌ High-dimensional data (curse of dimensionality)
- ❌ Potrebna interpretabilnost → Logistic/Tree
- ❌ Real-time predictions (<1ms) → Linear models
- ❌ Noisy data → Tree-based (robusniji)

---

## Kako KNN Radi

**Koncept:** "Birds of a feather flock together"

### Algoritam:
```
TRAINING FAZA:
  └─ Store all training data (X_train, y_train)
     Ništa više! 🎉

PREDICTION FAZA (za novu tačku x):
  1. Calculate distance između x i SVIH training tačaka
  2. Find K nearest neighbors (K tačaka sa najmanjom distance)
  3. Prediction:
     - Classification: Majority vote (najčešća klasa među K suseda)
     - Regression: Average vrednost među K suseda
```

**Primer (Classification):**
```
Training Data:
  o o o     x x x
  o   o     x   x
    o         x

Nova tačka: ? (označena sa *)

K=3 (3 najbliža suseda):
  o o o     x x x
  o   o *   x   x
    o         x
    
Nearest 3: [o, o, x]
Majority vote: o (2 vs 1)
Prediction: Class o ✅
```

---

## Distance Metrics

KNN koristi **distance** da nađe "bliske" susede. Različite metrics daju različite rezultate!

### 1. Euclidean Distance (Default)

**Formula:**
```
d(x, y) = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]

Standardna "straight-line" distance
```

### 2. Manhattan Distance

**Formula:**
```
d(x, y) = |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|

"City block" distance (kao da ideš ulicama grada)
```

### 3. Minkowski Distance (Generalization)

**Formula:**
```
d(x, y) = (Σ|xᵢ-yᵢ|ᵖ)^(1/p)

p=1 → Manhattan
p=2 → Euclidean
```

### Python Implementacija:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Generate data
np.random.seed(42)
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# ⚠️ KNN ZAHTEVA SCALING!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==================== KNN CLASSIFIER ====================
print("="*60)
print("K-NEAREST NEIGHBORS CLASSIFIER")
print("="*60)

# Train (instant!)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # Just stores data!

# Predictions
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nK=5, Euclidean distance")
print(f"Test Accuracy: {accuracy:.3f}")

# Visualize decision boundary
def plot_knn_boundary(model, X, y, title):
    """Plot KNN decision boundary."""
    plt.figure(figsize=(10, 6))
    
    # Mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', 
                edgecolors='k', s=100, label='Class 0', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', 
                edgecolors='k', s=100, label='Class 1', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_knn_boundary(knn, X_scaled, y, 'KNN Decision Boundary (K=5)')

# ==================== DIFFERENT METRICS ====================
print("\n" + "="*60)
print("COMPARISON: Different Distance Metrics")
print("="*60)

metrics = ['euclidean', 'manhattan', 'minkowski']

for metric in metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_metric.fit(X_train, y_train)
    acc = knn_metric.score(X_test, y_test)
    
    print(f"{metric.capitalize():12s}: Accuracy = {acc:.3f}")

print("\nObservation: Euclidean je najčešće najbolji (default)")
```

---

## Choosing K (Number of Neighbors)

**K** je **najvažniji hyperparameter** u KNN!
```
K → Small (npr. K=1):
  ├─ Veoma fleksibilan (memorizes training data)
  ├─ Low bias, HIGH variance
  ├─ Osetljiv na noise i outliers
  └─ Overfitting risk ⚠️

K → Large (npr. K=50):
  ├─ Smoother decision boundary
  ├─ HIGH bias, low variance
  ├─ Robusniji na noise
  └─ Underfitting risk ⚠️

K → Optimal (između):
  └─ Balance bias-variance
```

### Finding Optimal K:
```python
# Test different K values
k_values = range(1, 31)
train_scores = []
test_scores = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    
    train_scores.append(knn_k.score(X_train, y_train))
    test_scores.append(knn_k.score(X_test, y_test))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Train Accuracy', linewidth=2)
plt.plot(k_values, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs K\n(Small K → Overfitting, Large K → Underfitting)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 31, 5))
plt.tight_layout()
plt.show()

# Best K
best_k = k_values[np.argmax(test_scores)]
best_acc = max(test_scores)
print(f"\n🏆 Best K: {best_k} (Test Accuracy: {best_acc:.3f})")
```

**Opservacije:**
- **K=1**: Perfect train accuracy (100%), ali loš test → Overfitting!
- **K rasте**: Train accuracy pada, test accuracy prvo raste pa pada
- **Optimal K**: Obično između 3-15

**Pravilo:** Koristi **odd K** za binary classification (izbegava ties u voting)

---

## Weights: Uniform vs Distance

**Problem:** Svi K suseda imaju jednak uticaj, čak i oni daleki.

**Rešenje:** Weight susede po **inverse of distance** - bliži = jači uticaj!
```python
# ==================== UNIFORM vs DISTANCE WEIGHTS ====================
print("\n" + "="*60)
print("UNIFORM vs DISTANCE WEIGHTS")
print("="*60)

# Uniform weights (default)
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_uniform.fit(X_train, y_train)
acc_uniform = knn_uniform.score(X_test, y_test)

# Distance weights
knn_distance = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_distance.fit(X_train, y_train)
acc_distance = knn_distance.score(X_test, y_test)

print(f"Uniform weights:  {acc_uniform:.3f}")
print(f"Distance weights: {acc_distance:.3f}")

if acc_distance > acc_uniform:
    print(f"\n✅ Distance weights better by +{(acc_distance - acc_uniform):.3f}")
else:
    print("\n⚠️ Uniform weights sufficient for this dataset")

# Visualize both
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (model, title) in enumerate([(knn_uniform, 'Uniform Weights'),
                                        (knn_distance, 'Distance Weights')]):
    ax = axes[idx]
    
    # Mesh
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], c='red', marker='o', s=50, alpha=0.7, edgecolors='k')
    ax.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], c='blue', marker='s', s=50, alpha=0.7, edgecolors='k')
    
    acc = model.score(X_test, y_test)
    ax.set_title(f'{title}\nAccuracy: {acc:.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

**Preporuka:** Start sa `weights='uniform'`, ako vidiš overfitting probaj `weights='distance'`

---

## KNN Regression

KNN može raditi i **regression** - predicts **average** vrednost K suseda.
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=200,
    n_features=1,
    noise=10,
    random_state=42
)

X_reg_scaled = StandardScaler().fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

# ==================== KNN REGRESSOR ====================
print("\n" + "="*60)
print("KNN REGRESSOR")
print("="*60)

# Train
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_r, y_train_r)

# Predictions
y_pred_r = knn_reg.predict(X_test_r)

# Metrics
r2 = r2_score(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print(f"K=5")
print(f"R²:   {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Visualize
X_plot = np.linspace(X_reg_scaled.min(), X_reg_scaled.max(), 300).reshape(-1, 1)
y_plot = knn_reg.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_r, y_train_r, alpha=0.6, label='Train Data', edgecolors='k')
plt.scatter(X_test_r, y_test_r, alpha=0.6, label='Test Data', color='orange', edgecolors='k')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'KNN (K=5, R²={r2:.2f})')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('KNN Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nObservation: KNN regression gives 'step-like' predictions")
```

---

## Curse of Dimensionality

**Problem:** KNN performanse **drastično opadaju** sa većim brojem features!

**Zašto?**
```
Low dimensions (2D-3D):
  └─ Distance ima smisla (blizu = slično)

High dimensions (100D+):
  ├─ SVE tačke su približno JEDNAKO daleke!
  ├─ "Nearest" neighbors više nisu "near"
  └─ KNN ne radi dobro ⚠️
```

### Demonstration:
```python
from sklearn.datasets import make_classification

print("\n" + "="*60)
print("CURSE OF DIMENSIONALITY")
print("="*60)

# Test different dimensions
dimensions = [2, 5, 10, 20, 50, 100]
accuracies = []

for n_features in dimensions:
    # Generate data
    X_dim, y_dim = make_classification(
        n_samples=500,
        n_features=n_features,
        n_informative=min(n_features, 10),
        n_redundant=max(0, n_features - 10),
        random_state=42
    )
    
    # Scale
    X_dim_scaled = StandardScaler().fit_transform(X_dim)
    
    # Split
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_dim_scaled, y_dim, test_size=0.2, random_state=42
    )
    
    # Train KNN
    knn_dim = KNeighborsClassifier(n_neighbors=5)
    knn_dim.fit(X_train_d, y_train_d)
    
    acc = knn_dim.score(X_test_d, y_test_d)
    accuracies.append(acc)
    
    print(f"Features: {n_features:3d} → Accuracy: {acc:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(dimensions, accuracies, 'o-', linewidth=2, markersize=10)
plt.xlabel('Number of Features (Dimensions)')
plt.ylabel('Test Accuracy')
plt.title('KNN Performance vs Dimensionality\n(Accuracy drops as dimensions increase)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n⚠️ KNN performance degrades significantly with high dimensions!")
print("✅ Solution: Use PCA or feature selection to reduce dimensions")
```

**Rešenja:**
- **PCA** (Principal Component Analysis) - Reduce dimensions
- **Feature Selection** - Keep only important features
- **Use different algorithm** - Tree-based ne pate od curse of dimensionality

---

## Feature Scaling: KRITIČNO!

**KNN je EKSTREMNO osetljiv na scaling** jer koristi distance!
```python
# Generate data sa različitim scales
X_unscaled = np.column_stack([
    np.random.randn(200),        # Feature 1: range ~[-3, 3]
    np.random.randn(200) * 1000  # Feature 2: range ~[-3000, 3000]
])
y_scale = (X_unscaled[:, 0] + X_unscaled[:, 1]/1000 > 0).astype(int)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_unscaled, y_scale, test_size=0.2, random_state=42
)

# ==================== WITHOUT SCALING ====================
print("\n" + "="*60)
print("KNN WITHOUT SCALING (BAD!)")
print("="*60)

knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_train_s, y_train_s)
acc_no_scale = knn_no_scale.score(X_test_s, y_test_s)

print(f"Accuracy without scaling: {acc_no_scale:.3f}")
print("⚠️ Feature 2 dominates distance calculation!")

# ==================== WITH SCALING ====================
print("\n" + "="*60)
print("KNN WITH SCALING (GOOD!)")
print("="*60)

scaler_demo = StandardScaler()
X_train_s_scaled = scaler_demo.fit_transform(X_train_s)
X_test_s_scaled = scaler_demo.transform(X_test_s)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_s_scaled, y_train_s)
acc_scaled = knn_scaled.score(X_test_s_scaled, y_test_s)

print(f"Accuracy with scaling: {acc_scaled:.3f}")
print(f"\n🚨 Improvement: +{(acc_scaled - acc_no_scale):.3f}")
print("✅ ALWAYS scale features before KNN!")
```

**Za detalje o Scaling, vidi:** `01_Data_Preprocessing/06_Feature_Scaling.md`

---

## Computational Complexity

**Training:** O(1) - Samo čuva data (instant!)

**Prediction:** O(n × d) - Mora računati distance do SVIH training samples!
- n = broj training samples
- d = broj features

**Problem:** Prediciton je **SPOR** na large datasets!

### Speed Optimization: Ball Tree & KD Tree
```python
from sklearn.neighbors import KNeighborsClassifier
import time

# Generate larger dataset
X_large, y_large = make_classification(
    n_samples=10000,
    n_features=10,
    random_state=42
)
X_large_scaled = StandardScaler().fit_transform(X_large)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_large_scaled, y_large, test_size=0.2, random_state=42
)

print("\n" + "="*60)
print("SPEED COMPARISON: Different Algorithms")
print("="*60)

algorithms = ['brute', 'ball_tree', 'kd_tree']

for algo in algorithms:
    knn_algo = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    
    # Train (instant for all)
    start = time.time()
    knn_algo.fit(X_train_l, y_train_l)
    train_time = time.time() - start
    
    # Predict (this is where differences show)
    start = time.time()
    y_pred_l = knn_algo.predict(X_test_l)
    predict_time = time.time() - start
    
    acc = accuracy_score(y_test_l, y_pred_l)
    
    print(f"\n{algo.upper()}:")
    print(f"  Train time:   {train_time:.4f}s")
    print(f"  Predict time: {predict_time:.4f}s")
    print(f"  Accuracy:     {acc:.3f}")

print("\nObservation:")
print("  - Brute force: Najjednostavniji, ali najsporiji")
print("  - Ball Tree / KD Tree: Brži za prediction (especially >20 dimensions)")
print("  - Default ('auto'): sklearn automatically chooses best")
```

**Preporuka:** Ostavi `algorithm='auto'` (default) - sklearn bira optimalno!

---

## Complete Example: Iris Classification
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("IRIS FLOWER CLASSIFICATION - KNN")
print("="*60)

# ==================== 1. LOAD DATA ====================
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"\nDataset: Iris")
print(f"Samples: {X_iris.shape[0]}")
print(f"Features: {X_iris.shape[1]}")
print(f"Classes: {len(np.unique(y_iris))} (Setosa, Versicolor, Virginica)")

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# ==================== 3. SCALING ====================
scaler_i = StandardScaler()
X_train_i_scaled = scaler_i.fit_transform(X_train_i)
X_test_i_scaled = scaler_i.transform(X_test_i)

print(f"\nTrain: {X_train_i.shape}")
print(f"Test:  {X_test_i.shape}")
print("✅ Features scaled")

# ==================== 4. BASELINE - K=5 ====================
print("\n" + "="*60)
print("BASELINE - KNN (K=5)")
print("="*60)

knn_baseline = KNeighborsClassifier(n_neighbors=5)
knn_baseline.fit(X_train_i_scaled, y_train_i)

y_pred_baseline = knn_baseline.predict(X_test_i_scaled)
acc_baseline = accuracy_score(y_test_i, y_pred_baseline)

print(f"Test Accuracy: {acc_baseline:.3f}")

# ==================== 5. FIND OPTIMAL K ====================
print("\n" + "="*60)
print("FINDING OPTIMAL K")
print("="*60)

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train_i_scaled, y_train_i, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, 'o-', linewidth=2)
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal K (5-Fold CV)')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.tight_layout()
plt.show()

best_k = k_range[np.argmax(cv_scores)]
best_cv_score = max(cv_scores)
print(f"\n🏆 Best K: {best_k} (CV Accuracy: {best_cv_score:.3f})")

# ==================== 6. HYPERPARAMETER TUNING ====================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - GridSearchCV")
print("="*60)

param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("Running GridSearchCV...")
knn_grid.fit(X_train_i_scaled, y_train_i)

print(f"\nBest parameters: {knn_grid.best_params_}")
print(f"Best CV Accuracy: {knn_grid.best_score_:.3f}")

# ==================== 7. FINAL EVALUATION ====================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

best_knn = knn_grid.best_estimator_
y_pred_best = best_knn.predict(X_test_i_scaled)

acc_best = accuracy_score(y_test_i, y_pred_best)
print(f"Test Accuracy: {acc_best:.3f}")

print("\nClassification Report:")
print(classification_report(y_test_i, y_pred_best, 
                           target_names=iris.target_names))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_i = confusion_matrix(y_test_i, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_i, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== 8. MODEL COMPARISON ====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Baseline (K=5)', f'Optimal K (K={best_k})', 'Tuned (GridSearch)'],
    'Accuracy': [acc_baseline, 
                 KNeighborsClassifier(n_neighbors=best_k).fit(X_train_i_scaled, y_train_i).score(X_test_i_scaled, y_test_i),
                 acc_best]
})

print(comparison.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
plt.bar(comparison['Model'], comparison['Accuracy'],
        color=['gray', 'blue', 'green'], alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Iris Classification')
plt.ylim([0.9, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: {knn_grid.best_params_}")
print(f"   Test Accuracy: {acc_best:.3f}")

# ==================== 9. SAVE MODEL ====================
import joblib

joblib.dump(best_knn, 'knn_iris.pkl')
joblib.dump(scaler_i, 'scaler_iris.pkl')

print("\n✅ Model saved: knn_iris.pkl")
print("✅ Scaler saved: scaler_iris.pkl")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Key Hyperparameters
```python
KNeighborsClassifier(
    n_neighbors=5,          # ⭐ K (1-20 usually)
    weights='uniform',      # 'uniform' ili 'distance'
    metric='minkowski',     # 'euclidean', 'manhattan', 'minkowski'
    p=2,                    # Power for Minkowski (p=2 → Euclidean)
    algorithm='auto',       # 'auto', 'ball_tree', 'kd_tree', 'brute'
    n_jobs=-1               # Parallelization
)
```

**Tuning Strategy:**
1. **n_neighbors**: Start sa 5, test 1-20 (cross-validation)
2. **weights**: Probaj oba ('uniform' i 'distance')
3. **metric**: Euclidean je obično dovoljan
4. **algorithm**: Ostavi 'auto'

**Za Hyperparameter Tuning, vidi:** `05_Model_Evaluation_and_Tuning/05_Hyperparameter_Tuning.md`

---

## Best Practices

### ✅ DO:

1. **UVEK scale features** - StandardScaler obavezan!
2. **Use odd K** za binary classification (izbegava ties)
3. **Cross-validation** za izbor K
4. **Start sa K=5** kao baseline
5. **Low-dimensional data** (<20 features) - KNN radi najbolje
6. **weights='distance'** ako vidiš overfitting
7. **Feature selection** ako imaš mnogo features

### ❌ DON'T:

1. **Ne koristi na large datasets** (>100k) - Prediction je SPOR!
2. **Ne zaboravi scaling** - Katastrofa bez njega
3. **Ne koristi na high-dimensional** (>50 features) - Curse of dimensionality
4. **Ne koristi K=1** (prevelik overfitting)
5. **Ne ignoriši outliers** - Jako utiču na KNN

---

## Common Pitfalls

### Greška 1: No Scaling
```python
# ❌ NAJGORA GREŠKA
knn_bad = KNeighborsClassifier(n_neighbors=5)
knn_bad.fit(X_train, y_train)  # Features nisu skalirani!

# ✅ DOBRO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
knn_good = KNeighborsClassifier(n_neighbors=5)
knn_good.fit(X_scaled, y_train)
```

### Greška 2: Large Dataset
```python
# ❌ LOŠE - 1M samples, prediction će trajati večno!
knn_slow = KNeighborsClassifier()
knn_slow.fit(X_huge, y_huge)
predictions = knn_slow.predict(X_test_huge)  # 🐌🐌🐌

# ✅ DOBRO - Koristi Logistic Regression ili Tree-based
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_huge, y_huge)  # ⚡ Mnogo brže!
```

### Greška 3: High Dimensions
```python
# ❌ LOŠE - 100 features, KNN ne radi dobro
knn_bad = KNeighborsClassifier()
knn_bad.fit(X_high_dim, y)  # 100 features

# ✅ DOBRO - Reduce dimensions prvo
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_high_dim)
knn_good = KNeighborsClassifier()
knn_good.fit(X_reduced, y)
```

---

## Kada Koristiti KNN?

### ✅ Idealno Za:

- **Small datasets** (<10k samples)
- **Low-dimensional data** (<20 features)
- **Non-linear boundaries**
- **Pattern recognition** (images, handwriting)
- **Baseline model** (brz test)
- **No assumptions** o data distribuciji

### ❌ Izbegavaj Za:

- **Large datasets** (>100k) → Logistic/XGBoost
- **High-dimensional** (>50 features) → Tree-based
- **Real-time predictions** → Linear models
- **Noisy data** → Tree-based (robusniji)
- **Interpretabilnost** → Logistic/Single Tree

---

## KNN vs Other Algorithms

| Aspekt | KNN | Logistic Regression | SVM | Random Forest |
|--------|-----|---------------------|-----|---------------|
| **Training Speed** | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Prediction Speed** | ⭐ Spor! | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐ Veliko | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Non-linear** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Scaling Required** | ✅ Obavezno! | ✅ Preporučeno | ✅ Obavezno! | ❌ Ne |
| **High Dimensions** | ❌ Loše | ✅ OK | ⭐⭐⭐ | ✅ OK |
| **Interpretability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## Rezime

| Aspekt | KNN |
|--------|-----|
| **Tip** | Classification & Regression (Lazy Learning) |
| **Interpretabilnost** | ⭐⭐⭐ Umeren (vidiš susede) |
| **Training Speed** | ⭐⭐⭐⭐⭐ Instant (nema učenje!) |
| **Prediction Speed** | ⭐ Spor (mora računati sve distance) |
| **Memory** | ⭐ Veliko (čuva sve training data) |
| **Performance** | ⭐⭐⭐ Dobar (small, low-dim data) |
| **Handles Non-linearity** | ⭐⭐⭐⭐ Dobar |
| **Feature Scaling** | ✅ **OBAVEZNO** |
| **Curse of Dimensionality** | 🚨 Veliki problem! |
| **Best For** | Small datasets, low dimensions, baseline |

---

## Quick Decision Tree
```
Start
  ↓
Small dataset (<10k)?
  ↓ Yes
Low-dimensional (<20 features)?
  ↓ Yes
Potreban brz baseline?
  ↓ Yes
→ KNN ✅

Ako dataset je veliki (>100k):
  └─ Logistic Regression ili Tree-based

Ako mnogo features (>50):
  └─ PCA → KNN ili Tree-based

Ako treba production speed:
  └─ Linear models ili Tree-based
```

---

**Key Takeaway:** KNN je **najjednostavniji ML algoritam** - perfektan za **učenje** i **brze baseline modele**. Radi odlično na **small, low-dimensional datasets** ali **pati od curse of dimensionality** i **spor je na prediction**. **KRITIČNO**: **UVEK scale features**! Za production obično se zamenjuje sa Logistic Regression ili Tree-based modelima. 🎯