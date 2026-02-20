# Support Vector Machines (SVM)

Support Vector Machines su **moćni algoritmi** koji grade **optimal decision boundary** između klasa. Odlični za small-to-medium datasets sa kompleksnim decision boundaries.

**Zašto su SVM-ovi važni?**
- **Kernel trick** - Može raditi sa non-linear boundaries
- **Robusni** - Fokusiraju se na "teške" primere (support vectors)
- **Dobar za high-dimensional data** - Radi dobro čak i kad features > samples
- **Margin maximization** - Nalazi "najbolji" boundary (najdalji od obe klase)
- **Versatile** - Classification, regression, outlier detection

**Kada koristiti SVM?**
- ✅ Small-to-medium datasets (<10k samples)
- ✅ High-dimensional data (text, genomics)
- ✅ Non-linear decision boundaries
- ✅ Clear margin of separation između klasa
- ✅ Outlier detection (One-Class SVM)

**Kada NE koristiti:**
- ❌ Large datasets (>100k) → Tree-based (brži)
- ❌ Jako noisy data → Tree-based (robusniji)
- ❌ Potrebna interpretabilnost → Logistic/Tree
- ❌ Potrebne verovatnoće → Logistic (SVM daje scores, ne probabilities)
- ❌ Ekstremno imbalanced data → Tree-based sa class_weight

---

## Intuicija: Maximum Margin Classifier

**Cilj:** Naći **hyperplane** (decision boundary) koja **maksimalno razdvaja** dve klase.

### Šta je Hyperplane?
```
2D: Linija (y = mx + b)
3D: Ravan (ax + by + cz = d)
n-D: Hyperplane (w₁x₁ + w₂x₂ + ... + wₙxₙ + b = 0)
```

### Margin:
```
      Class 0          MARGIN          Class 1
         o                              x
       o   o            |              x   x
         o              |                x
    ─────────────────────────────────────────
         o          boundary              x
       o   o            |              x   x
         o              |                x

Margin = Distance od najbližih tačaka (support vectors) do boundary
```

**SVM cilj:** Maksimizuj margin → Boundary što dalje od obe klase!

**Support Vectors:** Tačke **na margini** - jedine tačke koje utiču na boundary!

---

## Linear SVM

**Matematika (Simplified):**
```
Decision function:
f(x) = w·x + b

Prediction:
  f(x) ≥ 0 → Class 1
  f(x) < 0 → Class 0

Cilj: Maksimizuj margin = 2 / ||w||

Subject to: y_i(w·x_i + b) ≥ 1 for all training samples
```

### Python Implementacija:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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

# ⚠️ SVM ZAHTEVA SCALING!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==================== LINEAR SVM ====================
print("="*60)
print("LINEAR SVM")
print("="*60)

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

y_pred = svm_linear.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.3f}")
print(f"Number of Support Vectors: {len(svm_linear.support_)}")
print(f"Support Vector Indices: {svm_linear.support_[:10]}...")  # First 10

# Visualize
def plot_svm_decision_boundary(model, X, y, title):
    """Plot SVM decision boundary."""
    plt.figure(figsize=(10, 6))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, 
                 colors=['red', 'white', 'blue'])
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])
    
    # Plot data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', 
                edgecolors='k', s=100, label='Class 0', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', 
                edgecolors='k', s=100, label='Class 1', alpha=0.7)
    
    # Highlight support vectors
    plt.scatter(X[model.support_, 0], X[model.support_, 1], 
                s=200, linewidth=2, facecolors='none', 
                edgecolors='green', label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_svm_decision_boundary(svm_linear, X_scaled, y, 'Linear SVM Decision Boundary')
```

---

## C Parameter (Regularization)

**C kontroliše trade-off** između:
- **Maximizing margin** (wider margin, more tolerance for errors)
- **Minimizing training errors** (narrower margin, fewer errors)
```
C → Small (npr. 0.1):
  ├─ Wider margin
  ├─ More violations (soft margin)
  ├─ Better generalization (less overfitting)
  └─ Underfitting risk if too small

C → Large (npr. 100):
  ├─ Narrower margin
  ├─ Fewer violations (hard margin)
  ├─ Better training accuracy
  └─ Overfitting risk if too large
```

### Comparison:
```python
# Test different C values
C_values = [0.01, 0.1, 1, 10, 100]

fig, axes = plt.subplots(1, len(C_values), figsize=(20, 4))

for idx, C in enumerate(C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    
    test_acc = svm.score(X_test, y_test)
    n_support = len(svm.support_)
    
    # Plot
    ax = axes[idx]
    
    # Mesh
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.3, colors=['red', 'white', 'blue'])
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    ax.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], c='red', marker='o', s=50, alpha=0.7)
    ax.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], c='blue', marker='s', s=50, alpha=0.7)
    ax.scatter(X_scaled[svm.support_, 0], X_scaled[svm.support_, 1], 
               s=150, linewidth=2, facecolors='none', edgecolors='green')
    
    ax.set_title(f'C={C}\nAcc={test_acc:.2f}, SVs={n_support}')
    ax.set_xlabel('Feature 1')
    if idx == 0:
        ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\nObservations:")
print("  Small C → More support vectors, wider margin")
print("  Large C → Fewer support vectors, narrower margin")
```

---

## Kernel Trick

**Problem:** Linear SVM ne može razdvojiti **non-linearly separable** data.

**Rešenje:** Mapiranje u **higher-dimensional space** gde postaje linearno separabilno!
```
Original Space (2D):       Higher-Dimensional Space (3D+):
   x   x   x                       x   x   x
     o o o                           o o o
   x   x   x                       x   x   x
                                   
Ne može linearna linija!    →   Može linearni plane!
```

**Kernel Trick:** Ne moramo eksplicitno računati transformaciju - kernel funkcija to radi implicitno!

### Kernels:

#### 1. **RBF (Radial Basis Function)** - Najčešći!
```
K(x, x') = exp(-γ ||x - x'||²)

γ (gamma) kontroliše "influence" svakog training sample:
  - Small γ: Daleki uticaj → Smoother boundary
  - Large γ: Blizu uticaj → Complex boundary (overfitting risk)
```
```python
# ==================== RBF KERNEL ====================
print("\n" + "="*60)
print("RBF KERNEL")
print("="*60)

# Generate non-linear data
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)
X_moons_scaled = StandardScaler().fit_transform(X_moons)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_moons_scaled, y_moons, test_size=0.2, random_state=42
)

# Linear SVM (will fail!)
svm_linear_m = SVC(kernel='linear', C=1.0)
svm_linear_m.fit(X_train_m, y_train_m)
linear_acc = svm_linear_m.score(X_test_m, y_test_m)

# RBF SVM (will succeed!)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_m, y_train_m)
rbf_acc = svm_rbf.score(X_test_m, y_test_m)

print(f"Linear SVM Accuracy: {linear_acc:.3f}")
print(f"RBF SVM Accuracy:    {rbf_acc:.3f}")
print(f"Improvement: +{(rbf_acc - linear_acc):.3f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (model, title) in enumerate([(svm_linear_m, 'Linear Kernel'),
                                        (svm_rbf, 'RBF Kernel')]):
    ax = axes[idx]
    
    # Mesh
    x_min, x_max = X_moons_scaled[:, 0].min() - 1, X_moons_scaled[:, 0].max() + 1
    y_min, y_max = X_moons_scaled[:, 1].min() - 1, X_moons_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_moons_scaled[y_moons==0, 0], X_moons_scaled[y_moons==0, 1], 
               c='red', marker='o', edgecolors='k', s=80, alpha=0.7)
    ax.scatter(X_moons_scaled[y_moons==1, 0], X_moons_scaled[y_moons==1, 1], 
               c='blue', marker='s', edgecolors='k', s=80, alpha=0.7)
    
    acc = model.score(X_test_m, y_test_m)
    ax.set_title(f'{title}\nAccuracy: {acc:.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

#### 2. **Polynomial Kernel**
```
K(x, x') = (γ·x·x' + r)^d

degree (d): Polynomial degree (2, 3, 4, ...)
  - degree=2: Quadratic
  - degree=3: Cubic
```
```python
# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0, gamma='scale')
svm_poly.fit(X_train_m, y_train_m)
poly_acc = svm_poly.score(X_test_m, y_test_m)

print(f"\nPolynomial (degree=3) SVM Accuracy: {poly_acc:.3f}")
```

#### 3. **Sigmoid Kernel**
```
K(x, x') = tanh(γ·x·x' + r)

Retko korišćen u praksi.
```

---

## Gamma Parameter (RBF & Poly)

**Gamma** kontroliše **influence** svakog training sample.
```
γ → Small (npr. 0.01):
  ├─ Daleki samples utiču
  ├─ Smooth decision boundary
  └─ Underfitting risk

γ → Large (npr. 10):
  ├─ Samo bliski samples utiču
  ├─ Complex decision boundary
  └─ Overfitting risk (memorizes training data)
```

### Comparison:
```python
gamma_values = [0.01, 0.1, 1, 10]

fig, axes = plt.subplots(1, len(gamma_values), figsize=(16, 4))

for idx, gamma in enumerate(gamma_values):
    svm_g = SVC(kernel='rbf', C=1.0, gamma=gamma)
    svm_g.fit(X_train_m, y_train_m)
    
    test_acc = svm_g.score(X_test_m, y_test_m)
    
    ax = axes[idx]
    
    # Mesh
    x_min, x_max = X_moons_scaled[:, 0].min() - 1, X_moons_scaled[:, 0].max() + 1
    y_min, y_max = X_moons_scaled[:, 1].min() - 1, X_moons_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = svm_g.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_moons_scaled[y_moons==0, 0], X_moons_scaled[y_moons==0, 1], 
               c='red', marker='o', s=50, alpha=0.7, edgecolors='k')
    ax.scatter(X_moons_scaled[y_moons==1, 0], X_moons_scaled[y_moons==1, 1], 
               c='blue', marker='s', s=50, alpha=0.7, edgecolors='k')
    
    ax.set_title(f'gamma={gamma}\nAcc={test_acc:.2f}')
    ax.set_xlabel('Feature 1')
    if idx == 0:
        ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("\nObservations:")
print("  Small gamma → Smooth boundary (może underfitting)")
print("  Large gamma → Complex boundary (može overfitting)")
```

---

## SVR (Support Vector Regression)

SVM može raditi i **regression**!

**Cilj:** Naći funkciju gde većina tačaka pada **unutar epsilon-tube** (margin).
```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X_reg_scaled = StandardScaler().fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

# ==================== SVR ====================
print("\n" + "="*60)
print("SUPPORT VECTOR REGRESSION (SVR)")
print("="*60)

# Linear SVR
svr_linear = SVR(kernel='linear', C=1.0)
svr_linear.fit(X_train_r, y_train_r)

# RBF SVR
svr_rbf = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_rbf.fit(X_train_r, y_train_r)

# Predictions
y_pred_linear = svr_linear.predict(X_test_r)
y_pred_rbf = svr_rbf.predict(X_test_r)

# Metrics
r2_linear = r2_score(y_test_r, y_pred_linear)
r2_rbf = r2_score(y_test_r, y_pred_rbf)

print(f"Linear SVR R²: {r2_linear:.3f}")
print(f"RBF SVR R²:    {r2_rbf:.3f}")

# Visualize
X_plot = np.linspace(X_reg_scaled.min(), X_reg_scaled.max(), 300).reshape(-1, 1)
y_plot_linear = svr_linear.predict(X_plot)
y_plot_rbf = svr_rbf.predict(X_plot)

plt.figure(figsize=(12, 6))
plt.scatter(X_train_r, y_train_r, alpha=0.6, label='Train Data', edgecolors='k')
plt.scatter(X_test_r, y_test_r, alpha=0.6, label='Test Data', color='orange', edgecolors='k')
plt.plot(X_plot, y_plot_linear, 'r-', linewidth=2, label=f'Linear SVR (R²={r2_linear:.2f})')
plt.plot(X_plot, y_plot_rbf, 'g-', linewidth=2, label=f'RBF SVR (R²={r2_rbf:.2f})')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Support Vector Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Feature Scaling: KRITIČNO!

**SVM je EKSTREMNO osetljiv na feature scaling!**

### Zašto?

SVM koristi **distance** između tačaka. Ako jedna feature ima range [0, 1] a druga [0, 10000], druga će dominirati!
```python
# Generate data sa različitim scales
X_unscaled = np.column_stack([
    np.random.randn(200),        # Feature 1: mean=0, std=1
    np.random.randn(200) * 1000  # Feature 2: mean=0, std=1000
])
y_scale = (X_unscaled[:, 0] + X_unscaled[:, 1]/1000 > 0).astype(int)

X_train_scale, X_test_scale, y_train_scale, y_test_scale = train_test_split(
    X_unscaled, y_scale, test_size=0.2, random_state=42
)

# ==================== BEZ SCALING ====================
print("\n" + "="*60)
print("SVM WITHOUT SCALING (BAD!)")
print("="*60)

svm_no_scale = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_no_scale.fit(X_train_scale, y_train_scale)
acc_no_scale = svm_no_scale.score(X_test_scale, y_test_scale)

print(f"Accuracy without scaling: {acc_no_scale:.3f}")

# ==================== SA SCALING ====================
print("\n" + "="*60)
print("SVM WITH SCALING (GOOD!)")
print("="*60)

scaler_demo = StandardScaler()
X_train_scale_scaled = scaler_demo.fit_transform(X_train_scale)
X_test_scale_scaled = scaler_demo.transform(X_test_scale)

svm_scaled = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_scaled.fit(X_train_scale_scaled, y_train_scale)
acc_scaled = svm_scaled.score(X_test_scale_scaled, y_test_scale)

print(f"Accuracy with scaling: {acc_scaled:.3f}")
print(f"\n🚨 Improvement: +{(acc_scaled - acc_no_scale):.3f}")
print("✅ ALWAYS scale features before SVM!")
```

**Za detalje o Scaling, vidi:** `01_Data_Preprocessing/06_Feature_Scaling.md`

---

## Complete Example: Digit Recognition
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DIGIT RECOGNITION (0-9) - SVM")
print("="*60)

# ==================== 1. LOAD DATA ====================
digits = load_digits()
X_digits = digits.data  # 8x8 images flattened to 64 features
y_digits = digits.target

print(f"\nDataset: {digits.DESCR.split(chr(10))[0]}")
print(f"Samples: {X_digits.shape[0]}")
print(f"Features: {X_digits.shape[1]}")
print(f"Classes: {len(np.unique(y_digits))} (digits 0-9)")

# Visualize some digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flat):
    ax.imshow(digits.images[idx], cmap='gray')
    ax.set_title(f'Label: {y_digits[idx]}')
    ax.axis('off')
plt.suptitle('Sample Digits')
plt.tight_layout()
plt.show()

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=42, stratify=y_digits
)

# ==================== 3. SCALING ====================
scaler_d = StandardScaler()
X_train_d_scaled = scaler_d.fit_transform(X_train_d)
X_test_d_scaled = scaler_d.transform(X_test_d)

print(f"\nTrain: {X_train_d.shape}")
print(f"Test:  {X_test_d.shape}")
print("✅ Features scaled")

# ==================== 4. BASELINE - LINEAR SVM ====================
print("\n" + "="*60)
print("BASELINE - LINEAR SVM")
print("="*60)

svm_baseline = SVC(kernel='linear', C=1.0)
svm_baseline.fit(X_train_d_scaled, y_train_d)

y_pred_baseline = svm_baseline.predict(X_test_d_scaled)
acc_baseline = accuracy_score(y_test_d, y_pred_baseline)

print(f"Linear SVM Accuracy: {acc_baseline:.3f}")

# ==================== 5. RBF SVM - DEFAULT ====================
print("\n" + "="*60)
print("RBF SVM - DEFAULT")
print("="*60)

svm_rbf_default = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf_default.fit(X_train_d_scaled, y_train_d)

y_pred_rbf = svm_rbf_default.predict(X_test_d_scaled)
acc_rbf = accuracy_score(y_test_d, y_pred_rbf)

print(f"RBF SVM Accuracy: {acc_rbf:.3f}")
print(f"Improvement over linear: +{(acc_rbf - acc_baseline):.3f}")

# ==================== 6. HYPERPARAMETER TUNING ====================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - GridSearchCV")
print("="*60)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    SVC(),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Running GridSearchCV...")
svm_grid.fit(X_train_d_scaled, y_train_d)

print(f"\nBest parameters: {svm_grid.best_params_}")
print(f"Best CV Accuracy: {svm_grid.best_score_:.3f}")

# ==================== 7. FINAL EVALUATION ====================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

best_svm = svm_grid.best_estimator_
y_pred_best = best_svm.predict(X_test_d_scaled)

acc_best = accuracy_score(y_test_d, y_pred_best)
print(f"Test Accuracy: {acc_best:.3f}")

print("\nClassification Report:")
print(classification_report(y_test_d, y_pred_best))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_d = confusion_matrix(y_test_d, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_d, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - Digit Recognition')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== 8. MISCLASSIFIED EXAMPLES ====================
print("\n" + "="*60)
print("MISCLASSIFIED EXAMPLES")
print("="*60)

# Find misclassified
misclassified_idx = np.where(y_test_d != y_pred_best)[0]
print(f"Total misclassified: {len(misclassified_idx)} / {len(y_test_d)}")

if len(misclassified_idx) > 0:
    # Show first 10
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flat):
        if idx < len(misclassified_idx):
            img_idx = misclassified_idx[idx]
            ax.imshow(X_test_d[img_idx].reshape(8, 8), cmap='gray')
            ax.set_title(f'True: {y_test_d[img_idx]}, Pred: {y_pred_best[img_idx]}')
            ax.axis('off')
    plt.suptitle('Misclassified Digits')
    plt.tight_layout()
    plt.show()

# ==================== 9. MODEL COMPARISON ====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Linear SVM', 'RBF SVM (default)', 'RBF SVM (tuned)'],
    'Accuracy': [acc_baseline, acc_rbf, acc_best]
})

print(comparison.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
plt.bar(comparison['Model'], comparison['Accuracy'], 
        color=['gray', 'blue', 'green'], alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Digit Recognition')
plt.ylim([0.9, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: RBF SVM (tuned)")
print(f"   Parameters: C={svm_grid.best_params_['C']}, gamma={svm_grid.best_params_['gamma']}")
print(f"   Test Accuracy: {acc_best:.3f}")

# ==================== 10. SAVE MODEL ====================
import joblib

joblib.dump(best_svm, 'svm_digit_recognition.pkl')
joblib.dump(scaler_d, 'scaler_digit_recognition.pkl')

print("\n✅ Model saved: svm_digit_recognition.pkl")
print("✅ Scaler saved: scaler_digit_recognition.pkl")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Key Hyperparameters
```python
SVC(
    C=1.0,              # ⭐ Regularization (0.1-100)
    kernel='rbf',       # ⭐ 'linear', 'rbf', 'poly', 'sigmoid'
    gamma='scale',      # ⭐ Kernel coefficient (0.001-10)
    degree=3,           # Polynomial degree (only for poly kernel)
    class_weight=None,  # 'balanced' za imbalanced data
    probability=False,  # True ako trebaš probabilities (sporije!)
    random_state=42
)
```

**Tuning Strategy:**
1. Start sa `kernel='rbf'` (najčešće najbolji)
2. Tune `C` (0.1, 1, 10, 100)
3. Tune `gamma` ('scale', 0.001, 0.01, 0.1, 1)
4. Ako RBF ne radi, probaj `kernel='linear'`
5. Retko: `kernel='poly'` sa `degree=2` ili `3`

**Za Hyperparameter Tuning, vidi:** `05_Model_Evaluation_and_Tuning/05_Hyperparameter_Tuning.md`

---

## Best Practices

### ✅ DO:

1. **UVEK scale features** - StandardScaler obavezan!
2. **Start sa RBF kernel** - Najčešće najbolji
3. **Tune C i gamma zajedno** - GridSearchCV
4. **Use small datasets** - SVM skalira O(n²) ili O(n³)
5. **class_weight='balanced'** za imbalanced data
6. **probability=False** ako ne treba (brže)

### ❌ DON'T:

1. **Ne koristi na large datasets** (>100k) - Preee spor!
2. **Ne zaboravi scaling** - Katastrofalne performanse bez njega
3. **Ne koristi linear kernel za non-linear** - RBF je bolji
4. **Ne preteruj sa gamma** - Large gamma = overfitting
5. **Ne ignoriši tuning** - Default params retko optimalni

---

## Common Pitfalls

### Greška 1: No Scaling
```python
# ❌ NAJGORA GREŠKA - No scaling
svm_bad = SVC(kernel='rbf')
svm_bad.fit(X_train, y_train)  # Katastrofa!

# ✅ DOBRO - Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
svm_good = SVC(kernel='rbf')
svm_good.fit(X_scaled, y_train)
```

### Greška 2: Large Dataset
```python
# ❌ LOŠE - 1M samples, trajaće večno!
svm_slow = SVC()
svm_slow.fit(X_huge, y_huge)  # 🐌🐌🐌

# ✅ DOBRO - Koristi SGDClassifier (linear SVM approximation)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='hinge')  # Hinge loss = linear SVM
sgd.fit(X_huge, y_huge)  # ⚡ Brzo!
```

### Greška 3: Wrong Kernel
```python
# ❌ LOŠE - Linear kernel za non-linear data
svm_bad = SVC(kernel='linear')
svm_bad.fit(X_nonlinear, y)

# ✅ DOBRO - RBF kernel
svm_good = SVC(kernel='rbf', gamma='scale')
svm_good.fit(X_nonlinear, y)
```

---

## Kada Koristiti SVM?

### ✅ Idealno Za:

- **Small-to-medium datasets** (<10k samples)
- **High-dimensional data** (text classification, genomics)
- **Clear margin** između klasa
- **Non-linear boundaries** (sa RBF kernel)
- **Binary classification** (najbolji use case)
- **Outlier detection** (One-Class SVM)

### ❌ Izbegavaj Za:

- **Large datasets** (>100k) → Logistic/XGBoost (brži)
- **Very noisy data** → Tree-based (robusniji)
- **Potrebne verovatnoće** → Logistic Regression (direktne probabilities)
- **Interpretabilnost** → Logistic/Single Tree
- **Ekstremno imbalanced** → Tree-based sa class_weight

---

## SVM vs Other Algorithms

| Aspekt | SVM | Logistic Regression | Random Forest | XGBoost |
|--------|-----|---------------------|---------------|---------|
| **Small Data** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Large Data** | ⭐ Spor! | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Non-linear** | ⭐⭐⭐⭐⭐ RBF | ⭐⭐ Poly features | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Interpretability** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Training Speed** | ⭐⭐ Spor | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Scaling Required** | ✅ Obavezno! | ✅ Preporučeno | ❌ Ne | ❌ Ne |
| **Hyperparameter Tuning** | ⭐⭐ Dosta | ⭐⭐⭐⭐ Malo | ⭐⭐⭐ Umeren | ⭐⭐ Dosta |

---

## Rezime

| Aspekt | SVM |
|--------|-----|
| **Tip** | Classification & Regression |
| **Interpretabilnost** | ⭐⭐ Nizak (support vectors vidljivi) |
| **Training Speed** | ⭐⭐ Spor (O(n²) do O(n³)) |
| **Prediction Speed** | ⭐⭐⭐ Umeren |
| **Performance** | ⭐⭐⭐⭐ Excellent (small-medium data) |
| **Handles Non-linearity** | ⭐⭐⭐⭐⭐ Excellent (sa RBF) |
| **Feature Scaling** | ✅ **OBAVEZNO** |
| **Overfitting Risk** | ⭐⭐⭐ Umeren (tuning C i gamma) |
| **Best For** | Small data, high-dimensional, clear margin |

---

## Quick Decision Tree
```
Start
  ↓
Small-to-medium dataset (<10k)?
  ↓ Yes
High-dimensional data (text, genomics)?
  ↓ Yes
Clear margin između klasa?
  ↓ Yes
→ SVM (RBF kernel) ✅

Ako dataset je veliki (>100k):
  └─ Logistic Regression ili XGBoost

Ako treba interpretabilnost:
  └─ Logistic Regression ili Single Tree

Ako data je jako noisy:
  └─ Random Forest ili XGBoost
```

---

**Key Takeaway:** SVM je **moćan algoritam za small-to-medium datasets** sa **kompleksnim decision boundaries**. **RBF kernel** sa **C i gamma tuning** daje odlične rezultate. **KRITIČNO**: **UVEK scale features** pre SVM-a! Za large datasets koristi Logistic Regression ili XGBoost (mnogo brži). 🎯