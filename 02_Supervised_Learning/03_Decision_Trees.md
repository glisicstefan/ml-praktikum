# Decision Trees

Decision Trees su **interpretabilni algoritmi** koji donose odluke kroz **seriju if-else pravila**. Rade i za classification i za regression. 

**Zašto su Decision Trees važni?**
- **Intuitivni** - Lako se razumeju (čak i non-technical ljudi)
- **Vizuelni** - Možeš nacrtati ceo decision process
- **Ne zahtevaju scaling** - Rade sa raw data
- **Handle non-linearity** - Automatski uhvataju kompleksne veze
- **Feature importance** - Vidiš koje features su najvažnije
- **Osnova za ensemble metode** - Random Forest i Gradient Boosting grade trees!

**Kada koristiti Decision Trees?**
- ✅ Potrebna interpretabilnost
- ✅ Mixed feature types (numerical + categorical)
- ✅ Non-linear relationships
- ✅ Ne želiš da razmišljaš o scaling
- ✅ Brz prototip ili baseline

**Kada NE koristiti:**
- ❌ Kao finalni production model (overfittuju lako!)
- ❌ Za najbolje performanse (koristi Random Forest ili XGBoost)
- ❌ Linearne veze (Linear/Logistic su bolji i brži)

---

## Kako Decision Tree Radi?

Decision Tree **rekurzivno particionise** feature space u **regione** i pravi **predictions** na osnovu toga.

### Proces:
```
1. Start sa svim podacima (root node)
2. Nađi najbolji feature i threshold za split
3. Podeli podatke u 2 grupe (levo i desno)
4. Ponovi proces za svaku grupu (rekurzivno)
5. Zaustavi kada je dostignut stopping kriterijum
```

**Primer:**
```
Dataset: Predviđanje da li osoba igra tenis

Weather | Temperature | Humidity | Wind   | Play Tennis?
--------|-------------|----------|--------|-------------
Sunny   | Hot         | High     | Weak   | No
Sunny   | Hot         | High     | Strong | No
Overcast| Hot         | High     | Weak   | Yes
Rain    | Mild        | High     | Weak   | Yes
...

Decision Tree:
                   [Weather]
                  /    |    \
            Sunny/  Overcast \Rain
                /      |      \
              [No]    [Yes]   [Wind]
                              /    \
                        Weak/      \Strong
                            /        \
                          [Yes]      [No]
```

---

## Classification Trees

### Splitting Criteria

Cilj: Naći **najbolji split** koji **maksimizuje čistoću** (purity) children nodes.

**Dva glavna kriterijuma:**

#### 1. Gini Impurity (Default u sklearn)
```
Gini(node) = 1 - Σ(p_i)²

gde je p_i = proportion of class i u node-u

Gini = 0 → Potpuno čist node (sve iste klase)
Gini = 0.5 → Maksimalno nečist (50/50 split)
```

**Primer:**
```python
import numpy as np

def gini_impurity(y):
    """Calculate Gini impurity."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    gini = 1 - np.sum(proportions ** 2)
    return gini

# Node sa 100 samples: 50 class 0, 50 class 1
y1 = np.array([0]*50 + [1]*50)
print(f"Gini (50/50 split): {gini_impurity(y1):.3f}")  # 0.5

# Node sa 100 samples: 90 class 0, 10 class 1
y2 = np.array([0]*90 + [1]*10)
print(f"Gini (90/10 split): {gini_impurity(y2):.3f}")  # 0.18

# Pure node: 100 class 0
y3 = np.array([0]*100)
print(f"Gini (pure node):   {gini_impurity(y3):.3f}")  # 0.0
```

#### 2. Entropy (Information Gain)
```
Entropy(node) = -Σ(p_i × log₂(p_i))

Information Gain = Entropy(parent) - Weighted_Avg(Entropy(children))

Entropy = 0 → Potpuno čist node
Entropy = 1 → Maksimalno nečist (binary)
```
```python
def entropy(y):
    """Calculate entropy."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    entropy_val = -np.sum(proportions * np.log2(proportions + 1e-10))
    return entropy_val

# Node sa 100 samples: 50 class 0, 50 class 1
print(f"Entropy (50/50 split): {entropy(y1):.3f}")  # 1.0

# Node sa 100 samples: 90 class 0, 10 class 1
print(f"Entropy (90/10 split): {entropy(y2):.3f}")  # 0.469

# Pure node
print(f"Entropy (pure node):   {entropy(y3):.3f}")  # 0.0
```

**Gini vs Entropy:**
- **Gini**: Brži za compute (nema log)
- **Entropy**: Teoretski malo bolji (information theory)
- **U praksi**: Skoro identični rezultati! Koristi Gini (default).

---

## Regression Trees

Za regression, cilj je **minimizovati variance** (MSE) u svakom node-u.

**Splitting Criteria:**
```
MSE(node) = (1/n) × Σ(y_i - ȳ)²

gde je ȳ = mean vrednost y u node-u

Cilj: Split koji minimizuje weighted average MSE children nodes
```

**Prediction:** Mean vrednost svih samples u leaf node-u.

---

## Python Implementacija - Classification
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset:", iris.DESCR.split('\n')[0])
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== DEFAULT TREE (OVERFITS!) ====================
print("\n" + "="*60)
print("DEFAULT DECISION TREE (No Constraints)")
print("="*60)

tree_default = DecisionTreeClassifier(random_state=42)
tree_default.fit(X_train, y_train)

y_train_pred = tree_default.predict(X_train)
y_test_pred = tree_default.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTrain Accuracy: {train_acc:.3f}")
print(f"Test Accuracy:  {test_acc:.3f}")
print(f"Tree Depth:     {tree_default.get_depth()}")
print(f"Number of Leaves: {tree_default.get_n_leaves()}")

if train_acc == 1.0:
    print("\n⚠️ Perfect train accuracy → OVERFITTING!")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree_default, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree - Default (Overfitted)")
plt.tight_layout()
plt.show()

# ==================== PRUNED TREE (Better!) ====================
print("\n" + "="*60)
print("PRUNED DECISION TREE (max_depth=3)")
print("="*60)

tree_pruned = DecisionTreeClassifier(
    max_depth=3,           # Limit depth
    random_state=42
)
tree_pruned.fit(X_train, y_train)

y_train_pred_p = tree_pruned.predict(X_train)
y_test_pred_p = tree_pruned.predict(X_test)

train_acc_p = accuracy_score(y_train, y_train_pred_p)
test_acc_p = accuracy_score(y_test, y_test_pred_p)

print(f"\nTrain Accuracy: {train_acc_p:.3f}")
print(f"Test Accuracy:  {test_acc_p:.3f}")
print(f"Tree Depth:     {tree_pruned.get_depth()}")
print(f"Number of Leaves: {tree_pruned.get_n_leaves()}")

print(f"\n✅ Better generalization! (Test accuracy improved: {test_acc_p - test_acc:+.3f})")

# Visualize pruned tree
plt.figure(figsize=(20, 10))
plot_tree(tree_pruned,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Decision Tree - Pruned (max_depth=3)")
plt.tight_layout()
plt.show()
```

---

## Tree Parameters (Stopping Criteria)

Ovi parametri kontrolišu **tree growth** i **sprečavaju overfitting**:
```python
DecisionTreeClassifier(
    criterion='gini',           # 'gini' ili 'entropy'
    max_depth=None,             # Max dubina tree-a (npr. 3, 5, 10)
    min_samples_split=2,        # Min samples za split node
    min_samples_leaf=1,         # Min samples u leaf node
    max_features=None,          # Max features za split (None = all)
    max_leaf_nodes=None,        # Max broj leaf nodes
    min_impurity_decrease=0.0,  # Min decrease in impurity za split
    random_state=42
)
```

### Najvažniji Parametri:

#### 1. **max_depth** - Najvažniji!
```python
# Test different depths
depths = [1, 2, 3, 5, 10, None]
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Plot
plt.figure(figsize=(10, 6))
depth_labels = [str(d) if d is not None else 'None' for d in depths]
x_pos = np.arange(len(depths))

plt.plot(x_pos, train_scores, 'o-', label='Train Accuracy', linewidth=2)
plt.plot(x_pos, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Performance vs Max Depth')
plt.xticks(x_pos, depth_labels)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find best depth
best_depth_idx = np.argmax(test_scores)
best_depth = depths[best_depth_idx]
print(f"\n🏆 Best max_depth: {best_depth} (Test Acc: {test_scores[best_depth_idx]:.3f})")
```

#### 2. **min_samples_split** & **min_samples_leaf**
```python
# min_samples_split: Minimum samples required to split
tree_split = DecisionTreeClassifier(min_samples_split=20, random_state=42)
tree_split.fit(X_train, y_train)

print(f"\nmin_samples_split=20:")
print(f"  Test Accuracy: {tree_split.score(X_test, y_test):.3f}")
print(f"  Tree Depth: {tree_split.get_depth()}")

# min_samples_leaf: Minimum samples in leaf node
tree_leaf = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
tree_leaf.fit(X_train, y_train)

print(f"\nmin_samples_leaf=10:")
print(f"  Test Accuracy: {tree_leaf.score(X_test, y_test):.3f}")
print(f"  Tree Depth: {tree_leaf.get_depth()}")
```

---

## Regression Trees
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate regression data
X_reg, y_reg = make_regression(
    n_samples=200,
    n_features=1,
    noise=10,
    random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train regression tree
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train_r, y_train_r)

y_train_pred_r = tree_reg.predict(X_train_r)
y_test_pred_r = tree_reg.predict(X_test_r)

# Metrics
train_r2 = r2_score(y_train_r, y_train_pred_r)
test_r2 = r2_score(y_test_r, y_test_pred_r)
test_rmse = np.sqrt(mean_squared_error(y_test_r, y_test_pred_r))

print(f"\nRegression Tree (max_depth=3):")
print(f"  Train R²: {train_r2:.3f}")
print(f"  Test R²:  {test_r2:.3f}")
print(f"  Test RMSE: {test_rmse:.3f}")

# Visualize predictions
X_plot = np.linspace(X_reg.min(), X_reg.max(), 300).reshape(-1, 1)
y_plot = tree_reg.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_r, y_train_r, alpha=0.6, label='Train Data', edgecolors='k')
plt.scatter(X_test_r, y_test_r, alpha=0.6, label='Test Data', edgecolors='k', color='orange')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Decision Tree Prediction')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title(f'Decision Tree Regressor (max_depth=3, R²={test_r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Notice: Predictions are "step-like" (piecewise constant)
```

**Napomena:** Regression tree predictions su **piecewise constant** (step functions). Za smooth predictions, koristi Random Forest ili Gradient Boosting!

---

## Feature Importance

Decision Trees automatski računaju **feature importance** - koliko svaka feature doprinosi smanjenju impurity.
```python
# Feature importance
importances = tree_pruned.feature_importances_
feature_names = iris.feature_names

# DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], 
         color='skyblue', alpha=0.7, edgecolor='black')
plt.xlabel('Importance')
plt.title('Feature Importance - Decision Tree')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

**Interpretacija:**
- **Importance = 0**: Feature nije korišćena ni u jednom split-u
- **Importance > 0**: Feature je korišćena (veća vrednost = važnija)
- **Sum = 1.0**: Sve importances sum-uju na 1

---

## Overfitting Problem

**Decision Trees LAKO OVERFITTUJU** ako ih ne kontrolišeš!
```python
# Extreme overfitting example
X_overfit, y_overfit = make_regression(n_samples=100, n_features=5, noise=20, random_state=42)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_overfit, y_overfit, test_size=0.2, random_state=42
)

# No constraints → Overfit!
tree_overfit = DecisionTreeRegressor(random_state=42)
tree_overfit.fit(X_train_o, y_train_o)

# With constraints → Better!
tree_controlled = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree_controlled.fit(X_train_o, y_train_o)

# Compare
print("\nOVERFITTING COMPARISON:")
print("\nNo Constraints:")
print(f"  Train R²: {tree_overfit.score(X_train_o, y_train_o):.3f}")
print(f"  Test R²:  {tree_overfit.score(X_test_o, y_test_o):.3f}")
print(f"  Depth: {tree_overfit.get_depth()}")
print(f"  Leaves: {tree_overfit.get_n_leaves()}")

print("\nWith Constraints:")
print(f"  Train R²: {tree_controlled.score(X_train_o, y_train_o):.3f}")
print(f"  Test R²:  {tree_controlled.score(X_test_o, y_test_o):.3f}")
print(f"  Depth: {tree_controlled.get_depth()}")
print(f"  Leaves: {tree_controlled.get_n_leaves()}")

gap_overfit = tree_overfit.score(X_train_o, y_train_o) - tree_overfit.score(X_test_o, y_test_o)
gap_controlled = tree_controlled.score(X_train_o, y_train_o) - tree_controlled.score(X_test_o, y_test_o)

print(f"\nTrain-Test Gap:")
print(f"  No Constraints: {gap_overfit:.3f} 🚨")
print(f"  With Constraints: {gap_controlled:.3f} ✅")
```

---

## Complete Example: Titanic Survival Prediction
```python
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TITANIC SURVIVAL PREDICTION - DECISION TREE")
print("="*60)

# ==================== 1. LOAD DATA ====================
# Using seaborn's built-in dataset
titanic = sns.load_dataset('titanic')

print(f"\nDataset shape: {titanic.shape}")
print("\nFirst 5 rows:")
print(titanic.head())

# ==================== 2. DATA PREPROCESSING ====================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Select features
df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Encoding categorical features
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

print("\nMissing values:")
print(df.isnull().sum())

# Features and target
X_titanic = df.drop('survived', axis=1)
y_titanic = df['survived']

feature_names_titanic = X_titanic.columns.tolist()

print(f"\nFeatures: {feature_names_titanic}")
print(f"Target: survived (0=No, 1=Yes)")

# ==================== 3. TRAIN-TEST SPLIT ====================
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42, stratify=y_titanic
)

print(f"\nTrain set: {X_train_t.shape}")
print(f"Test set:  {X_test_t.shape}")

# ==================== 4. BASELINE ====================
print("\n" + "="*60)
print("BASELINE - Predict Most Frequent Class")
print("="*60)

most_frequent_t = y_train_t.mode()[0]
y_baseline_t = np.full(len(y_test_t), most_frequent_t)
baseline_acc_t = accuracy_score(y_test_t, y_baseline_t)

print(f"Baseline Accuracy: {baseline_acc_t:.3f}")

# ==================== 5. DECISION TREE - TUNING ====================
print("\n" + "="*60)
print("DECISION TREE - HYPERPARAMETER TUNING")
print("="*60)

# Test different max_depths
depths_t = [2, 3, 4, 5, 6, 7, 8, None]
results_t = []

for depth in depths_t:
    tree_t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_t.fit(X_train_t, y_train_t)
    
    train_acc_t = tree_t.score(X_train_t, y_train_t)
    test_acc_t = tree_t.score(X_test_t, y_test_t)
    
    results_t.append({
        'max_depth': depth if depth is not None else 'None',
        'Train Acc': train_acc_t,
        'Test Acc': test_acc_t,
        'Depth': tree_t.get_depth(),
        'Leaves': tree_t.get_n_leaves()
    })

results_df_t = pd.DataFrame(results_t)
print("\nHyperparameter Tuning Results:")
print(results_df_t.to_string(index=False))

# Best depth
best_idx_t = results_df_t['Test Acc'].idxmax()
best_depth_t = results_df_t.loc[best_idx_t, 'max_depth']
best_test_acc_t = results_df_t.loc[best_idx_t, 'Test Acc']

print(f"\n🏆 Best max_depth: {best_depth_t} (Test Acc: {best_test_acc_t:.3f})")

# ==================== 6. FINAL MODEL ====================
print("\n" + "="*60)
print("FINAL MODEL")
print("="*60)

# Use best depth
if best_depth_t == 'None':
    final_tree = DecisionTreeClassifier(random_state=42)
else:
    final_tree = DecisionTreeClassifier(max_depth=int(best_depth_t), random_state=42)

final_tree.fit(X_train_t, y_train_t)

y_test_pred_t = final_tree.predict(X_test_t)
test_acc_final = accuracy_score(y_test_t, y_test_pred_t)

print(f"\nTest Accuracy: {test_acc_final:.3f}")
print(f"Improvement over baseline: +{(test_acc_final - baseline_acc_t):.3f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_t, y_test_pred_t, target_names=['Not Survived', 'Survived']))

# Confusion Matrix
cm_t = confusion_matrix(y_test_t, y_test_pred_t)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix - Titanic Survival')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== 7. FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importances_t = final_tree.feature_importances_
importance_df_t = pd.DataFrame({
    'Feature': feature_names_titanic,
    'Importance': importances_t
}).sort_values('Importance', ascending=False)

print("\n" + importance_df_t.to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(importance_df_t['Feature'], importance_df_t['Importance'],
         color='coral', alpha=0.7, edgecolor='black')
plt.xlabel('Importance')
plt.title('Feature Importance - Titanic Survival Prediction')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ==================== 8. VISUALIZE TREE ====================
plt.figure(figsize=(25, 15))
plot_tree(final_tree,
          feature_names=feature_names_titanic,
          class_names=['Not Survived', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=11)
plt.title(f"Decision Tree - Titanic (max_depth={best_depth_t})")
plt.tight_layout()
plt.show()

# ==================== 9. EXAMPLE PREDICTIONS ====================
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

# Example passengers
examples = pd.DataFrame({
    'pclass': [1, 3],
    'sex': [1, 0],  # 1=female, 0=male
    'age': [25, 30],
    'sibsp': [0, 1],
    'parch': [0, 2],
    'fare': [100, 15],
    'embarked': [0, 2]  # 0=C, 2=S
})

predictions = final_tree.predict(examples)
probabilities = final_tree.predict_proba(examples)

print("\nExample Passengers:")
for i in range(len(examples)):
    print(f"\nPassenger {i+1}:")
    print(f"  Class: {examples.loc[i, 'pclass']}, Sex: {'Female' if examples.loc[i, 'sex']==1 else 'Male'}, Age: {examples.loc[i, 'age']}")
    print(f"  Prediction: {'Survived' if predictions[i]==1 else 'Not Survived'}")
    print(f"  Probability: {probabilities[i, 1]:.3f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Key Hyperparameters
```python
DecisionTreeClassifier(
    criterion='gini',           # 'gini' (default) ili 'entropy'
    max_depth=None,             # ⭐ NAJVAŽNIJI (3-10 usually)
    min_samples_split=2,        # Min samples za split (2-20)
    min_samples_leaf=1,         # Min samples u leaf (1-10)
    max_features=None,          # Max features za split
    max_leaf_nodes=None,        # Max broj leaves
    min_impurity_decrease=0.0,  # Min impurity decrease
    random_state=42
)
```

**Tuning Strategy:**
1. Start sa `max_depth` (najvažniji!)
2. Tune `min_samples_split` i `min_samples_leaf`
3. Opciono: `max_leaf_nodes` ili `max_features`

**Za Hyperparameter Tuning, vidi:** `05_Model_Evaluation_and_Tuning/05_Hyperparameter_Tuning.md`

---

## Best Practices

### ✅ DO:

1. **UVEK ograniči tree depth** - max_depth=3 do 10 je OK
2. **Visualize tree** - Proveri da li ima smisla
3. **Check feature importance** - Insights za business
4. **Cross-validation** - Za hyperparameter tuning
5. **Use as baseline** - Brz prototip pre ensemble metoda
6. **Monitor train-test gap** - Indikator overfit-a

### ❌ DON'T:

1. **Ne ostavljaj None constraints** - Overfit je garantovan!
2. **Ne koristi kao finalni production model** - Random Forest/XGBoost su bolji
3. **Ne zaboravi encoding** - Trees ne rade sa string categorical data
4. **Ne ignoriši missing values** - Moraš ih handle-ovati
5. **Ne poredi direktno sa scaled modelima** - Trees ne trebaju scaling

---

## Common Pitfalls

### Greška 1: No Constraints (Overfitting)
```python
# ❌ LOŠE - Overfit garantovan
tree_bad = DecisionTreeClassifier()
tree_bad.fit(X_train, y_train)
# Train: 100%, Test: 70%

# ✅ DOBRO - Controlled growth
tree_good = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
tree_good.fit(X_train, y_train)
# Train: 85%, Test: 82%
```

### Greška 2: Not Encoding Categorical Features
```python
# ❌ LOŠE - String values
df['category'] = ['A', 'B', 'C', ...]  # sklearn će pući!

# ✅ DOBRO - Encoded
df['category'] = df['category'].map({'A': 0, 'B': 1, 'C': 2})
```

### Greška 3: Using Trees sa Mnogo Noise
```python
# Decision Trees su osetljivi na noise
# ❌ LOŠE - Single tree na noisy data
tree = DecisionTreeClassifier(max_depth=10)
tree.fit(X_noisy, y_noisy)

# ✅ DOBRO - Ensemble method (averaging reduces noise)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10, n_estimators=100)
rf.fit(X_noisy, y_noisy)
```

---

## Kada Koristiti Decision Trees?

### ✅ Idealno Za:

- **Interpretabilnost je ključna** (medical, legal, finance)
- **Brz prototip** ili baseline model
- **EDA** (feature importance insights)
- **Mixed feature types** (numerical + categorical)
- **Non-linear relationships**
- **Teaching/explaining ML** (najlakši za razumeti)

### ❌ Izbegavaj Za:

- **Production models** → Random Forest / XGBoost
- **Linearne veze** → Linear/Logistic Regression (brže i bolje)
- **Najbolje performanse** → Ensemble methods
- **Stabilnost** → Single tree je nestabilan (mali change u data → veliki change u tree)

---

## Rezime

| Aspekt | Opis |
|--------|------|
| **Tip** | Classification & Regression |
| **Interpretabilnost** | ⭐⭐⭐⭐⭐ Excellent (vizuelno + rules) |
| **Training Speed** | ⭐⭐⭐⭐ Brz |
| **Prediction Speed** | ⭐⭐⭐⭐⭐ Veoma brz (O(log n)) |
| **Handles Non-linearity** | ✅ Da |
| **Feature Scaling** | ❌ Ne treba |
| **Handles Mixed Types** | ✅ Da (numerical + categorical) |
| **Handles Missing** | ❌ Ne (mora se impute) |
| **Overfitting Risk** | 🚨 VEOMA VISOK (bez constraints) |
| **Stability** | ❌ Nestabilan (mali change → veliko change) |
| **Best For** | Interpretabilnost, baseline, EDA |

---

## Quick Decision Tree
```
Start  
  ↓  
Potrebna interpretabilnost?  
  ↓ Yes  
Non-linear relationships?  
  ↓ Yes  
Mixed feature types?  
  ↓ Yes  
→ DECISION TREE ✅
```

**Key Takeaway:** Decision Trees su **super za razumevanje problema** i **brzi baseline**, ali **overfittuju lako**! Za production, **UVEK** koristi **Random Forest** ili **XGBoost** (koji grade multiple trees i average-uju predictions). Single Decision Tree retko ide u production! 🎯