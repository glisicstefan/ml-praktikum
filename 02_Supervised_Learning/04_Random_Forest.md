# Random Forest

Random Forest je **ensemble algoritam** koji kombinuje **mnogo Decision Trees** da dobije bolji i robusniji model. Jedan od **najpopularnijih ML algoritama**!

**Zašto je Random Forest toliko dobar?**
- **Nadmašuje pojedinačni Decision Tree** - Mnogo bolja generalizacija
- **Smanjuje overfitting** - Averaging eliminuje individualne greške
- **Robustan** - Radi dobro na većini problema "out of the box"
- **Handles non-linearity** - Kao Decision Trees, ali bolje
- **Feature importance** - Vidiš koje features su važne
- **Ne zahteva scaling** - Radi sa raw data

**Kada koristiti Random Forest?**
- ✅ Structured/tabular data
- ✅ Potrebna dobra performance bez mnogo tuninga
- ✅ Non-linear relationships
- ✅ Mixed feature types (numerical + categorical)
- ✅ Feature importance je važna
- ✅ Brz prototip sa odličnim rezultatima

**Kada NE koristiti:**
- ❌ Ekstremno veliki datasets (spor!) → LightGBM/XGBoost
- ❌ Potrebna interpretabilnost (crna kutija) → Logistic/Single Tree
- ❌ Linearne veze (overkill) → Linear/Logistic
- ❌ Images/Text/Sequences → Neural Networks

---

## Bagging (Bootstrap Aggregating)

**Problem sa Decision Trees:** Jedan tree je **nestabilan** - mali change u data → potpuno drugačiji tree.

**Rešenje:** Napravi **mnogo trees** na **različitim subsetima** data i **average-uj** predictions!

### Kako Bagging Radi:
```
1. Uzmi N bootstrap samples iz original dataset-a
   (bootstrap = random sampling sa replacement)

2. Treniraj po jedan Decision Tree na svakom sample-u

3. Predictions:
   - Classification: Voting (majority vote)
   - Regression: Average predictions
```

**Primer:**
```
Original dataset: 100 samples

Bootstrap Sample 1: [12, 45, 67, 12, 89, ...] (100 samples, neke su duplikati)
→ Train Tree 1

Bootstrap Sample 2: [3, 78, 45, 90, 3, ...]
→ Train Tree 2

Bootstrap Sample 3: [56, 12, 23, 56, 78, ...]
→ Train Tree 3

...

Bootstrap Sample N: [...]
→ Train Tree N

Final Prediction = Average/Vote svih N trees
```

---

## Random Forest = Bagging + Random Features

**Random Forest ide korak dalje:**

Nije samo bootstrap samples, već i **random subset of features** na svakom split-u!

### Algoritam:
```
For each tree in forest:
  1. Bootstrap sample (random rows sa replacement)
  
  2. For each split in tree:
     - Randomly select m features (out of total p features)
     - Find best split SAMO među tim m features
     - Split node
  
  3. Grow tree do max_depth (ili drugi stopping criteria)

Final prediction = Average/Vote svih trees
```

**Parametar:** `max_features` kontroliše koliko features se razmatra na svakom split-u.

**Tipične vrednosti:**
- **Classification**: `sqrt(p)` (default) - npr. 10 features → razmatra 3
- **Regression**: `p/3` (default) - npr. 10 features → razmatra 3-4
- **Manji max_features** → Više diversity (trees su različitiji) → Manje overfitting
- **Veći max_features** → Manje diversity → Bolji pojedinačni trees

---

## Python Implementacija
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Dataset: Breast Cancer")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== SINGLE DECISION TREE ====================
print("\n" + "="*60)
print("SINGLE DECISION TREE")
print("="*60)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_train_acc = tree.score(X_train, y_train)
tree_test_acc = tree.score(X_test, y_test)

print(f"Train Accuracy: {tree_train_acc:.3f}")
print(f"Test Accuracy:  {tree_test_acc:.3f}")
print(f"Train-Test Gap: {tree_train_acc - tree_test_acc:.3f}")

if tree_train_acc == 1.0:
    print("⚠️ Perfect train accuracy → OVERFITTING!")

# ==================== RANDOM FOREST ====================
print("\n" + "="*60)
print("RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

print(f"Train Accuracy: {rf_train_acc:.3f}")
print(f"Test Accuracy:  {rf_test_acc:.3f}")
print(f"Train-Test Gap: {rf_train_acc - rf_test_acc:.3f}")

print(f"\n✅ Random Forest improvement: +{(rf_test_acc - tree_test_acc):.3f}")
print(f"✅ Reduced overfitting (smaller gap)")
```

---

## Key Hyperparameters
```python
RandomForestClassifier(
    n_estimators=100,           # ⭐ Broj trees (50-500)
    criterion='gini',           # 'gini' ili 'entropy'
    max_depth=None,             # ⭐ Max dubina svakog tree-a
    min_samples_split=2,        # Min samples za split
    min_samples_leaf=1,         # Min samples u leaf
    max_features='sqrt',        # ⭐ Features per split (sqrt, log2, int)
    max_samples=None,           # Bootstrap sample size (None = all)
    bootstrap=True,             # Use bootstrap? (uvek True)
    oob_score=False,            # ⭐ Out-of-bag validation
    n_jobs=-1,                  # ⭐ Parallelization (use all cores)
    random_state=42
)
```

### Najvažniji Parametri:

#### 1. **n_estimators** - Broj Trees
```python
# Test different number of trees
n_trees = [10, 50, 100, 200, 500]
train_scores = []
test_scores = []

for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    
    print(f"n_estimators={n:3d}: Train={train_scores[-1]:.3f}, Test={test_scores[-1]:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_trees, train_scores, 'o-', label='Train Accuracy', linewidth=2)
plt.plot(n_trees, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest Performance vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n✅ Test accuracy stabilizuje se nakon ~100 trees")
```

**Opservacija:**
- **Više trees = Bolje** (do neke granice)
- **Diminishing returns** nakon ~100-200 trees
- **Ne overfit-uje** sa više trees (za razliku od neuralnih mreža!)
- **Trade-off**: Više trees = sporije trening i prediction

**Preporuka:** Start sa 100, ako imaš vreme povećaj na 200-500.

#### 2. **max_depth** - Dubina Svakog Tree-a
```python
# Test different max_depths
depths = [5, 10, 20, 30, None]
train_scores_d = []
test_scores_d = []

for depth in depths:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_scores_d.append(rf.score(X_train, y_train))
    test_scores_d.append(rf.score(X_test, y_test))
    
    depth_str = str(depth) if depth is not None else 'None'
    print(f"max_depth={depth_str:5s}: Train={train_scores_d[-1]:.3f}, Test={test_scores_d[-1]:.3f}")

# Plot
plt.figure(figsize=(10, 6))
depth_labels = [str(d) if d is not None else 'None' for d in depths]
x_pos = np.arange(len(depths))
plt.plot(x_pos, train_scores_d, 'o-', label='Train Accuracy', linewidth=2)
plt.plot(x_pos, test_scores_d, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest Performance vs Max Depth')
plt.xticks(x_pos, depth_labels)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Preporuka:**
- **None** (default): OK za većinu slučajeva
- **Ograniči** (10-30) ako vidiš overfitting ili želiš brže
- **Manja dubina** = brži trening, manja memoria

#### 3. **max_features** - Features per Split
```python
# Test different max_features
max_feats = ['sqrt', 'log2', 0.5, None]  # None = all features
train_scores_f = []
test_scores_f = []

for feat in max_feats:
    rf = RandomForestClassifier(n_estimators=100, max_features=feat, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_scores_f.append(rf.score(X_train, y_train))
    test_scores_f.append(rf.score(X_test, y_test))
    
    print(f"max_features={str(feat):6s}: Train={train_scores_f[-1]:.3f}, Test={test_scores_f[-1]:.3f}")
```

**Interpretacija:**
- **'sqrt'** (default za classification): Dobra diversity
- **'log2'**: Još više diversity
- **None** (all features): Manje diversity, ali jači pojedinačni trees
- **Manji max_features** → Više diversity → Bolje generalizacija

**Preporuka:** Ostavi default ('sqrt').

#### 4. **min_samples_split & min_samples_leaf**
```python
# Test min_samples_split
rf_split = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=20,  # Default je 2
    random_state=42,
    n_jobs=-1
)
rf_split.fit(X_train, y_train)

print(f"\nmin_samples_split=20:")
print(f"  Test Accuracy: {rf_split.score(X_test, y_test):.3f}")

# Test min_samples_leaf
rf_leaf = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=10,  # Default je 1
    random_state=42,
    n_jobs=-1
)
rf_leaf.fit(X_train, y_train)

print(f"\nmin_samples_leaf=10:")
print(f"  Test Accuracy: {rf_leaf.score(X_test, y_test):.3f}")
```

**Kad koristiti:**
- Povećaj ove parametre ako vidiš **overfitting**
- Smanjuju kompleksnost svakog tree-a
- **Trade-off**: Manje overfitting vs slabiji pojedinačni trees

---

## Out-of-Bag (OOB) Score

**Šta je OOB?**

Bootstrap sampling uzima **~63%** original data za svaki tree (zbog replacement). Preostalih **~37%** data **NIJE korišćeno** za taj tree.

Te "out-of-bag" samples možeš koristiti za **BESPLATNU VALIDACIJU** (bez potrebe za validation set)!
```python
# Random Forest sa OOB score
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,      # Enable OOB
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)

print(f"\nOOB Score: {rf_oob.oob_score_:.3f}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.3f}")
print("\nOOB score je aproksimacija test score-a!")
print("Ako su blizu → model generalizuje dobro ✅")
```

**Prednosti OOB:**
- ✅ **Besplatna validacija** (ne gubi training data)
- ✅ **Aproksimacija cross-validation**
- ✅ **Brže od CV** (samo jedan fit)

**Preporuka:** Uvek uključi `oob_score=True` tokom razvoja!

---

## Feature Importance

Random Forest daje **agregiranu feature importance** preko svih trees.
```python
# Feature importance
importances = rf.feature_importances_
feature_names = cancer.feature_names

# DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# Visualization
plt.figure(figsize=(10, 8))
top_20 = importance_df.head(20)
plt.barh(top_20['Feature'], top_20['Importance'], color='forestgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Sum check
print(f"\nSum of importances: {importances.sum():.3f} (should be 1.0)")
```

**Interpretacija:**
- **Importance = 0**: Feature nije korišćena u nijednom split-u
- **Importance > 0**: Feature je korišćena (veća vrednost = važnija)
- **Relativna važnost**: Uporedi features međusobno

**Use case:**
- **Feature selection** - Ukloni features sa malom importance
- **Business insights** - Koji faktori najviše utiču?

---

## Bias-Variance Tradeoff

**Single Decision Tree:**
- **Low Bias** (može naučiti kompleksne veze)
- **High Variance** (nestabilan, mali change → veliko change)

**Random Forest:**
- **Low Bias** (još uvek može naučiti kompleksne veze)
- **Low Variance** ✅ (averaging redukuje variance!)
```
Variance Reduction = Why Random Forest Works!

Ako imaš N nezavisnih prediktora sa variance σ²:
Variance(Average) = σ² / N

Više trees → Manja variance → Bolja generalizacija
```

---

## Comparison: Decision Tree vs Random Forest
```python
from sklearn.model_selection import cross_val_score

print("="*60)
print("DECISION TREE vs RANDOM FOREST")
print("="*60)

# Decision Tree
tree_comp = DecisionTreeClassifier(random_state=42)
tree_cv_scores = cross_val_score(tree_comp, X_train, y_train, cv=5)

print(f"\nDecision Tree (5-Fold CV):")
print(f"  Mean Accuracy: {tree_cv_scores.mean():.3f}")
print(f"  Std Deviation: {tree_cv_scores.std():.3f}")
print(f"  Individual folds: {[f'{s:.3f}' for s in tree_cv_scores]}")

# Random Forest
rf_comp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_cv_scores = cross_val_score(rf_comp, X_train, y_train, cv=5)

print(f"\nRandom Forest (5-Fold CV):")
print(f"  Mean Accuracy: {rf_cv_scores.mean():.3f}")
print(f"  Std Deviation: {rf_cv_scores.std():.3f}")
print(f"  Individual folds: {[f'{s:.3f}' for s in rf_cv_scores]}")

print(f"\nImprovement:")
print(f"  Accuracy: +{(rf_cv_scores.mean() - tree_cv_scores.mean()):.3f}")
print(f"  Stability: {tree_cv_scores.std() - rf_cv_scores.std():.3f} (lower is better)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CV Scores comparison
axes[0].boxplot([tree_cv_scores, rf_cv_scores], labels=['Decision Tree', 'Random Forest'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Cross-Validation Score Distribution')
axes[0].grid(True, alpha=0.3)

# Bar comparison
methods = ['Decision Tree', 'Random Forest']
means = [tree_cv_scores.mean(), rf_cv_scores.mean()]
stds = [tree_cv_scores.std(), rf_cv_scores.std()]

x_pos = np.arange(len(methods))
axes[1].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=10, 
            color=['coral', 'forestgreen'], edgecolor='black')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(methods)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Mean CV Accuracy ± Std')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

**Key Observations:**
- ✅ **Random Forest ima višu accuracy**
- ✅ **Random Forest ima manju variance** (stabilniji)
- ✅ **Random Forest je robusniji** na različite subsets data

---

## Complete Example: Credit Card Default Prediction
```python
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CREDIT CARD DEFAULT PREDICTION - RANDOM FOREST")
print("="*60)

# ==================== 1. GENERATE SYNTHETIC DATA ====================
from sklearn.datasets import make_classification

X_credit, y_credit = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.85, 0.15],  # Imbalanced: 85% no default, 15% default
    random_state=42
)

print(f"\nDataset shape: {X_credit.shape}")
print(f"Class distribution:")
print(f"  No Default (0): {np.sum(y_credit == 0)} ({np.sum(y_credit == 0)/len(y_credit)*100:.1f}%)")
print(f"  Default (1):    {np.sum(y_credit == 1)} ({np.sum(y_credit == 1)/len(y_credit)*100:.1f}%)")
print("⚠️ Imbalanced dataset!")

# ==================== 2. TRAIN-TEST SPLIT ====================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
)

print(f"\nTrain set: {X_train_c.shape}")
print(f"Test set:  {X_test_c.shape}")

# ==================== 3. BASELINE ====================
print("\n" + "="*60)
print("BASELINE MODEL")
print("="*60)

# Majority class prediction
most_frequent_c = np.bincount(y_train_c).argmax()
y_baseline_c = np.full(len(y_test_c), most_frequent_c)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

baseline_acc = accuracy_score(y_test_c, y_baseline_c)
baseline_recall = recall_score(y_test_c, y_baseline_c, zero_division=0)

print(f"Baseline (Predict Majority Class):")
print(f"  Accuracy: {baseline_acc:.3f}")
print(f"  Recall:   {baseline_recall:.3f}")
print("\n⚠️ Accuracy looks good but recall is 0 (doesn't catch defaults!)")

# ==================== 4. RANDOM FOREST - DEFAULT ====================
print("\n" + "="*60)
print("RANDOM FOREST - DEFAULT PARAMETERS")
print("="*60)

rf_default = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf_default.fit(X_train_c, y_train_c)

y_pred_default = rf_default.predict(X_test_c)

acc_default = accuracy_score(y_test_c, y_pred_default)
precision_default = precision_score(y_test_c, y_pred_default)
recall_default = recall_score(y_test_c, y_pred_default)
f1_default = f1_score(y_test_c, y_pred_default)

print(f"Default Random Forest:")
print(f"  OOB Score:  {rf_default.oob_score_:.3f}")
print(f"  Test Accuracy:  {acc_default:.3f}")
print(f"  Test Precision: {precision_default:.3f}")
print(f"  Test Recall:    {recall_default:.3f}")
print(f"  Test F1-Score:  {f1_default:.3f}")

# ==================== 5. RANDOM FOREST - BALANCED ====================
print("\n" + "="*60)
print("RANDOM FOREST - BALANCED (class_weight)")
print("="*60)

rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf_balanced.fit(X_train_c, y_train_c)

y_pred_balanced = rf_balanced.predict(X_test_c)

acc_balanced = accuracy_score(y_test_c, y_pred_balanced)
precision_balanced = precision_score(y_test_c, y_pred_balanced)
recall_balanced = recall_score(y_test_c, y_pred_balanced)
f1_balanced = f1_score(y_test_c, y_pred_balanced)

print(f"Balanced Random Forest:")
print(f"  OOB Score:  {rf_balanced.oob_score_:.3f}")
print(f"  Test Accuracy:  {acc_balanced:.3f}")
print(f"  Test Precision: {precision_balanced:.3f}")
print(f"  Test Recall:    {recall_balanced:.3f} ✅ (Much better!)")
print(f"  Test F1-Score:  {f1_balanced:.3f}")

# ==================== 6. HYPERPARAMETER TUNING ====================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    scoring='f1',  # Optimize for F1 (better for imbalanced)
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Running RandomizedSearchCV (20 iterations)...")
rf_search.fit(X_train_c, y_train_c)

print(f"\nBest parameters: {rf_search.best_params_}")
print(f"Best CV F1-Score: {rf_search.best_score_:.3f}")

# ==================== 7. FINAL MODEL EVALUATION ====================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

best_rf = rf_search.best_estimator_
y_pred_best = best_rf.predict(X_test_c)

acc_best = accuracy_score(y_test_c, y_pred_best)
precision_best = precision_score(y_test_c, y_pred_best)
recall_best = recall_score(y_test_c, y_pred_best)
f1_best = f1_score(y_test_c, y_pred_best)

print(f"Best Random Forest (Tuned):")
print(f"  Test Accuracy:  {acc_best:.3f}")
print(f"  Test Precision: {precision_best:.3f}")
print(f"  Test Recall:    {recall_best:.3f}")
print(f"  Test F1-Score:  {f1_best:.3f}")

print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_best, 
                           target_names=['No Default', 'Default']))

# Confusion Matrix
cm_c = confusion_matrix(y_test_c, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix - Credit Card Default Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== 8. FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importances_c = best_rf.feature_importances_
feature_names_c = [f'Feature_{i+1}' for i in range(X_credit.shape[1])]

importance_df_c = pd.DataFrame({
    'Feature': feature_names_c,
    'Importance': importances_c
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df_c.head(10).to_string(index=False))

# Visualization
plt.figure(figsize=(10, 6))
top_10_c = importance_df_c.head(10)
plt.barh(top_10_c['Feature'], top_10_c['Importance'], 
         color='forestgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ==================== 9. MODEL COMPARISON ====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['Baseline', 'RF Default', 'RF Balanced', 'RF Tuned'],
    'Accuracy': [baseline_acc, acc_default, acc_balanced, acc_best],
    'Recall': [baseline_recall, recall_default, recall_balanced, recall_best],
    'F1-Score': [0.0, f1_default, f1_balanced, f1_best]
})

print(comparison_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['Accuracy', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    axes[idx].bar(comparison_df['Model'], comparison_df[metric],
                  color=['gray', 'blue', 'orange', 'green'], alpha=0.7, edgecolor='black')
    axes[idx].set_ylabel(metric)
    axes[idx].set_title(f'{metric} Comparison')
    axes[idx].tick_params(axis='x', rotation=15)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].set_ylim([0, 1])

plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: RF Tuned")
print(f"   Improvement over baseline:")
print(f"     Recall: +{recall_best - baseline_recall:.3f}")
print(f"     F1: +{f1_best:.3f}")

# ==================== 10. SAVE MODEL ====================
import joblib

joblib.dump(best_rf, 'random_forest_credit_default.pkl')
print("\n✅ Model saved: random_forest_credit_default.pkl")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Best Practices

### ✅ DO:

1. **Start sa default parametrima** - Rade dobro odmah!
2. **Use n_jobs=-1** - Paralelizacija ubrzava trening
3. **Enable oob_score=True** - Besplatna validacija
4. **Check feature importance** - Business insights
5. **Balance classes** - `class_weight='balanced'` za imbalanced data
6. **Tune n_estimators prvo** - Najlakši za tune
7. **Cross-validation** - Za pouzdanu evaluaciju
8. **Compare sa single tree** - Vidi koliko si dobio

### ❌ DON'T:

1. **Ne scale features** - Random Forest ne treba scaling!
2. **Ne koristi za ekstremno velike datasets** - Koristi LightGBM/XGBoost
3. **Ne ignoriši imbalance** - class_weight rešava
4. **Ne preteruj sa max_depth=None** - Uspori trening bez koristi
5. **Ne zaboravi n_jobs=-1** - Gubi vreme

---

## Common Pitfalls

### Greška 1: Scaling Features (Nepotrebno)
```python
# ❌ NEPOTREBNO - Random Forest ne treba scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
rf.fit(X_scaled, y_train)  # Waste of time!

# ✅ DOBRO - Direktno
rf.fit(X_train, y_train)
```

### Greška 2: Default Parameters na Imbalanced Data
```python
# ❌ LOŠE - Ignoriše minority class
rf_bad = RandomForestClassifier(n_estimators=100)
rf_bad.fit(X_imbalanced, y_imbalanced)

# ✅ DOBRO
rf_good = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf_good.fit(X_imbalanced, y_imbalanced)
```

### Greška 3: Prevelik n_estimators bez Koristi
```python
# ❌ LOŠE - 1000 trees je overkill (sporo, nema benefit)
rf_overkill = RandomForestClassifier(n_estimators=1000)

# ✅ DOBRO - 100-200 je dovoljno
rf_good = RandomForestClassifier(n_estimators=100)
```

---

## Kada Koristiti Random Forest?

### ✅ Idealno Za:

- **Structured/tabular data** (CSV, databases)
- **Mixed feature types** (numerical + categorical)
- **Non-linear relationships**
- **Brz prototip** sa odličnim rezultatima
- **Feature importance** insights
- **Robust baseline** pre kompleksnijih modela
- **Out-of-the-box performance** (malo tuninga)

### ❌ Izbegavaj Za:

- **Ekstremno veliki datasets** (milioni rows) → LightGBM/XGBoost (brži)
- **Real-time predictions** (latency critical) → Linear/Logistic (brži)
- **Interpretabilnost** (individual predictions) → Single Tree, Linear/Logistic
- **Images/Text/Sequences** → Neural Networks (CNN/RNN/Transformer)
- **Linearne veze** → Linear/Logistic (jednostavnije i brže)

---

## Random Forest vs Gradient Boosting

| Aspekt | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| **Training** | Parallelno (trees nezavisni) | Sekvenencijalno (trees zavise) |
| **Speed** | ⭐⭐⭐⭐ Brži | ⭐⭐⭐ Sporiji |
| **Performance** | ⭐⭐⭐⭐ Odličan | ⭐⭐⭐⭐⭐ Najbolji |
| **Overfitting** | ⭐⭐⭐⭐ Robustan | ⭐⭐⭐ Može overfitovati |
| **Hyperparameter Tuning** | ⭐⭐⭐⭐ Malo potrebno | ⭐⭐ Više potrebno |
| **Default Performance** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Solidno |

**Preporuka:**
- **Start sa Random Forest** - Brz i robustan
- **Ako trebaš još bolje** - Idi na XGBoost/LightGBM

---

## Rezime

| Aspekt | Opis |
|--------|------|
| **Tip** | Classification & Regression (Ensemble) |
| **Interpretabilnost** | ⭐⭐ Nizak (feature importance OK) |
| **Training Speed** | ⭐⭐⭐ Umeren (paralelizacija pomaže) |
| **Prediction Speed** | ⭐⭐⭐ Umeren |
| **Performance** | ⭐⭐⭐⭐ Excellent out-of-the-box |
| **Handles Non-linearity** | ✅ Da |
| **Feature Scaling** | ❌ Ne treba |
| **Handles Mixed Types** | ✅ Da |
| **Overfitting Risk** | ⭐⭐⭐⭐ Nizak (averaging) |
| **Stability** | ⭐⭐⭐⭐⭐ Visoka (za razliku od single tree) |
| **Hyperparameter Tuning** | ⭐⭐⭐⭐ Malo potrebno |
| **Best For** | Structured data, brz prototip, robust baseline |

---

## Quick Decision Tree
```
Start
  ↓
Structured/tabular data?
  ↓ Yes
Non-linear relationships?
  ↓ Yes
Želiš odlične rezultate brzo?
  ↓ Yes
Ne treba ekstremna interpretabilnost?
  ↓ Yes
→ RANDOM FOREST ✅

Ako trebaš još bolje:
  └─ XGBoost / LightGBM (još 2-5% bolje, ali zahteva tuning)
```

---

**Key Takeaway:** Random Forest je **go-to algoritam za structured data** kada želiš **odlične rezultate bez mnogo tuninga**. Radi "out of the box", robustan je, i daje feature importance. Za **production** i **Kaggle competitions**, može se nadmašiti sa XGBoost/LightGBM, ali Random Forest je **često dovoljan** i **mnogo brži za razvoj**! 🎯