# Naive Bayes

Naive Bayes je **probabilistički algoritam** baziran na **Bayes' Theorem**. Izuzetno **brz** i **efekasan**, posebno za **text classification**!

**Zašto je Naive Bayes važan?**
- **Ekstremno brz** - Trening i prediction u milisekundama
- **Odličan za text** - Spam detection, sentiment analysis, document classification
- **Low memory** - Ne zahteva mnogo resursa
- **Radi sa high-dimensional data** - Text features, word counts
- **Probabilistic predictions** - Daje verovatnoće, ne samo labels

**Kada koristiti Naive Bayes?**
- ✅ Text classification (spam, sentiment, categories)
- ✅ Real-time predictions (veoma brz)
- ✅ Large datasets (skalira odlično)
- ✅ High-dimensional data (thousands of features)
- ✅ Baseline model za NLP tasks

**Kada NE koristiti:**
- ❌ Features su jako zavisne jedne od drugih
- ❌ Potrebna najbolja accuracy (Tree-based su bolji)
- ❌ Small datasets sa kompleksnim vezama
- ❌ Potrebna interpretabilnost individual predictions

---

## Bayes' Theorem

**Osnova Naive Bayes algoritma.**

### Formula:
```
P(A|B) = P(B|A) × P(A) / P(B)

P(A|B) = Posterior probability (verovatnoća A dato B)
P(B|A) = Likelihood (verovatnoća B dato A)
P(A)   = Prior probability (osnovna verovatnoća A)
P(B)   = Evidence (normalizacioni faktor)
```

### Za Classification:
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Želimo: P(Class|Features) - verovatnoća klase dato features

Računamo:
  - P(Class): Koliko često se klasa pojavljuje u training data
  - P(Features|Class): Koliko često se features pojavljuju u toj klasi
  - P(Features): Koliko često se features pojavljuju (često ignorišemo jer je isto za sve klase)
```

### Primer - Email Spam Detection:
```
Email: "Buy cheap watches now!"

P(Spam | "Buy cheap watches now") = ?

P(Spam) = 0.3 (30% emails su spam)
P(Ham)  = 0.7 (70% emails su ham)

P("Buy" | Spam) = 0.1
P("cheap" | Spam) = 0.2
P("watches" | Spam) = 0.15
P("now" | Spam) = 0.05

P("Buy" | Ham) = 0.01
P("cheap" | Ham) = 0.005
P("watches" | Ham) = 0.01
P("now" | Ham) = 0.02

Naive assumption: Words su nezavisne
P(Words | Spam) = P("Buy"|Spam) × P("cheap"|Spam) × P("watches"|Spam) × P("now"|Spam)
                = 0.1 × 0.2 × 0.15 × 0.05
                = 0.00015

P(Words | Ham) = 0.01 × 0.005 × 0.01 × 0.02
               = 0.000000001

P(Spam | Words) ∝ P(Words | Spam) × P(Spam) = 0.00015 × 0.3 = 0.000045
P(Ham | Words)  ∝ P(Words | Ham) × P(Ham)   = 0.000000001 × 0.7 = 0.0000000007

P(Spam | Words) >> P(Ham | Words)

Prediction: SPAM ✅
```

---

## "Naive" Assumption

**Zašto "Naive"?**

Algoritam pretpostavlja da su **sve features NEZAVISNE** jedna od druge dato klasu.
```
Naive assumption:
P(x₁, x₂, ..., xₙ | Class) = P(x₁|Class) × P(x₂|Class) × ... × P(xₙ|Class)

Realnost:
Features NISU nezavisne! (npr. "cheap" i "buy" često idu zajedno u spam-u)

Ali to ne smeta:
Naive Bayes radi ODLIČNO u praksi uprkos ovoj "naive" pretpostavci!
```

**Zašto radi?**

Iako nisu tačne verovatnoće, **relative ranking** klasa je obično ispravan:
- Ne treba nam tačna P(Spam) = 0.87, već samo da P(Spam) > P(Ham)

---

## Types of Naive Bayes

### 1. GaussianNB (Continuous Features)

**Use case:** Continuous numerical features koje prate **Gaussian (normal) distribuciju**.

**Pretpostavka:**
```
P(xᵢ | Class) = (1 / √(2πσ²)) × exp(-(xᵢ - μ)² / (2σ²))

μ = mean za feature i u Class
σ² = variance za feature i u Class
```
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== GAUSSIAN NAIVE BAYES ====================
print("="*60)
print("GAUSSIAN NAIVE BAYES")
print("="*60)

# Train (instant!)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Show probabilities
print("\nFirst 5 predictions with probabilities:")
print("True | Pred | P(Setosa) | P(Versicolor) | P(Virginica)")
for i in range(5):
    print(f" {y_test[i]}   |  {y_pred[i]}   |   {y_proba[i,0]:.3f}   |     {y_proba[i,1]:.3f}     |    {y_proba[i,2]:.3f}")
```

**Kada koristiti:**
- ✅ Continuous features (age, salary, temperature)
- ✅ Features aproximativno normalno distribuirani
- ❌ Text data (koristi MultinomialNB)

---

### 2. MultinomialNB (Count Features)

**Use case:** **Count data** - word counts, TF-IDF, frequencies.

**Glavni use case: TEXT CLASSIFICATION!**

**Pretpostavka:**
```
P(xᵢ | Class) = (count of xᵢ in Class + α) / (total count in Class + α×n_features)

α = Laplace smoothing parameter (default 1.0)
```
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# ==================== TEXT CLASSIFICATION ====================
print("\n" + "="*60)
print("MULTINOMIAL NAIVE BAYES - TEXT CLASSIFICATION")
print("="*60)

# Sample text data
texts = [
    "I love this movie, it's amazing",
    "Great film, highly recommend",
    "Best movie ever, fantastic",
    "Terrible movie, waste of time",
    "Awful film, don't watch it",
    "Horrible, the worst movie",
    "Good movie, enjoyed it",
    "Bad movie, not recommended"
]
labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1=positive, 0=negative

# Vectorize text (convert to word counts)
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_text.shape}")

# Train
mnb = MultinomialNB()
mnb.fit(X_text, labels)

# Test on new texts
test_texts = [
    "Great movie, loved it",
    "Terrible, waste of money"
]
X_test_text = vectorizer.transform(test_texts)
predictions = mnb.predict(X_test_text)
probabilities = mnb.predict_proba(X_test_text)

print("\nPredictions:")
for i, text in enumerate(test_texts):
    sentiment = "POSITIVE" if predictions[i] == 1 else "NEGATIVE"
    print(f"\nText: '{text}'")
    print(f"  Prediction: {sentiment}")
    print(f"  P(Negative): {probabilities[i,0]:.3f}")
    print(f"  P(Positive): {probabilities[i,1]:.3f}")
```

**Kada koristiti:**
- ✅ **Text classification** (spam, sentiment, topics)
- ✅ Word counts, TF-IDF features
- ✅ Any count/frequency data
- ❌ Continuous features (koristi GaussianNB)

---

### 3. BernoulliNB (Binary Features)

**Use case:** **Binary features** (0/1, True/False, present/absent).

**Primer:** Document contains word or not (ne broji koliko puta).
```python
from sklearn.naive_bayes import BernoulliNB

# ==================== BERNOULLI NAIVE BAYES ====================
print("\n" + "="*60)
print("BERNOULLI NAIVE BAYES")
print("="*60)

# Binary vectorizer (word present=1, absent=0)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_binary = CountVectorizer(binary=True)  # Binary mode!
X_binary = vectorizer_binary.fit_transform(texts)

print("Binary features (word present/absent):")
print(X_binary.toarray()[:3])  # First 3 documents

# Train
bnb = BernoulliNB()
bnb.fit(X_binary, labels)

# Test
X_test_binary = vectorizer_binary.transform(test_texts)
predictions_bnb = bnb.predict(X_test_binary)

print("\nBernoulli predictions:")
for i, text in enumerate(test_texts):
    sentiment = "POSITIVE" if predictions_bnb[i] == 1 else "NEGATIVE"
    print(f"'{text}' → {sentiment}")
```

**Kada koristiti:**
- ✅ Binary features
- ✅ Document classification (word present/absent)
- ⚠️ Obično MultinomialNB je bolji za text

---

### 4. ComplementNB (Imbalanced Text Data)

**Use case:** MultinomialNB za **imbalanced datasets**.

**Razlika:** Koristi complement klase (sve ostale klase) umesto same klase.
```python
from sklearn.naive_bayes import ComplementNB

# ==================== COMPLEMENT NAIVE BAYES ====================
print("\n" + "="*60)
print("COMPLEMENT NAIVE BAYES (for imbalanced data)")
print("="*60)

# Imbalanced data (90% negative, 10% positive)
texts_imb = texts + ["Bad movie"] * 10  # Add more negatives
labels_imb = labels + [0] * 10

print(f"Class distribution:")
print(f"  Negative (0): {labels_imb.count(0)}")
print(f"  Positive (1): {labels_imb.count(1)}")

X_imb = vectorizer.fit_transform(texts_imb)

# MultinomialNB
mnb_imb = MultinomialNB()
mnb_imb.fit(X_imb, labels_imb)

# ComplementNB
cnb = ComplementNB()
cnb.fit(X_imb, labels_imb)

# Compare on balanced test set
from sklearn.metrics import classification_report

print("\nMultinomialNB on imbalanced data:")
print(classification_report([0, 1, 0, 1], 
                           mnb_imb.predict(vectorizer.transform(test_texts + test_texts)),
                           target_names=['Negative', 'Positive']))

print("\nComplementNB on imbalanced data:")
print(classification_report([0, 1, 0, 1],
                           cnb.predict(vectorizer.transform(test_texts + test_texts)),
                           target_names=['Negative', 'Positive']))
```

**Kada koristiti:**
- ✅ Imbalanced text data
- ✅ MultinomialNB daje loše rezultate na minority class
- ⚠️ Start sa MultinomialNB, probaj ComplementNB ako minority class loš

---

## Laplace Smoothing (Add-One Smoothing)

**Problem:** Šta ako word **nikad nije viđena** u training data za neku klasu?
```
P("unicorn" | Spam) = 0 / 1000 = 0

Problem: 0 × anything = 0
Ceo proizvod postaje 0! 🚨
```

**Rešenje: Laplace Smoothing**
```
P(word | Class) = (count + α) / (total + α × vocabulary_size)

α = smoothing parameter (default 1.0)

Efekat:
  - Nikad 0 probability
  - Unseen words dobijaju malu verovatnoću
```
```python
# ==================== LAPLACE SMOOTHING ====================
print("\n" + "="*60)
print("LAPLACE SMOOTHING")
print("="*60)

# Test different alpha values
alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

for alpha in alphas:
    mnb_alpha = MultinomialNB(alpha=alpha)
    mnb_alpha.fit(X_text, labels)
    
    # Predict
    preds = mnb_alpha.predict(X_test_text)
    
    print(f"alpha={alpha}: Predictions = {preds}")

print("\nObservation:")
print("  - Small alpha (0.1): Less smoothing, trusts training data more")
print("  - Large alpha (5.0): More smoothing, less trust in training data")
print("  - Default (1.0): Usually works well")
```

**Preporuka:** Ostavi default `alpha=1.0`, tune samo ako vidiš problem sa rare words.

---

## Complete Example: Spam Email Detection
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SPAM EMAIL DETECTION - NAIVE BAYES")
print("="*60)

# ==================== 1. CREATE DATASET ====================
# Simulate spam/ham emails
spam_emails = [
    "Win free money now! Click here",
    "Congratulations! You won $1000000",
    "Buy cheap watches online",
    "Get viagra pills discount",
    "Earn money from home fast",
    "Free lottery winner notification",
    "Claim your prize now immediately",
    "Buy cheap software licenses",
    "Get rich quick opportunity",
    "Work from home make money"
] * 20  # Repeat to get more data

ham_emails = [
    "Hi, how are you doing today?",
    "Meeting scheduled for tomorrow at 3pm",
    "Please review the attached document",
    "Thank you for your email",
    "Let's catch up over coffee sometime",
    "Project deadline is next Friday",
    "Can you send me the report?",
    "Happy birthday! Hope you have great day",
    "Reminder about team lunch today",
    "Please confirm your attendance"
] * 20

# Combine
emails = spam_emails + ham_emails
labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1=spam, 0=ham

print(f"\nDataset size: {len(emails)} emails")
print(f"  Spam: {labels.count(1)}")
print(f"  Ham:  {labels.count(0)}")

# ==================== 2. TRAIN-TEST SPLIT ====================
from sklearn.model_selection import train_test_split

X_train_email, X_test_email, y_train_email, y_test_email = train_test_split(
    emails, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrain: {len(X_train_email)}")
print(f"Test:  {len(X_test_email)}")

# ==================== 3. FEATURE EXTRACTION ====================
print("\n" + "="*60)
print("FEATURE EXTRACTION - TF-IDF")
print("="*60)

# TF-IDF vectorization (better than simple counts)
tfidf = TfidfVectorizer(
    max_features=100,      # Top 100 words
    stop_words='english',  # Remove common words (the, is, at, ...)
    lowercase=True
)

X_train_tfidf = tfidf.fit_transform(X_train_email)
X_test_tfidf = tfidf.transform(X_test_email)

print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Feature matrix: {X_train_tfidf.shape}")

# Show top features
feature_names = tfidf.get_feature_names_out()
print(f"\nSample features: {feature_names[:20].tolist()}")

# ==================== 4. BASELINE - MOST FREQUENT ====================
print("\n" + "="*60)
print("BASELINE - Predict Most Frequent Class")
print("="*60)

most_frequent = max(set(y_train_email), key=y_train_email.count)
y_baseline = [most_frequent] * len(y_test_email)

baseline_acc = accuracy_score(y_test_email, y_baseline)
print(f"Baseline Accuracy: {baseline_acc:.3f}")

# ==================== 5. MULTINOMIAL NAIVE BAYES ====================
print("\n" + "="*60)
print("MULTINOMIAL NAIVE BAYES")
print("="*60)

# Train
mnb_spam = MultinomialNB()
mnb_spam.fit(X_train_tfidf, y_train_email)

# Predictions
y_pred_mnb = mnb_spam.predict(X_test_tfidf)
y_proba_mnb = mnb_spam.predict_proba(X_test_tfidf)

# Metrics
acc_mnb = accuracy_score(y_test_email, y_pred_mnb)
print(f"Test Accuracy: {acc_mnb:.3f}")
print(f"Improvement over baseline: +{(acc_mnb - baseline_acc):.3f}")

print("\nClassification Report:")
print(classification_report(y_test_email, y_pred_mnb, 
                           target_names=['Ham', 'Spam']))

# Confusion Matrix
cm_email = confusion_matrix(y_test_email, y_pred_mnb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_email, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Spam Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== 6. FEATURE IMPORTANCE ====================
print("\n" + "="*60)
print("MOST INDICATIVE WORDS")
print("="*60)

# Get log probabilities for each class
log_prob_spam = mnb_spam.feature_log_prob_[1]  # Class 1 (spam)
log_prob_ham = mnb_spam.feature_log_prob_[0]   # Class 0 (ham)

# Most indicative of spam
spam_words_idx = np.argsort(log_prob_spam)[-10:][::-1]
print("\nTop 10 SPAM indicators:")
for idx in spam_words_idx:
    word = feature_names[idx]
    prob = np.exp(log_prob_spam[idx])
    print(f"  {word:15s}: {prob:.4f}")

# Most indicative of ham
ham_words_idx = np.argsort(log_prob_ham)[-10:][::-1]
print("\nTop 10 HAM indicators:")
for idx in ham_words_idx:
    word = feature_names[idx]
    prob = np.exp(log_prob_ham[idx])
    print(f"  {word:15s}: {prob:.4f}")

# ==================== 7. TEST ON NEW EMAILS ====================
print("\n" + "="*60)
print("PREDICTIONS ON NEW EMAILS")
print("="*60)

new_emails = [
    "Congratulations! You have won a free iPhone. Claim now!",
    "Hi John, can we schedule a meeting for next week?",
    "Buy cheap medications online without prescription",
    "Reminder: Your dentist appointment is tomorrow at 2pm"
]

new_tfidf = tfidf.transform(new_emails)
new_predictions = mnb_spam.predict(new_tfidf)
new_probabilities = mnb_spam.predict_proba(new_tfidf)

for i, email in enumerate(new_emails):
    label = "SPAM" if new_predictions[i] == 1 else "HAM"
    prob_spam = new_probabilities[i, 1]
    
    print(f"\nEmail: '{email}'")
    print(f"  Prediction: {label}")
    print(f"  P(Spam): {prob_spam:.3f}")

# ==================== 8. MODEL COMPARISON ====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Test GaussianNB (will be worse for text)
gnb_spam = GaussianNB()
gnb_spam.fit(X_train_tfidf.toarray(), y_train_email)  # Need dense array
acc_gnb = gnb_spam.score(X_test_tfidf.toarray(), y_test_email)

# Test ComplementNB
cnb_spam = ComplementNB()
cnb_spam.fit(X_train_tfidf, y_train_email)
acc_cnb = cnb_spam.score(X_test_tfidf, y_test_email)

comparison = pd.DataFrame({
    'Model': ['Baseline', 'GaussianNB', 'MultinomialNB', 'ComplementNB'],
    'Accuracy': [baseline_acc, acc_gnb, acc_mnb, acc_cnb]
})

print("\n" + comparison.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
plt.bar(comparison['Model'], comparison['Accuracy'],
        color=['gray', 'orange', 'green', 'blue'], alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Spam Detection')
plt.ylim([0.5, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print(f"\n🏆 Best Model: MultinomialNB")
print(f"   Test Accuracy: {acc_mnb:.3f}")

# ==================== 9. SAVE MODEL ====================
import joblib

joblib.dump(mnb_spam, 'naive_bayes_spam.pkl')
joblib.dump(tfidf, 'tfidf_spam.pkl')

print("\n✅ Model saved: naive_bayes_spam.pkl")
print("✅ Vectorizer saved: tfidf_spam.pkl")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! ✅")
print("="*60)
```

---

## Key Hyperparameters
```python
# MultinomialNB (najčešći za text)
MultinomialNB(
    alpha=1.0,           # ⭐ Laplace smoothing (0.1-10)
    fit_prior=True,      # Learn class priors from data
    class_prior=None     # Manual class priors (usually leave None)
)

# GaussianNB
GaussianNB(
    priors=None,         # Manual class priors
    var_smoothing=1e-9   # Variance smoothing (stability)
)

# ComplementNB
ComplementNB(
    alpha=1.0,           # Laplace smoothing
    norm=False           # Normalize weights
)
```

**Tuning Strategy:**
1. **alpha**: Obično 1.0 je OK, probaj 0.1-10 ako treba
2. Ostalo: Default values su dovoljni

---

## Pros and Cons

### ✅ Prednosti:

1. **Ekstremno brz** - Trening i prediction u milisekundama
2. **Skalabilan** - Radi na ogromnim datasets
3. **Low memory** - Ne zahteva mnogo resursa
4. **Odličan za text** - Best for spam, sentiment, document classification
5. **Probabilistic** - Daje P(Class|Features)
6. **Works sa high dimensions** - Thousands of features OK
7. **Simple** - Lak za razumevanje i implementaciju

### ❌ Mane:

1. **"Naive" assumption** - Features aren't actually independent
2. **Ne radi dobro** sa features koje su jako zavisne
3. **Slabiji od Tree-based** za structured data (CSV)
4. **Continuous features problem** - GaussianNB pretpostavlja normalnu distribuciju
5. **Zero probability** - Needs Laplace smoothing

---

## Best Practices

### ✅ DO:

1. **MultinomialNB za text** - Gotovo uvek najbolji izbor
2. **TF-IDF umesto CountVectorizer** - Bolje results
3. **Remove stop words** - 'the', 'is', 'at' ne pomažu
4. **Laplace smoothing** - Default alpha=1.0 je OK
5. **Baseline model za NLP** - Brz test pre kompleksnijih modela
6. **Check class balance** - Ako imbalanced, probaj ComplementNB

### ❌ DON'T:

1. **Ne koristi za structured/tabular data** - Tree-based su bolji
2. **Ne koristi GaussianNB za text** - MultinomialNB je pravi
3. **Ne očekuj najbolje performanse** - Good baseline, ali ne SOTA
4. **Ne zaboravi feature extraction** - Text mora biti vectorizovan
5. **Ne ignoriši feature engineering** - TF-IDF, n-grams pomažu

---

## Common Pitfalls

### Greška 1: Wrong Naive Bayes Type
```python
# ❌ LOŠE - GaussianNB za text data
gnb_bad = GaussianNB()
gnb_bad.fit(X_text_counts, y)  # Text features nisu Gaussian!

# ✅ DOBRO - MultinomialNB za text
mnb_good = MultinomialNB()
mnb_good.fit(X_text_counts, y)
```

### Greška 2: No Feature Extraction
```python
# ❌ LOŠE - Ne možeš direktno trenirati na text
texts = ["hello world", "goodbye world"]
mnb.fit(texts, labels)  # Error!

# ✅ DOBRO - Vectorize prvo
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
mnb.fit(X, labels)
```

### Greška 3: Ignoring Imbalanced Data
```python
# ❌ LOŠE - MultinomialNB na jako imbalanced
# 95% ham, 5% spam → loš recall na spam

# ✅ DOBRO - ComplementNB
cnb = ComplementNB()
cnb.fit(X_imbalanced, y_imbalanced)
```

---

## Kada Koristiti Naive Bayes?

### ✅ Idealno Za:

- **Text classification** (spam, sentiment, categories)
- **Real-time predictions** (veoma brz)
- **Large datasets** (millions of samples)
- **High-dimensional** (thousands of features)
- **Baseline NLP model**
- **Low resources** (embedded systems)

### ❌ Izbegavaj Za:

- **Structured/tabular data** → Tree-based (Random Forest, XGBoost)
- **Features su jako zavisne** → Other algorithms
- **Potrebna najbolja accuracy** → Deep Learning (BERT, etc.)
- **Small text datasets** → Try SVM or Logistic too

---

## Naive Bayes vs Other Algorithms

| Aspekt | Naive Bayes | Logistic Regression | SVM | Random Forest |
|--------|-------------|---------------------|-----|---------------|
| **Training Speed** | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Prediction Speed** | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ Low | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Text Classification** | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Structured Data** | ⭐⭐ Weak | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Interpretability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **High Dimensions** | ✅ Excellent | ✅ Good | ⭐⭐⭐ | ⭐⭐⭐ |

---

## Rezime

| Aspekt | Naive Bayes |
|--------|-------------|
| **Tip** | Probabilistic Classification |
| **Interpretabilnost** | ⭐⭐⭐ Umeren (probabilities vidljive) |
| **Training Speed** | ⭐⭐⭐⭐⭐ Instant |
| **Prediction Speed** | ⭐⭐⭐⭐⭐ Instant |
| **Memory** | ⭐⭐⭐⭐⭐ Very low |
| **Performance (Text)** | ⭐⭐⭐⭐ Excellent |
| **Performance (Structured)** | ⭐⭐ Weak |
| **Feature Independence** | ⚠️ Naivna pretpostavka |
| **Best For** | Text classification, real-time, large scale |

---

## Quick Decision Tree
```
Start
  ↓
Text classification?
  ↓ Yes
Need fast training/prediction?
  ↓ Yes
Large dataset OK?
  ↓ Yes
→ NAIVE BAYES (MultinomialNB) ✅

Ako structured/tabular data:
  └─ Tree-based (Random Forest, XGBoost)

Ako treba najbolja accuracy:
  └─ Deep Learning (BERT) ili Ensemble

Ako continuous features:
  └─ GaussianNB (probaj, ali Tree-based često bolji)
```

---

**Key Takeaway:** Naive Bayes je **go-to algoritam za text classification** - ekstremno brz, low memory, skalabilan. **MultinomialNB** dominira spam detection i sentiment analysis. Uprkos "naive" pretpostavci, radi **odlično u praksi**! Za structured data, tree-based algoritmi su bolji, ali za **text** - Naive Bayes je **excellent baseline**! 🎯