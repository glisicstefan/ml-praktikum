# ML Praktikum 🚀

**Sveobuhvatan Machine Learning praktikum na srpskom jeziku** sa praktičnim Python primerima i detaljnim objašnjenjima.

---

## 📖 O Projektu

ML Praktikum je **besplatni, open-source** Machine Learning resurs kreiran za:
- 🎓 Studente koji uče Machine Learning
- 💼 Data Science praktičare koji žele da osvježe znanje  
- 🔬 Sve koji žele dubinsko razumevanje ML algoritama
- 🇷🇸 Govornike srpskog jezika koji preferiraju sadržaj na maternjem jeziku

### 📝 Stil i Pristup

- **🇷🇸 Jezik:** Srpski sa engleskim tehničkim terminima (npr. "overfitting", "cross-validation", "hyperparameter")
- **💻 Kod:** Obilje Python primera - 800 do 1000+ linija po lekciji
- **📊 Vizualizacije:** Matplotlib/Seaborn grafikoni za svaki koncept
- **✅ Praksa:** Best practices, common pitfalls, decision frameworks
- **🎯 Fokus:** Praktična primena, ne samo teorija
- **🤖 Kreiran:** Uz pomoć Claude AI (Anthropic)

**Napomena:** Praktikum je pisan na srpskom jeziku, ali koristi mnogo engleskih tehničkih termina koji su standardni u industriji (što olakšava prelazak na englesku literaturu).

**💡 O Kodu:** Većina code snippeta u praktikumu sadrži radne primere sa vizualizacijama koje možeš direktno pokrenuti. Međutim, neki delovi prikazuju kod kao **ilustraciju pristupa** i zahtevaju da adaptiraš primere za svoje podatke. Fokus je na razumevanju koncepata i pristupa, ne na execution svakog snippeta.

---

#### ✅ Kompletno (Spremno za korišćenje):

**📊 01_Data_Preprocessing** (10 lekcija)
- Data Cleaning, EDA, Transformations, Encoding, Scaling, Splitting, Feature Engineering, Feature Selection, Handling Imbalanced Data, ML Pipeline

**🎯 05_Model_Evaluation_and_Tuning** (7 lekcija)
- Classification Metrics, Regression Metrics, Cross-Validation, Bias-Variance Tradeoff, Hyperparameter Tuning, Regularization, Model Interpretation

#### 🚧 U Izradi:

**🤖 02_Supervised_Learning** (9 lekcija planirano)
- Linear Regression, Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, Algorithm Comparison

**🔍 03_Unsupervised_Learning** (planirano)

**🧠 04_Deep_Learning** (planirano)

**🚀 06_Deployment** (planirano)

---

## 📂 Struktura Projekta
```
ML-Praktikum/
│
├── README.md                
├── LICENSE                          
├── requirements.txt                           
│
├── 00_ML_Workflow.md             
│
├── 01_Data_Preprocessing/          
│   ├── 01_Data_Cleaning.md
│   ├── 02_Exploratory_Data_Analysis.md
│   ├── 03_Data_Transformation.md
│   ├── 04_Encoding_Techniques.md
│   ├── 05_Feature_Scaling.md
│   ├── 06_Train_Test_Split.md
│   ├── 07_Feature_Creation.md 
│   ├── 08_Feature_Selection.md
│   ├── 09_Handling_Imbalanced_Data.md
│   └── 10_ML_Pipeline.md
│
├── 02_Supervised_Learning/          
│   ├── 01_Linear_Regression.md
│   ├── 02_Logistic_Regression.md
│   ├── 03_Decision_Trees.md
│   ├── 04_Random_Forest.md
│   ├── 05_Gradient_Boosting.md
│   ├── 06_Support_Vector_Machines.md
│   ├── 07_K_Nearest_Neighbors.md
│   ├── 08_Naive_Bayes.md
│   └── 09_Algorithm_Comparison.md
│
├── 03_Unsupervised_Learning/         
│   └── (u izradi)
│
├── 04_Deep_Learning/                 
│   └── (u izradi)
│
├── 05_Model_Evaluation_and_Tuning/    
│   ├── 01_Classification_Metrics.md
│   ├── 02_Regression_Metrics.md
│   ├── 03_Cross_Validation.md
│   ├── 04_Bias_Variance_Tradeoff.md
│   ├── 05_Hyperparameter_Tuning.md
│   ├── 06_Regularization.md
│   └── 07_Model_Interpretation.md
│
└── 06_Deployment/
    ├── 01_Model_Serialization.md
    ├── 02_API_Development_FastAPI.md
    ├── 03_Containerization_Docker.md
    ├── 04_Cloud_Deployment.md
    ├── 05_Model_Monitoring.md
    └── 06_MLOps_Best_Practices.md
```

---

## 📚 Kako Koristiti Ovaj Praktikum

### Preporučena Putanja (Za Početnike):

1. **Start** → Pročitaj [00_ML_Workflow.md](00_ML_Workflow.md) za big picture
2. **Then** → `01_Data_Preprocessing/` (sve lekcije redom)
3. **Then** → `02_Supervised_Learning/` (izaberi algoritam koji te zanima)
4. **Then** → `05_Model_Evaluation_and_Tuning/` (nauči da evaluiraš i optimizuješ)
5. **Finally** → Radi svoje projekte!

### Saveti Za Učenje:

✅ **Čitaj teoriju pažljivo** - razumevanje koncepata je ključno  
✅ **Eksperimentišu sa kodom** - prilagodi primere svojim podacima  
✅ **Pravi beleške** - zapiši ključne koncepte svojim rečima  
✅ **Radi projekte** - primeni naučeno na realnim podacima  
✅ **Pitaj pitanja** - otvori GitHub Issue ako nešto nije jasno  

---

## 🛠️ Tehnologije i Biblioteke

Praktikum pokriva sledeće Python biblioteke:

**Osnove:**
- `numpy` - Rad sa nizovima i matricama
- `pandas` - Manipulacija tabelarnim podacima
- `matplotlib` - Vizualizacije
- `seaborn` - Statistički grafikoni

**Machine Learning:**
- `scikit-learn` - Glavni ML toolkit
- `xgboost` - Gradient boosting
- `lightgbm` - Brži gradient boosting
- `catboost` - Gradient boosting sa kategoričkim features

**Interpretacija:**
- `shap` - Model interpretation
- `lime` - Local interpretable explanations

**Tuning:**
- `optuna` - Hyperparameter optimization

**Statistika:**
- `scipy` - Statistički testovi
- `statsmodels` - Detaljne statističke analize

---

## 📜 Licenca

Ovaj projekat je licenciran pod **MIT licencom** - slobodno koristite, delite i modifikujte!

Videti [LICENSE](LICENSE) fajl za detalje.

---

## 🙏 Zahvalnice

### Kreacija
Ovaj praktikum je kreiran uz pomoć **Claude AI** (Anthropic) - AI asistent koji je pomogao u strukturiranju, pisanju i optimizaciji svih lekcija.

### Inspiracija
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)
- Scikit-learn dokumentacija i primeri

---

## 📬 Kontakt

- **GitHub Issues:** [Prijavite problem ili predložite feature](https://github.com/glisicstefan/ML-Praktikum/issues)
- **Email:** stefanglisic08@gmail.com

---

<div align="center">

**Srećno učenje! 🚀📚**

</div>