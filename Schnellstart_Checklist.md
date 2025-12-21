# Schnellstart: Bank-Projekt Checklist (Gruppe 8)

## 🎯 Projektübersicht
- **Datensatz:** bank.csv (11.162 Samples × 17 Attribute)
- **Zielvariable:** `deposit` (binär: yes/no) - Kunde hat Termingeld abgeschlossen?
- **Aufgabe:** Vorhersage-Modell für Kunden, die wahrscheinlich ein Termingeld abschließen
- **Gewichtung:** EDA (20%) + Data Cleaning (20%) + ML (40%) + Evaluation (20%)

---

## 📋 Phase 1: Explorative Datenanalyse (20%)

### Zu tun:
- [ ] Daten einlesen mit `pd.read_csv('bank.csv')`
- [ ] `shape`, `dtypes`, `describe()` anschauen
- [ ] Fehlende Werte und Duplikate prüfen (`isnull()`, `duplicated()`)
- [ ] Für numerische Variablen:
  - [ ] Histogramme erstellen (age, balance, duration, campaign, pdays, previous)
  - [ ] Boxplots für Ausreißer (vor allem balance!)
  - [ ] Statistiken (Mean, Median, Min, Max, Quartile)
- [ ] Für kategorische Variablen:
  - [ ] Value Counts für alle Kategorien
  - [ ] Abschlussrate pro Kategorie (`groupby().deposit.value_counts()`)
- [ ] Korrelationen:
  - [ ] Korrelationsmatrix der numerischen Variablen
  - [ ] Heatmap visualisieren
  - [ ] Korrelation mit Zielvariable `deposit`
- [ ] Spezielle Befunde dokumentieren:
  - [ ] pdays = -1 bedeutet "kein früherer Kontakt" (nicht 0!)
  - [ ] Saisonalität in `month`
  - [ ] Extreme Werte in `balance` und `campaign`

### Dokuemenation:
Schreiben Sie Markdown-Text für jeden Punkt, warum interessant für Modellierung ist.

---

## 🔧 Phase 2: Data Cleaning & Feature Engineering (20%)

### Zu tun:

#### 2.1 Fehlende Werte
- [ ] Prüfen auf NaN/NULL (es sollten keine sein, aber dokumentieren!)
- [ ] pdays = -1 → neue Features:
  ```python
  df['contacted_before'] = (df['pdays'] != -1).astype(int)
  df['days_since_contact'] = df['pdays'].apply(lambda x: x if x != -1 else 0)
  ```

#### 2.2 Kategorische Variablen encodieren
- [ ] **One-Hot-Encoding für:** default, housing, loan, contact
  ```python
  df = pd.get_dummies(df, columns=['default', 'housing', 'loan', 'contact'], 
                      drop='first', dtype=int)
  ```
- [ ] **Ordinal/Label Encoding für:** education (primary=1, sec=2, tert=3)
  ```python
  education_map = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 0}
  df['education_ord'] = df['education'].map(education_map)
  ```
- [ ] **Frequency/Target Encoding für:** job (viele Kategorien)
  ```python
  job_target = df.groupby('job')['deposit'].apply(lambda x: (x == 'yes').sum() / len(x))
  df['job_target_enc'] = df['job'].map(job_target)
  ```
- [ ] **Zyklisch codieren für:** month (12 → 1 ist zyklisch!)
  ```python
  df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
  df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
  ```

#### 2.3 Neue Features generieren
- [ ] Finanzielle Features:
  ```python
  df['has_credit'] = ((df['housing']=='yes') | (df['loan']=='yes')).astype(int)
  df['credit_load'] = (df['housing']=='yes').astype(int) + (df['loan']=='yes').astype(int)
  ```
- [ ] Altersgruppen:
  ```python
  df['age_group'] = pd.cut(df['age'], bins=[0,25,35,50,60,100], labels=['18-25','26-35','36-50','51-60','60+'])
  ```
- [ ] Duration-Kategorien:
  ```python
  df['duration_cat'] = pd.cut(df['duration'], bins=[0,180,300,600,4000], 
                               labels=['short','medium','long','very_long'])
  ```
- [ ] Kontakt-Intensität:
  ```python
  df['contact_intensity'] = df['campaign'] / (df['previous'] + 1)
  ```

#### 2.4 Featurausreißer-Behandlung
- [ ] Überprüfen: Sollen Extreme in `balance` und `campaign` behalten/transformiert werden?
  ```python
  # Option 1: Behalten
  # Option 2: Log-Transformation
  df['balance_log'] = np.log1p(df['balance'] - df['balance'].min())
  df['campaign_log'] = np.log1p(df['campaign'])
  ```

#### 2.5 Skalieren numerischer Features
- [ ] RobustScaler verwenden (wegen Ausreißer in balance):
  ```python
  from sklearn.preprocessing import RobustScaler
  scaler = RobustScaler()
  numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
  df_scaled = df.copy()
  df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])
  ```

#### 2.6 Zielvariable vorbereiten
- [ ] In binär konvertieren:
  ```python
  y = (df['deposit'] == 'yes').astype(int)
  ```
- [ ] Klassen-Imbalance prüfen:
  ```python
  print(y.value_counts())
  print(y.value_counts(normalize=True))
  # Häufig: ~88% negative, ~12% positive → IMBALANCE!
  ```

#### 2.7 Train-Test-Split
- [ ] **Stratified Split** (beachtet Klassen-Verhältnis):
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  ```

---

## 🤖 Phase 3: Machine Learning (40%)

### 3.1 Unüberwachtes Lernen: K-Means Clustering

#### Ziel: Kundensegmentierung ohne Labels

- [ ] Features für Clustering wählen:
  ```python
  X_clustering = X_train[['age', 'balance', 'duration', 'campaign', 'previous']]
  ```

- [ ] Elbow-Methode anwenden:
  ```python
  from sklearn.cluster import KMeans
  
  inertias = []
  for k in range(2, 11):
      km = KMeans(n_clusters=k, random_state=42, n_init=10)
      km.fit(X_clustering_scaled)
      inertias.append(km.inertia_)
  
  plt.plot(range(2,11), inertias, 'bo-')
  plt.xlabel('k')
  plt.ylabel('Inertia')
  plt.show()
  ```

- [ ] Optimal k wählen (meist k=3 oder k=4)

- [ ] Finales Modell trainieren:
  ```python
  kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
  clusters = kmeans.fit_predict(X_clustering_scaled)
  ```

- [ ] Cluster charakterisieren:
  ```python
  for i in range(4):
      cluster_data = X_train[clusters == i]
      print(f"Cluster {i}: {len(cluster_data)} Kunden")
      print(f"  Alter: {cluster_data['age'].mean():.1f}j")
      print(f"  Balance: €{cluster_data['balance'].mean():.0f}")
      # ... weitere Charakteristiken
  ```

### 3.2 Supervised Learning: Klassifikation

**Implementieren Sie mindestens 2-3 verschiedene Algorithmen:**

#### Option A: Logistische Regression

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
```

**Vorteil:** Schnell, interpretierbar
**Nachteil:** Nur lineare Grenzen

#### Option B: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, 
                             min_samples_split=10, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Feature Importance anschauen
fi = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(fi.head(10))
```

**Vorteil:** Robust, Feature-Importance, nonlinear
**Nachteil:** Black Box, Overfitting-Gefahr

#### Option C: Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=5, random_state=42)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]
```

**Vorteil:** Hohe Accuracy, Feature-Importance
**Nachteil:** Komplex, Overfitting-Gefahr, langsamer

### 3.3 Hyperparameter-Tuning

- [ ] GridSearchCV für Random Forest:
  ```python
  from sklearn.model_selection import GridSearchCV
  
  param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth': [5, 10, 15, 20],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
  }
  
  grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
                      cv=5, scoring='roc_auc', n_jobs=-1)
  grid.fit(X_train, y_train)
  
  print(f"Beste Parameter: {grid.best_params_}")
  best_model = grid.best_estimator_
  ```

---

## 📊 Phase 4: Evaluation & Reflexion (20%)

### 4.1 Evaluationsmetriken auf Test-Set

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Für bestes Modell
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Alle Metriken
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Verwirrungsmatrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nVerwirrungsmatrix:\n{cm}")
```

### 4.2 ROC und PR-Kurven

```python
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ROC-Kurve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax1.plot(fpr, tpr, 'b-', label=f'ROC (AUC={roc_auc_score(y_test, y_pred_proba):.3f})')
ax1.plot([0, 1], [0, 1], 'r--', label='Zufälliger Klassifizierer')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC-Kurve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Precision-Recall-Kurve
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
ax2.plot(rec, prec, 'g-')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall-Kurve')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=300)
plt.show()
```

### 4.3 Fehleranalyse

```python
# Welche Samples werden falsch klassifiziert?
false_negatives = X_test[(y_pred == 0) & (y_test == 1)]
false_positives = X_test[(y_pred == 1) & (y_test == 0)]

print(f"False Negatives (echte Abschlüsse, die wir übersehen): {len(false_negatives)}")
print(f"False Positives (Fehlalarme): {len(false_positives)}")

if len(false_negatives) > 0:
    print(f"\nFN Charakteristiken:")
    print(f"  Durchschnittsdauer: {false_negatives['duration'].mean():.0f}s vs {X_test['duration'].mean():.0f}s")
    print(f"  Durchschnittsalter: {false_negatives['age'].mean():.0f}j vs {X_test['age'].mean():.0f}j")
```

### 4.4 Modell-Vergleich

```python
# Erstellen Sie eine Vergleichstabelle
results = pd.DataFrame({
    'Modell': ['Logistische Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_gb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_log),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_gb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_gb)
    ]
})

print(results)
```

### 4.5 Dokumentation der Reflexion

Schreiben Sie einen Abschnitt für jeden Punkt:

1. **Zusammenfassung der Befunde:**
   - Welche Features waren wichtig?
   - Welcher Algorithmus war am besten?
   - Welche Geschäfts-Erkenntnisse?

2. **Limitationen:**
   - Klasseneimbalance (12% positive)
   - Fehlende externe Features (Makroökonomik, KYC-Daten)
   - Zeitreihen-Aspekt ignoriert

3. **Verbesserungsmöglichkeiten:**
   - Ensemble-Methoden kombinieren
   - Kostenbasierte Klassifikation (False Positives vs. False Negatives gewichten)
   - SMOTE für Übersampling der Minderheitsklasse
   - Feature Interaction stärker untersuchen

4. **Geschäftliche Implikationen:**
   - Wie würde das Modell im Produktiveinsatz funktionieren?
   - ROI-Kalkulation?
   - Fairness-Aspekte (Diskriminierung)?

---

## 📝 Notebook-Struktur (Empfehlung)

```
1. Imports und Setup
2. Daten einlesen
3. EDA (Abschnitt mit Markdown + Code + Visualisierung)
   3.1 Grundlegende Struktur
   3.2 Deskriptive Statistik
   3.3 Visualisierungen
   3.4 Korrelationen
   3.5 Dokumentation
4. Data Cleaning & Feature Engineering
   4.1 Behandlung fehlender Werte
   4.2 Encoding kategorischer Variablen
   4.3 Feature Engineering
   4.4 Skalierung
   4.5 Train-Test-Split
5. K-Means Clustering
   5.1 Elbow-Methode
   5.2 Finales Modell
   5.3 Cluster-Charakterisierung
6. Supervised Learning
   6.1 Logistische Regression
   6.2 Random Forest
   6.3 Gradient Boosting
   6.4 Hyperparameter-Tuning
7. Evaluation
   7.1 Metriken
   7.2 ROC/PR-Kurven
   7.3 Fehleranalyse
   7.4 Modell-Vergleich
8. Reflexion und Fazit
```

---

## 🎓 Wichtige Konzepte

### TP, TN, FP, FN
- **TP (True Positive):** Model sagt Ja, Wahrheit ist Ja ✓
- **TN (True Negative):** Model sagt Nein, Wahrheit ist Nein ✓
- **FP (False Positive):** Model sagt Ja, Wahrheit ist Nein ✗ → Kosten!
- **FN (False Negative):** Model sagt Nein, Wahrheit ist Ja ✗ → Verlorener Umsatz!

### Wann welche Metrik?
- **Accuracy:** Nur wenn Klassen balanced sind
- **Precision:** Wenn FP teuer sind (z.B. Spam-Filter)
- **Recall:** Wenn FN teuer sind (z.B. Krebs-Screening)
- **F1:** Imbalanced Data (Standardmetrik für dieses Projekt!)
- **ROC-AUC:** Zum Vergleichen von Klassifizierern

### Overfitting vs. Underfitting
- **Overfitting:** Sehr gute Train-Metriken, schlechte Test-Metriken
  - Lösungen: Einfacheres Modell, weniger Features, Regularisierung
- **Underfitting:** Schlechte Train- und Test-Metriken
  - Lösungen: Komplexeres Modell, mehr Features, länger trainieren

---

## ✅ Final Checklist vor Abgabe

- [ ] Jupyter-Notebook enthält Gruppennummer und Namen + Matrikelnummern
- [ ] Alle Code-Zellen ausgeführt und Outputs sichtbar
- [ ] Ausreichend Markdown-Zellen mit Erklärungen
- [ ] Visualisierungen für alle Hauptschritte
- [ ] Evaluation-Metriken berechnet und interpretiert
- [ ] Reflexion geschrieben (mindestens 1-2 Absätze pro Punkt)
- [ ] Keine Fehler/Warnungen bei Ausführung
- [ ] Dateiname: `Gruppe8_[Vorname1_Vorname2].ipynb`
- [ ] Bis zur Deadline hochgeladen

---

## 🔗 Referenzen zu Vorlesungen

- **EDA:** 03_EXP.pdf - Kapitel Data Exploration, Data Profiling
- **Pandas:** 02_PANDAS.pdf - DataFrame, Abfragen, Aggregationen
- **Feature Engineering:** 06_FEAT.pdf - Encoding, Scaling, Imputation, PCA
- **Clustering:** 05_CLUST.pdf - K-Means, Elbow-Methode
- **Klassifikation:** 07_KLASS.pdf - Logistische Regression, Decision Trees
- **Regression & Evaluation:** 08_REGR.pdf - Metriken, Cross-Validation

---

**Viel Erfolg bei Ihrem Projekt! 🚀**