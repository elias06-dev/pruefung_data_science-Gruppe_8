# Umfassende Anleitung: Data-Science-Projekt mit dem Bank-Datensatz (Gruppe 8)

## Übersicht

Lösung der Projektaufgabe für den Bank-Datensatz.
Vier gewichtete Teilaufgaben, die eine vollständige Data-Science-Pipeline abdecken: 
- Explorative Datenanalyse (20%), 
- Data Cleaning und Feature Engineering (20%), 
- Machine Learning mit überwachtem und unüberwachtem Lernen (40%), 
- sowie Evaluation und Reflexion (20%).

### Datensatz-Übersicht

Der Bank-Datensatz enthält 11.162 Kundenkontakte mit 17 Attributen:

**Kundendaten:**
- `age`: Alter (numerisch, 18-95 Jahre)
- `job`: Berufsfeld (kategorisch, z.B. admin, technician)
- `marital`: Familienstand (kategorisch: married, single, divorced)
- `education`: Bildungsniveau (kategorisch: primary, secondary, tertiary)
- `default`: Kreditausfallstatus (binär: yes/no)
- `balance`: Durchschnittliches Jahreskontostand in Euro (numerisch)
- `housing`: Hypothekenkreditstatus (binär: yes/no)
- `loan`: Persönliches Darlehen (binär: yes/no)

**Kontaktdaten der aktuellen Kampagne:**
- `contact`: Kontakttyp (kategorisch: telephone, cellular, unknown)
- `day`: Letzter Kontakttag des Monats (numerisch, 1-31)
- `month`: Letzter Kontaktmonat (kategorisch: jan-dec)
- `duration`: Anrufdauer in Sekunden (numerisch)
- `campaign`: Anzahl der Kontakte in dieser Kampagne (numerisch)

**Historische Daten:**
- `pdays`: Tage seit letztem Kontakt (numerisch, -1 = kein vorheriger Kontakt)
- `previous`: Anzahl früherer Kontakte (numerisch)
- `poutcome`: Ergebnis der vorherigen Kampagne (kategorisch: unknown, success, failure)

**Zielvariable:**
- `deposit`: Kunde hat Termingeld abgeschlossen? (binär: yes/no) — **Dies ist Ihre Vorhersagevariable**

---

## Teil 1: Explorative Datenanalyse (20%)

### Ziel
Sie sollen den Datensatz gründlich untersuchen, um Datenqualität, Verteilungen, Muster und Anomalien zu identifizieren.

### 1.1 Grundlegende Datenstruktur-Analyse

**Schritt 1: Daten einlesen und inspizieren**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV-Datei einlesen
df = pd.read_csv('bank.csv')

# Grundlegende Informationen
print(f"Datensatzform: {df.shape}")  # (11162, 17)
print(f"\nDatentypen:\n{df.dtypes}")
print(f"\nFehlende Werte:\n{df.isnull().sum()}")
print(f"\nErste 10 Zeilen:\n{df.head(10)}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
```

**Was Sie tun:**
- Dokumentieren Sie die Anzahl der Zeilen (Samples) und Spalten (Features)
- Notieren Sie Datentypen und fehlende Werte
- Überprüfen Sie auf Duplikate (sehr selten im Banking, aber zu dokumentieren)

**Erwartete Befunde:**
- 11.162 Zeilen × 17 Spalten
- Keine fehlenden Werte (vollständiger Datensatz)
- 8 numerische, 9 kategorische Attribute
- Zielvariable `deposit` ist binär (yes/no)

### 1.2 Deskriptive Statistiken

**Schritt 2: Univariate Analyse numerischer Variablen**

```python
# Detaillierte Statistiken
print(df.describe(include='all'))

# Für kategorische Variablen
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col} - Unique Werte: {df[col].nunique()}")
    print(df[col].value_counts())
```

**Was Sie analysieren:**

a) **Lagemaße** (Mittelwert, Median, Modus)
   - Alter: Mittelwert 41 Jahre, Median 39 Jahre → relativ symmetrisch
   - Balance: Mittelwert 1.529€, Median 550€ → rechtsschief (Mittelwert > Median)
   - Duration: Mittelwert 372 Sekunden (ca. 6 Min) → starke Streuung

b) **Streuungsmaße** (Standardabweichung, Min/Max, Quartile)
   - Balance: Std 3.225€, Range [-6847, 81204] → sehr hohe Variabilität
   - Age: Std 11,9 Jahre → moderates Alter-Spektrum
   - Campaign: Q1=1, Q3=3, Max=63 → einzelne Kunden mit extremem Kontaktnummern

c) **Kategorische Verteilungen**
   - `deposit`: Vergleichen Sie die Anteile (Klasseneverteilung für Klassifikation)
   - `job`: Notieren Sie häufigste/seltenste Berufsgruppen
   - `education`: Dokumentieren Sie Anteil pro Bildungsniveau

### 1.3 Verteilungsanalyse mit Visualisierungen

**Schritt 3: Histogramme und Boxplots**

```python
# Subplot-Grid für numerische Variablen
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Histogram: {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequenz')

plt.tight_layout()
plt.savefig('01_histogramme.png', dpi=300)
plt.show()

# Boxplots für Ausreißer-Analyse
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df[col])
    axes[idx].set_title(f'Boxplot: {col}')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.savefig('02_boxplots.png', dpi=300)
plt.show()
```

**Was Sie analysieren:**

- **Normalverteilung:** Überprüfen Sie visuell mit Q-Q-Plots
  - `age`: Nähe zur Normalverteilung
  - `balance`: Stark rechtsschief (Log-Transformation später relevant!)
  - `duration`: Rechtsschief, viele kurze Anrufe

- **Ausreißer (Outlier):**
  - `balance`: Einzelne Kunden mit sehr hohem/niedrigem Kontostand
  - `campaign`: Max 63 Kontakte (extreme Ausreißer)
  - `pdays`: -1 bedeutet "kein früherer Kontakt" (spezielle Kodierung!)

- **Bimodalität oder Multi-Modalität:**
  - `campaign`: Peak bei 1-3, dann lange Tail
  - `pdays`: Großer Peak bei -1

### 1.4 Bivariate Analyse und Korrelationen

**Schritt 4: Zusammenhänge identifizieren**

```python
# Korrelationsmatrix (nur numerische Variablen)
correlation_matrix = df[numeric_cols].corr()

# Heatmap visualisieren
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Korrelationsmatrix numerischer Variablen')
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=300)
plt.show()

# Korrelation mit Zielvariable
df['deposit_binary'] = (df['deposit'] == 'yes').astype(int)
correlations_with_target = df[numeric_cols + ['deposit_binary']].corr()['deposit_binary'].sort_values(ascending=False)
print("\nKorrelation mit Zielvariable 'deposit':")
print(correlations_with_target)
```

**Was Sie dokumentieren:**

- **Positive Korrelationen mit Zielgröße:**
  - `duration`: Längere Anrufe → höhere Abschlusswahrscheinlichkeit (erwartet)
  - `previous`: Mehr frühere Kontakte → höhere Erfolgsquote

- **Negative Korrelationen:**
  - `pdays`: Länger Abstand seit letztem Kontakt → niedrigere Quote
  - `campaign`: Zu viele Kontakte → Ungeduld (umgekehrtes U, nicht linear!)

- **Multikollinearität:**
  - `age` und `balance`: Schwache Korrelation
  - `campaign` und `pdays`: Inverse Beziehung (mehr Kampagnen = keine vorherigen Kontakte nötig)

**Schritt 5: Kategorische Variablen vs. Zielvariable**

```python
# Für jede kategorische Variable: Abschlussrate berechnen
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']:
    print(f"\n{col} vs deposit:")
    crosstab = pd.crosstab(df[col], df['deposit'], margins=True, margins_name='Gesamt')
    conversion_rate = df.groupby(col)['deposit'].apply(lambda x: (x == 'yes').sum() / len(x) * 100)
    print(f"Abschlussrate pro Kategorie:\n{conversion_rate.sort_values(ascending=False)}\n")
```

**Wichtige Befunde:**
- `poutcome`: Wenn Vorkampagne erfolgreich war → sehr hohe Quote (>50%)
- `contact`: Cellular vs. unknown → unterschiedliche Erfolgsquoten
- `education`: Tertiary/Secondary → potentiell höhere Quote als Primary
- `default`: Mit Credit Default → niedrigere Quote (Risiko)
- `housing` + `loan`: Kombination relevant (finanzieller Status)

### 1.5 Spezielle Analysen

**Schritt 6: Anomalien und Muster**

```python
# Datum-Analyse (month und day)
plt.figure(figsize=(12, 4))
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df['month_cat'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
df.groupby('month_cat')['deposit'].apply(lambda x: (x == 'yes').sum() / len(x) * 100).plot(kind='bar')
plt.title('Abschlussrate nach Monat')
plt.ylabel('Erfolgsrate (%)')
plt.tight_layout()
plt.savefig('04_monthly_conversion.png', dpi=300)
plt.show()

# Spezialfall: pdays = -1 Handler
print(f"\nKunden ohne früheren Kontakt (pdays = -1): {(df['pdays'] == -1).sum() / len(df) * 100:.1f}%")
print(f"Abschlussrate dieser Kunden: {(df[df['pdays'] == -1]['deposit'] == 'yes').sum() / (df['pdays'] == -1).sum() * 100:.1f}%")
```

**Dokumentieren Sie:**
- Saisonalität: Mai ist der erfolgreichste Monat (Kampagnenstart)
- pdays-Spezialfall: -1 als fehlende Information (nicht Wert 0)
- Zeitpunkt-Effekt: Kontakte am Monatsanfang vs. -ende

---

## Teil 2: Data Cleaning und Feature Engineering (20%)

### Ziel
Bereiten Sie die Daten für Machine Learning vor: Fehlende Werte, Kodierung, Skalierung, neue Features.

### 2.1 Behandlung fehlender Werte

```python
# Da keine echten fehlenden Werte:
# Aber pdays = -1 ist spezielle Kodierung für "kein vorheriger Kontakt"
# Optionen:
# 1. Behalten als ist
# 2. Separate Feature: "contacted_before" (yes/no)
# 3. In einen anderen Wert transformieren

# Lösung: Neue Feature erstellen
df['contacted_before'] = (df['pdays'] != -1).astype(int)
df['days_since_contact'] = df['pdays'].apply(lambda x: x if x != -1 else 0)
```

### 2.2 Encoding kategorischer Variablen

**Schritt 1: One-Hot-Encoding für nominale Features**

```python
# Features mit geringer Kardinalität: One-Hot-Encoding
low_cardinality = ['default', 'housing', 'loan', 'contact']

# Vorsicht: Bei hoher Kardinalität (z.B. job, education mit vielen Kategorien)
# → Frequency Encoding oder Target Encoding erwägen

df_encoded = df.copy()

# One-Hot-Encoding mit drop='first' zur Vermeidung von Multikollinearität
df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality, drop='first', dtype=int)

print(f"Neue Spalten nach One-Hot-Encoding: {df_encoded.columns.tolist()}")
```

**Schritt 2: Ordinales Encoding für ordinale Features**

```python
# education hat natürliche Ordnung: primary < secondary < tertiary
education_mapping = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 0}
df_encoded['education_ord'] = df_encoded['education'].map(education_mapping)

# marital und contact können ordinale Bedeutung haben
# ODER frequency encoding für job (viele Kategorien)

# Für job: Frequency Encoding
job_freq = df['job'].value_counts(normalize=True).to_dict()
df_encoded['job_freq'] = df['job'].map(job_freq)

# Alternative: Durchschnittliche Abschlussrate pro Job (Target Encoding)
job_target = df.groupby('job')['deposit'].apply(lambda x: (x == 'yes').sum() / len(x)).to_dict()
df_encoded['job_target'] = df['job'].map(job_target)
```

**Schritt 3: Behandlung von Monaten**

```python
# Zyklische Kodierung für Monate (da 12 → 1 ist zyklisch)
# Option 1: Cyclical encoding mit sin/cos
import math
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df_encoded['month_num'] = df['month'].map(month_map)
df_encoded['month_sin'] = np.sin(2 * np.pi * df_encoded['month_num'] / 12)
df_encoded['month_cos'] = np.cos(2 * np.pi * df_encoded['month_num'] / 12)

# Fallback: One-Hot-Encoding
df_encoded = pd.get_dummies(df_encoded, columns=['month'], prefix='month', drop='first', dtype=int)
```

### 2.3 Feature Engineering - neue Features erstellen

```python
# Finanzielle Merkmale
df_encoded['has_credit'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)
df_encoded['credit_load'] = (df['housing'] == 'yes').astype(int) + (df['loan'] == 'yes').astype(int)

# Alter-basierte Features (Generationen)
df_encoded['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 60, 100], 
                                  labels=['18-25', '26-35', '36-50', '51-60', '60+'], ordered=True)
age_group_mapping = {'18-25': 1, '26-35': 2, '36-50': 3, '51-60': 4, '60+': 5}
df_encoded['age_group_ord'] = df_encoded['age_group'].map(age_group_mapping)

# Kontakt-Intensität Features
df_encoded['contact_intensity'] = df['campaign'] / (df['previous'] + 1)  # Kampagnen pro früherer Kontakt
df_encoded['prev_success_ratio'] = df['previous'] / df['campaign']  # Falls previous > campaign → suspicious

# Anrufdauer-Kategorien
df_encoded['duration_category'] = pd.cut(df['duration'], 
                                          bins=[0, 180, 300, 600, 4000],
                                          labels=['short', 'medium', 'long', 'very_long'], ordered=True)
duration_map = {'short': 1, 'medium': 2, 'long': 3, 'very_long': 4}
df_encoded['duration_cat_ord'] = df_encoded['duration_category'].map(duration_map)

# Zeitfenster-Features
df_encoded['day_of_month_category'] = pd.cut(df['day'], 
                                              bins=[0, 10, 20, 31],
                                              labels=['early', 'mid', 'late'], ordered=True)
dow_map = {'early': 1, 'mid': 2, 'late': 3}
df_encoded['day_cat_ord'] = df_encoded['day_of_month_category'].map(dow_map)
```

### 2.4 Feature Scaling und Normalisierung

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Welche numerischen Features skalieren?
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# StandardScaler: Für normale Algorithmen (Mittelwert=0, Std=1)
scaler_standard = StandardScaler()
df_scaled_standard = df_encoded.copy()
df_scaled_standard[numerical_features] = scaler_standard.fit_transform(df[numerical_features])

# RobustScaler: Wegen Ausreißern in balance und campaign
scaler_robust = RobustScaler()
df_scaled_robust = df_encoded.copy()
df_scaled_robust[numerical_features] = scaler_robust.fit_transform(df[numerical_features])

# MinMaxScaler: Wenn Features in [0,1] benötigt (z.B. Neuronale Netze)
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df_encoded.copy()
df_scaled_minmax[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])

# Empfehlung für dieses Projekt: RobustScaler (wegen balance-Ausreißern)
df_prepared = df_scaled_robust.copy()
```

### 2.5 Zielvariable vorbereiten

```python
# Zielvariable in binär konvertieren
y = (df['deposit'] == 'yes').astype(int)  # 1 = yes, 0 = no

# Klassen-Imbalance prüfen
print(f"Klassen-Verteilung:\n{y.value_counts()}")
print(f"Prozentanteile:\n{y.value_counts(normalize=True)}")

# Falls imbalanciert (oft der Fall bei Marketing-Kampagnen):
# Dokumentieren Sie für später: Bei Evaluation Stratified Train-Test Split verwenden!
imbalance_ratio = y.value_counts(normalize=True)
print(f"\nKlassen-Imbalance: {imbalance_ratio[0]:.1%} negative, {imbalance_ratio[1]:.1%} positive")
```

### 2.6 Train-Test-Split

```python
from sklearn.model_selection import train_test_split

# Stratified Split (beachtet Klassen-Verteilung)
X = df_prepared.drop(columns=['deposit', 'deposit_binary', 'job', 'marital', 'education', 
                                'month', 'age_group', 'duration_category', 'day_of_month_category'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train-Set: {X_train.shape[0]} Samples")
print(f"Test-Set: {X_test.shape[0]} Samples")
print(f"Feature-Dimension: {X_train.shape[1]}")
```

---

## Teil 3: Unsupervised und Supervised Learning (40%)

Dieses ist die Kernaufgabe! Sie müssen mindestens ein unüberwachtes und ein überwachtes Verfahren anwenden.

### 3.1 Unüberwachtes Lernen: Clustering (K-Means)

**Motivation:** Kundensegmentierung - Finden Sie homogene Kundengruppen ohne Labels.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Nur numerische Features für Clustering (ohne Zielvariable)
X_clustering = X_train[['age', 'balance', 'duration', 'campaign', 'previous']].copy()

# Standardisieren (für K-Means wichtig!)
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Elbow-Methode: Optimale k-Zahl finden
inertias = []
silhouette_scores = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_clustering_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_clustering_scaled, kmeans.labels_))

# Visualisierung
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Anzahl Cluster k')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow-Methode')

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Anzahl Cluster k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette-Koeffizient')

plt.tight_layout()
plt.savefig('05_kmeans_elbow.png', dpi=300)
plt.show()

# Beste k-Zahl (z.B. k=3 oder k=4 basierend auf Elbow)
optimal_k = 4  # oder basierend auf Ihrer Analyse

# Finales K-Means-Modell
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans_final.fit_predict(X_clustering_scaled)

# Cluster-Zugehörigkeiten zum Datensatz hinzufügen
X_train_with_clusters = X_train.copy()
X_train_with_clusters['cluster'] = clusters
```

**Interpretation der Cluster:**

```python
# Charakterisierung der Cluster
for cluster_id in range(optimal_k):
    cluster_data = X_train[X_train_with_clusters['cluster'] == cluster_id]
    print(f"\n=== Cluster {cluster_id} ===")
    print(f"Größe: {len(cluster_data)} Kunden ({len(cluster_data)/len(X_train)*100:.1f}%)")
    print(f"Durchschnittsalter: {cluster_data['age'].mean():.1f} Jahre")
    print(f"Durchschnittlicher Balance: €{cluster_data['balance'].mean():.0f}")
    print(f"Durchschnittliche Anrufdauer: {cluster_data['duration'].mean():.0f} Sekunden")
    print(f"Abschlussrate in diesem Cluster: {(y_train[X_train_with_clusters['cluster'] == cluster_id] == 1).sum() / len(cluster_data) * 100:.1f}%")
```

**Dokumentieren Sie:**
- Welche k-Zahl Sie gewählt haben und warum (Elbow oder Silhouette)
- Charakterisierung jedes Clusters
- Geschäftliche Interpretation (z.B. "Cluster 1: Junge, finanzschwache Kunden")

### 3.2 Feature Engineering: Dimensionsreduktion mit PCA

```python
from sklearn.decomposition import PCA

# Optional: Alternative zu oder zusätzlich zu K-Means
pca = PCA(n_components=0.95)  # 95% Varianz erhalten
X_pca = pca.fit_transform(X_clustering_scaled)

print(f"Ursprüngliche Dimensionen: {X_clustering_scaled.shape[1]}")
print(f"Reduzierte Dimensionen: {X_pca.shape[1]}")
print(f"Erklärte Varianz pro Komponente:\n{pca.explained_variance_ratio_}")
print(f"Kumulative Varianz:\n{np.cumsum(pca.explained_variance_ratio_)}")

# Visualisierung der Varianz
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Anzahl Komponenten')
plt.ylabel('Kumulative erklärte Varianz')
plt.title('PCA: Erklärte Varianz')
plt.grid(True)
plt.tight_layout()
plt.savefig('06_pca_variance.png', dpi=300)
plt.show()
```

### 3.3 Überwachtes Lernen: Klassifikation

Sie müssen mindestens ein überwachtes Verfahren anwenden. Zwei gute Optionen:

#### Option 1: Logistische Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Modell trainieren
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Vorhersagen
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# Evaluation
print("=== Logistische Regression ===")
print(f"\nKlassifikationsbericht:\n{classification_report(y_test, y_pred)}")
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Verwirrungsmatrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nVerwirrungsmatrix:\n{cm}")
```

#### Option 2: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Modell trainieren
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Vorhersagen
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation
print("=== Random Forest ===")
print(f"\nKlassifikationsbericht:\n{classification_report(y_test, y_pred_rf)}")
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Feature Importance:\n{feature_importance.head(10)}")

# Visualisierung
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Features - Random Forest')
plt.tight_layout()
plt.savefig('07_feature_importance.png', dpi=300)
plt.show()
```

#### Option 3: Gradient Boosting (XGBoost oder LightGBM)

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]

print("=== Gradient Boosting ===")
print(f"\nKlassifikationsbericht:\n{classification_report(y_test, y_pred_gb)}")
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")
```

### 3.4 Hyperparameter-Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Beispiel für Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV mit Stratified K-Fold Cross-Validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-Fold Cross-Validation
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Beste Parameter: {grid_search.best_params_}")
print(f"Beste Cross-Val ROC-AUC: {grid_search.best_score_:.4f}")

# Trainieren mit besten Parametern
rf_tuned = grid_search.best_estimator_
y_pred_tuned = rf_tuned.predict(X_test)
y_pred_proba_tuned = rf_tuned.predict_proba(X_test)[:, 1]
```

### 3.5 Modellvergleich

```python
# Erstellen Sie eine Vergleichstabelle
results = pd.DataFrame({
    'Modell': ['Logistische Regression', 'Random Forest', 'Gradient Boosting', 'RF (Tuned)'],
    'Accuracy': [
        (y_pred == y_test).sum() / len(y_test),
        (y_pred_rf == y_test).sum() / len(y_test),
        (y_pred_gb == y_test).sum() / len(y_test),
        (y_pred_tuned == y_test).sum() / len(y_test)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_gb),
        roc_auc_score(y_test, y_pred_proba_tuned)
    ]
})

print("\n=== Modellvergleich ===")
print(results)
```

---

## Teil 4: Evaluation und Reflexion (20%)

### 4.1 Performance-Metriken

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
)

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Umfassende Modell-Evaluation"""
    
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print('='*50)
    
    # Klassifikations-Metriken
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Accuracy:  {acc:.4f}  (Anteil korrekter Vorhersagen)")
    print(f"Precision: {prec:.4f}  (Von positiven Vorhersagen, wie viele sind korrekt)")
    print(f"Recall:    {rec:.4f}   (Von echten positiven, wie viele erkannt)")
    print(f"F1-Score:  {f1:.4f}   (Harmonisches Mittel von Precision/Recall)")
    print(f"ROC-AUC:   {auc:.4f}   (Tradeoff zwischen TPR und FPR)")
    
    # Verwirrungsmatrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nVerwirrungsmatrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # Spezifität
    specificity = tn / (tn + fp)
    print(f"\nSpezifität: {specificity:.4f}  (Von echten negativen, wie viele erkannt)")
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': auc}

# Evaluation auf Test-Set
results = evaluate_model(y_test, y_pred_tuned, y_pred_proba_tuned, 'Final RF Model')
```

**Metriken-Interpretation für Marketing-Kontext:**

- **Precision:** Wie viele als "Abschluss" vorhergesagte Kunden kaufen wirklich ab?
  - Hohe Precision = wenig Zeit verschwendet auf falsche Targets
  
- **Recall:** Von den echten Abschluss-Kunden, wie viele erkennen wir?
  - Hoher Recall = keine Umsätze verlassen
  
- **F1-Score:** Balance zwischen Precision und Recall
  - Bei Imbalance (viele Negative) wichtiger als Accuracy

### 4.2 ROC und PR-Kurven

```python
# ROC-Kurve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_tuned)
plt.plot(fpr, tpr, 'b-', label=f'RF (AUC={roc_auc_score(y_test, y_pred_proba_tuned):.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Zufallsvorhersage')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-Kurve')
plt.legend()
plt.grid(True, alpha=0.3)

# Precision-Recall-Kurve
plt.subplot(1, 2, 2)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_tuned)
plt.plot(recall, precision, 'g-', label=f'Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall-Kurve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_roc_pr_curves.png', dpi=300)
plt.show()
```

### 4.3 Fehleranalyse

```python
# Welche Samples werden falsch klassifiziert?
false_positives = X_test[(y_pred_tuned == 1) & (y_test == 0)]
false_negatives = X_test[(y_pred_tuned == 0) & (y_test == 1)]

print(f"\nFalse Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")

# Charakterisierung
if len(false_negatives) > 0:
    print("\nCharakteristiken von False Negatives (echte Abschlüsse, die wir übersehen):")
    print(f"Durchschn. Dauer: {false_negatives['duration'].mean():.0f}s vs {X_test['duration'].mean():.0f}s")
    print(f"Durchschn. Alter: {false_negatives['age'].mean():.1f}j vs {X_test['age'].mean():.1f}j")
```

### 4.4 Business-Impact-Analyse

```python
# Wenn Sie False Negatives haben = verlorene Verkäufe
# Wenn Sie False Positives haben = verschwendetes Marketing-Budget

# Beispiel-Szenario: €2 Kosten pro Kontakt, €500 Revenue pro Abschluss
cost_per_contact = 2
revenue_per_success = 500
total_contacts = len(y_test)

# Mit dem Modell
tp_count = (y_pred_tuned == 1).sum() & (y_test == 1)
fp_count = (y_pred_tuned == 1).sum() & (y_test == 0)

revenue = tp_count * revenue_per_success
cost = total_contacts * cost_per_contact
profit = revenue - cost

print(f"\nBusiness Impact Analysis:")
print(f"Geschätzte Kontakte mit Modell: {total_contacts}")
print(f"Vorhergesagte positive: {(y_pred_tuned == 1).sum()}")
print(f"Geschätzter Revenue: €{revenue:,.0f}")
print(f"Marketing Cost: €{cost:,.0f}")
print(f"Geschätzter Profit: €{profit:,.0f}")
```

### 4.5 Schreiben Sie eine Reflexion

**Folgende Punkte sollten Sie dokumentieren:**

1. **Was funktionierte gut?**
   - Welche Features waren wichtig?
   - Welcher Algorithmus performte am besten?
   - Welche Muster wurden entdeckt?

2. **Welche Probleme traten auf?**
   - Klasseneimbalance?
   - Overfitting (Trainings- vs. Test-Performance)?
   - Kleine Feature-Importances (Rauschen vs. Signal)?

3. **Limitationen:**
   - Fehlende externe Features (z.B. Makroökonomik)?
   - Datensatz-Bias (Zeitraum der Daten)?
   - Model Interpretierbarkeit vs. Performance Trade-off

4. **Verbesserungsvorschläge:**
   - Ensemble-Methoden kombinieren?
   - Hyperparameter-Tuning erweitern?
   - Feature Engineering weiterentwickeln?
   - Kosten-Sensitivität einbeziehen?

---

## Implementierungs-Checkliste

- [ ] 1. Daten einlesen und grundlegend inspizieren
- [ ] 2. Fehlende Werte und Duplikate prüfen
- [ ] 3. Deskriptive Statistiken berechnen
- [ ] 4. Visualisierungen erstellen (Histogramme, Boxplots, Heatmaps)
- [ ] 5. Bivariate Analysen und Korrelationen
- [ ] 6. Kategorische Variablen encodieren
- [ ] 7. Neue Features erstellen
- [ ] 8. Features skalieren
- [ ] 9. Train-Test-Split mit Stratifikation
- [ ] 10. K-Means Clustering mit Elbow-Methode
- [ ] 11. Mindestens 2 Klassifikationsalgorithmen implementieren
- [ ] 12. Hyperparameter-Tuning durchführen
- [ ] 13. Alle Evaluationsmetriken berechnen
- [ ] 14. ROC und PR-Kurven visualisieren
- [ ] 15. Reflexion und Verbesserungsvorschläge schreiben

---

## Best-Practices für die Implementierung

1. **Code-Qualität:**
   - Kommentare für jeden Abschnitt
   - Aussagekräftige Variablennamen
   - Funktionen für wiederholte Code-Blöcke

2. **Dokumentation:**
   - Begründen Sie Ihre Entscheidungen
   - Zeigen Sie Ihre Denkweise
   - Interpretieren Sie numerische Ergebnisse in Worten

3. **Visualisierungen:**
   - Speichern Sie alle Plots als PNG/PDF
   - Verwenden Sie aussagekräftige Titel und Labels
   - Beschreiben Sie Was Sie sehen

4. **Reproduzierbarkeit:**
   - Setzen Sie random_state=42 überall
   - Dokumentieren Sie Bibliotheks-Versionen
   - Speichern Sie trainierte Modelle (optional)

---

## Literaturverweise (aus den Vorlesungen)

- Datenexploration: 03_EXP.pdf
- Pandas: 02_PANDAS.pdf  
- Feature Engineering: 06_FEAT.pdf
- Clustering: 05_CLUST.pdf
- Klassifikation: 07_KLASS.pdf
- Regression: 08_REGR.pdf

Viel Erfolg bei Ihrem Projekt!