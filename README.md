### pruefung_data_science-Gruppe_8
Abgabe: 01.04.2026

## Aufgabe:

# Aufgabenstellung 

Im Rahmen der Prüfung bearbeiten Sie in Zweiergruppen an einen vorgegebenen Datensatz typische Elemente eines Data-Science-Projekts.
Ihre Lösungen fassen Sie in einem Jupyter-Notebook zusammen, das Sie an Hand der unten stehende vier Teilaufgaben strukturieren. 

Das Jupyter-Notebook ...
* enthält Ihre Gruppennummer (OPAL-Einschreibung) sowie Ihre Namen inkl. Matrikelnummer
* reichen Sie via OPAL bis zur dort angegebenen Abgabefrist ein


# Teilaufgaben

## 1. Explorative Datenanalyse (20%)

Untersuchen Sie den Datensatz eingehend und beschreiben Sie, welche Informationen gespeichert sind. Bestimmen Sie die grundlegenden Eigenschaften (Attributnamen und Wertebereiche), Verteilungen und Zusammenhänge. Nutzen Sie dazu geeignete Visualisierungstechniken (z.B. Histogramme, Streudiagramme, Boxplots) sowie statistische Kennzahlen (z.B. Mittelwerte, Standardabweichungen, Korrelationen). Identifizieren Sie mögliche Muster, Trends, Ausreißer oder Anomalien. Dokumentieren Sie Ihre Beobachtungen und Schlussfolgerungen.

## 2. Data Cleaning und Feature Engineering (20%)

Bereinigen Sie die Daten: Behandeln Sie fehlende Werte, inkonsistente oder fehlerhafte Einträge. Transformieren Sie die Daten bei Bedarf (z.B. Skalieren, Kodieren kategorialer Variablen). Erstellen Sie neue, abgeleitete Features, die die Modellleistung verbessern könnten. Begründen Sie Ihre Auswahl der Data-Cleaning-Schritte und Feature-Engineering-Maßnahmen.

## 3. Unsupervised und Supervised Learning (40%)

Formulieren Sie Zielsetzungen, die mittels Lernverfahren für die Daten ermitteln werden können (z.B. Vorhersage eine Attributs). 
Wenden Sie mindestens eine unüberwachte (z.B. Clusteranalyse, Dimensionsreduktion) sowie eine überwachte Lernmethode (z.B. Klassifikation, Regression) an.
Passen Sie die Modelle an, trainieren Sie sie und dokumentieren Sie die verwendeten Algorithmen sowie deren Parameter.
Präsentieren Sie die Ergebnisse und interpretieren Sie diese im Kontext Ihrer Aufgabenstellung.

## 4. Evalutation und Reflexion (20%)

Bewerten Sie die Performance Ihrer Modelle anhand geeigneter Metriken (z.B. Genauigkeit, F1-Score, RMSE).
Diskutieren Sie die Stärken und Schwächen Ihrer Vorgehensweise.
Reflektieren Sie über die Interpretierbarkeit der Modelle und die Robustheit Ihrer Ergebnisse.
Überlegen Sie, wie Ihr Ansatz verbessert werden könnte und welche Herausforderungen bei der Analyse aufgetreten sind.

# Bewertung

Die einzelnen Teilaufgaben sind wie oben angegeben gewichtet.
Die Teilaufgaben werden nach folgenedn Kriterien bewertet:
* Vollständigkeit und Nachvollziehbarkeit: Der Lösungsansatz umfasst alle geforderten Abschnitte und Schritte. Die Vorgehensweise ist nachvollziehbar dokumentiert. 
* Korrektheit und Angemessenheit der Methoden: Die eingesetzten Methoden (z. B. Visualisierungstechniken, Data Cleaning, Machine Learning-Modelle) sind korrekt angewandt und angemessen für die gestellte Aufgabenstellung. 
* Interpretation und Reflexion: Die Ergebnisse werden sachgerecht interpretiert, kritisch reflektiert und in Bezug zur Aufgabenstellung gesetzt. Es werden plausible Schlussfolgerungen gezogen, mögliche Limitationen erkannt sowie Verbesserungspotenziale aufgezeigt.
* Qualität der Präsentation und Dokumentation: Die Arbeit ist klar strukturiert, verständlich geschrieben und verständlich visualisiert. Es werden relevante Grafiken und Tabellen genutzt, um die Ergebnisse zu unterstützen. Der Text ist frei von Rechtschreib- und Grammatikfehlern.


# Datensätze

## Gruppenzuordnung
* Datensatz Adult für Gruppen mit **ungerader** Gruppennummer
* Datensatz Bank für Gruppen mit **gerader** Gruppennummer 


## Datensatz Adult


* age: the age of the individual in years,
* capital_gain: capital gain in the previous year,
* capital_loss: capital loss in the previous year,
* education: highest level of education achieved by the individual,
* education-num: a numeric form of the highest level of education achieved,
* fnlwgt: an estimate of the number of individuals in the population with the same demographics as this individual,
* hours_per_week: hours worked per week,
* marital_status: the marital status of the individual,
* native_country: the native country of the individual,
* occupation: the occupation of the individual,
* race: the individual's race,
* relationship: the individual's relationship status,
* sex: the individual's sex,
* workclass: the industry / sector that the individual works in.
* income: binary label encodes whether individual earns more or less than $50,000.


## Datensatz Bank

bank client data:
* 1 - age (numeric)
* 2 - job : type of job 
* 3 - marital : marital status 
* 4 - education
* 5 - default: has credit in default? (binary: "yes","no")
* 6 - balance: average yearly balance, in euros (numeric) 
* 7 - housing: has housing loan? (binary: "yes","no")
* 8 - loan: has personal loan? (binary: "yes","no")

related with the last contact of the current campaign:
* 9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
* 10 - day: last contact day of the month (numeric)
* 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
* 12 - duration: last contact duration, in seconds (numeric)

other attributes:
* 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
* 15 - previous: number of contacts performed before this campaign and for this client (numeric)
* 16 - poutcome: outcome of the previous marketing campaign 
* 17 - deposit - has the client subscribed a term deposit? (binary: "yes","no") 
