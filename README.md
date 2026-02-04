# Analyse verschiedener Verlustfunktionen zur Rekonstruktion von T1-T2-Spektren

Dieses Repository enthält die Implementierungen der einzelnen Verlustfunktionen sowie das Trainings- und Evaluationsskript der dazugehörigen Bachelorarbeit.

Zusätzlich sind **trainierte Modelle** und deren **Trainingshistorien** enthalten.

---

## 📂 Projektstruktur

* **`loss_functions.py`**
  Enthält alle implementierten Verlustfunktionen (z. B. MSE, KL-Divergenz, Wasserstein, Focal Loss, Tversky etc.).

* **`trainings_script.py`**
  Das Hauptskript zum Trainieren der neuronalen Netze.

* **`evaluation_script.py`**
  Skript zur Berechnung von Metriken (MSE, SSIM, Dice) und zur Erstellung von Plots/Visualisierungen.

* **`models/`**
  Alle besten trainierten Modelle der Verlustfunktionen.

---

## 🚀 Training (`trainings_script.py`)

Mit diesem Skript werden die Modelle trainiert.

### Nutzung
`python trainings_script.py [PARAMETER]`

### Parameter

* **`--loss`** (Pflichtfeld)
  Die zu trainierende Verlustfunktion.
  * *Optionen:* `mse`, `mae`, `kldiv`, `jsd`, `wbce`, `focal`, `softdice`, `tversky`, `semd`, `jsdsemd`, `maetversky`, `wbcedice`

* **`--data_dir`** (Pflichtfeld)
  Pfad zum Ordner mit den Trainingsdaten.

* **`--output_dir`** (Pflichtfeld)
  Pfad, in dem Modelle und Logs gespeichert werden.

* **`--epochs`**
  Gesamtanzahl der Trainingsepochen.
  * *Default:* `5000`

* **`--samples`**
  Anzahl der Samples pro Kompartment-Anzahl.
  * *Default:* `100000`

* **`--lr_list`**
  Liste der Lernraten, durch Komma getrennt.
  * *Default:* `0.01,0.005,0.001,0.0005,0.0001`

* **`--alpha`**
  Gewichtungsparameter Alpha (benötigt für `tversky` und `focal`).

* **`--beta`**
  Gewichtungsparameter Beta (benötigt für `tversky` und `wbce`).

* **`--gamma`**
  Fokussierungsparameter Gamma (benötigt für `focal`).

### 💡 Beispiele

**Training mit MSE:** 

`python trainings_script.py --loss mse --data_dir ./data/ --output_dir ./results/mse`


**Training mit Tversky Loss und Parametern:**

`python trainings_script.py --loss tversky --alpha 0.7 --beta 0.3 --lr_list 0.01,0.001 --data_dir ./data/ --output_dir ./results/tversky`

---

## 📊 Evaluation (`evaluation_script.py`)

Dieses Skript evaluiert trainierte Modelle, berechnet Metriken und erstellt Visualisierungen.

### Nutzung
`python evaluation_script.py [PARAMETER]`

### Parameter

* **`--model`**
  Die zu evaluierenden Modelle.
  * *Optionen:* `mse`, `mae`, `kldiv`, `jsd`, `wbce`, `focal`, `softdice`, `tversky`, `semd`, `jsdsemd`, `maetversky`, `wbcedice`
  * `all`, um alle Modelle zu evaluieren.

* **`--data_dir`**
  Pfad zu den Daten.

* **`--samples`**
  Anzahl der Testsamples, die für die Evaluation genutzt werden sollen pro Kompartment-Anzahl.

* **`--tasks`**
  Die Art der Evaluation, die ausgeführt werden soll.
  * `matrix`: Erstellt eine Cross-Loss Matrix.
  * `table`: Gibt eine Tabelle mit den Metriken MSE, Dice, SSIM, PSNR pro Kompartments-Anzahl aus.
  * `overview`: Visualisierung Ground Truth vs. Prediction (ein Sample pro Kompartment-Anzahl).
  * `indepth`: Detaillierte 2D- und 3D-Visualisierung inkl. Differenzplot.
  * `history`: Plottet den Verlauf von Trainings- und Validierungsverlust.
  * `all`: Führt alle Tasks aus.

### 💡 Beispiele

**Metrik-Tabelle für ein Modell erstellen:**

`python evaluation_script.py --model mse --tasks table --data_dir ./data/`


**Vergleich aller Modelle (History & Matrix):**

`python evaluation_script.py --model all --tasks history --data_dir ./data/`


**Detaillierte Analyse (Indepth) mit 500 Samples auf zwei Modellen:**

`python evaluation_script.py --model mse mae --tasks indepth --samples 500 --data_dir ./data/`

---

## 📦 Anforderungen

Alle benötigten Python-Bibliotheken sind in der Datei requirements.txt:

`pip install -r requirements.txt`
