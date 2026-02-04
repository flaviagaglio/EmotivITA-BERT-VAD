# EmotivITA-BERT-VAD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-ffd700.svg)](https://huggingface.co/transformers/)

**Regressione delle emozioni in testo italiano** tramite modelli **BERT** sul paradigma **VAD** (Valence, Arousal, Dominance), basato sul dataset **EmoITA**.

---

## Indice

- [Panoramica](#-panoramica)
- [Struttura del progetto](#-struttura-del-progetto)
- [Requisiti e setup](#-requisiti-e-setup)
- [Dataset EmoITA](#-dataset-emoita)
- [Task A: regressione single-target (V, A, D separati)](#-task-a-regressione-single-target-v-a-d-separati)
- [Task B: regressione multi-target (VAD congiunto)](#-task-b-regressione-multi-target-vad-congiunto)
- [Pipeline completa](#-pipeline-completa)
- [Spiegazione del codice](#-spiegazione-del-codice)
- [Configurazione](#-configurazione)
- [Risultati e valutazione](#-risultati-e-valutazione)
- [Grafici e risultati](#-grafici-e-risultati)
- [Riferimenti](#-riferimenti)

---

## Panoramica

Il progetto implementa **due approcci** per la predizione delle dimensioni emotive **VAD** su frasi in italiano:

| Approccio | Descrizione | Output |
|-----------|-------------|--------|
| **Task A** | Tre modelli BERT distinti: uno per **Valence**, uno per **Arousal**, uno per **Dominance**. Regressione single-target. | 3 modelli (`.pth`), metriche per dimensione |
| **Task B** | Un unico modello BERT che predice **V, A e D** insieme. Regressione multi-target. | 1 modello (`.pth`), metriche congiunte |

- **Valence (V)**: piacevolezza (1 = molto negativa, 5 = molto positiva)  
- **Arousal (A)**: intensità attivazione (1 = calmo, 5 = eccitato)  
- **Dominance (D)**: controllo (1 = dominato, 5 = dominante)

Entrambi i task usano **BERT italiano** pre-addestrato (`dbmdz/bert-base-italian-cased`), con **fine-tuning** parziale (ultimi 2 layer + regressore) e **early stopping** in validazione. Le metriche ufficiali sono **MAE** (Mean Absolute Error) e **Pearson r** per ogni dimensione.

---

## Struttura del progetto

```
REV EmotivITA-BERT-VAD/
├── requirements.txt
│
├── taskA/                          # Task A: tre modelli (V, A, D separati)
│   ├── config_valence.yaml         # Config training/val Valence
│   ├── config_arousal.yaml         # Config training/val Arousal
│   ├── config_dominance.yaml       # Config training/val Dominance
│   ├── config_valence_test.yaml    # Config test ufficiale Valence
│   ├── config_arousal_test.yaml
│   ├── config_dominance_test.yaml
│   ├── data/
│   │   ├── Development set.csv     # Dataset originale (train+val)
│   │   ├── Test set - Gold labels.csv
│   │   ├── train_valence.csv, val_valence.csv, test_valence.csv
│   │   ├── train_arousal.csv, val_arousal.csv, test_arousal.csv
│   │   └── train_dominance.csv, val_dominance.csv, test_dominance.csv
│   ├── code/
│   │   ├── model.py                # BertForSingleRegression
│   │   ├── train.py                # Training con early stopping
│   │   ├── predict.py              # Predizioni su test/val
│   │   ├── evaluation.py          # MAE e Pearson r
│   │   ├── prepare_taskA_datasets.py  # Split train/val/test per V,A,D
│   │   ├── check_datasets.py       # Controllo struttura CSV
│   │   └── comparativa_taskA.py    # Tabella + grafico V vs A vs D
│   └── risultati/
│       ├── model_valence.pth, model_arousal.pth, model_dominance.pth
│       ├── model_*_pred.csv, model_*_results.txt
│       └── comparativa_taskA.png
│
├── taskB/                          # Task B: un modello VAD congiunto
│   ├── config.yaml                 # Unica config (train, val, test, gold)
│   ├── data/
│   │   ├── Development set.csv
│   │   ├── Test set.csv            # Test senza etichette
│   │   ├── Test set - Gold labels.csv
│   │   ├── train.csv, val.csv      # Generati da prepare_dataset.py
│   ├── code/
│   │   ├── _paths.py               # Path assoluti rispetto a taskB/
│   │   ├── model.py                # BertForRegression (output dim=3)
│   │   ├── train.py                # Training + salvataggio loss
│   │   ├── predict.py              # Predizioni V,A,D su test
│   │   ├── evaluation.py           # MAE e Pearson per V, A, D
│   │   ├── prepare_dataset.py      # Split 80/20 train/val
│   │   ├── plot_loss.py            # Grafico train/val loss
│   │   ├── tabella_taskB.py        # Stampa tabella risultati
│   │   └── comparativa_AB.py       # Confronto Task A vs Task B
│   └── risultati/
│       ├── model.pth
│       ├── my_predictions.csv, risultati.txt
│       ├── train_losses.txt, val_losses.txt
│       ├── loss_curve.png          # Grafico train/val loss (plot_loss.py)
│       └── comparativa_AB.png
│
└── presentazione.pdf
```

---

## Requisiti e setup

### Requisiti di sistema

- **Python** 3.8 o superiore (consigliato 3.10+)
- **CUDA** opzionale ma consigliata per training più veloce

### 1. Clonare / scaricare il progetto

Assicurarsi di avere nella root i file CSV originali EmoITA:

- `taskA/data/Development set.csv`
- `taskA/data/Test set - Gold labels.csv`
- `taskB/data/Development set.csv`
- `taskB/data/Test set.csv`
- `taskB/data/Test set - Gold labels.csv`

### 2. Ambiente virtuale e dipendenze

Dalla **root del progetto**:

```bash
# Creare e attivare un ambiente virtuale
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# oppure: venv\Scripts\activate   # Windows

# Installare le dipendenze
pip install -r requirements.txt
```

### 3. Contenuto di `requirements.txt`

| Pacchetto      | Uso                          |
|----------------|------------------------------|
| `torch`        | Modelli e training           |
| `transformers` | BERT e tokenizer              |
| `pandas`       | Lettura CSV e preparazione    |
| `numpy`        | Calcoli numerici             |
| `scikit-learn` | Split, metriche (MAE)         |
| `scipy`        | Pearson r                     |
| `PyYAML`       | Config YAML                  |
| `matplotlib`   | Grafici comparativi e loss    |

Dopo il setup, tutti i comandi vanno eseguiti con l’ambiente attivato.

---

## Dataset EmoITA

I dati EmoITA forniscono frasi in italiano annotate con **V, A, D** in scala 1–5.

- **Development set**: usato per **train** e **validation** (split 80/20).
- **Test set** (con gold labels): usato **solo** per la valutazione finale (nessun training).

Formato atteso dei CSV:

- Colonne obbligatorie: `text`, e per i dati annotati anche `V`, `A`, `D`.
- Opzionale: colonna `id` per allineare predizioni e gold (Task B).

---

## Task A: regressione single-target (V, A, D separati)

Un modello BERT dedicato per ogni dimensione (Valence, Arousal, Dominance).

### 1. Preparazione dati

Dalla **root del progetto**:

```bash
python taskA/code/prepare_taskA_datasets.py
```

- Legge `taskA/data/Development set.csv` e `Test set - Gold labels.csv`.
- Crea train/val (80/20) e file test per **V**, **A** e **D** in `taskA/data/`:
  - `train_valence.csv`, `val_valence.csv`, `test_valence.csv`
  - `train_arousal.csv`, `val_arousal.csv`, `test_arousal.csv`
  - `train_dominance.csv`, `val_dominance.csv`, `test_dominance.csv`

### 2. (Opzionale) Controllo dataset

```bash
python taskA/code/check_datasets.py
```

Mostra colonne, numero di righe e prime righe di ogni CSV in `taskA/data/`.

### 3. Training

Si usa un config per dimensione. La **validation** è su `val_*`; il **test ufficiale** si fa con i config `*_test.yaml`.

```bash
cd taskA/code

# Training (validation per early stopping)
python train.py --config ../config_valence.yaml
python train.py --config ../config_arousal.yaml
python train.py --config ../config_dominance.yaml
```

I modelli vengono salvati in `taskA/risultati/` (es. `model_valence.pth`).

### 4. Predizioni

Sul set di **validation** (per coerenza con il training):

```bash
python predict.py --config ../config_valence.yaml
python predict.py --config ../config_arousal.yaml
python predict.py --config ../config_dominance.yaml
```

Per il **test ufficiale** (con gold labels):

```bash
python predict.py --config ../config_valence_test.yaml
python predict.py --config ../config_arousal_test.yaml
python predict.py --config ../config_dominance_test.yaml
```

Le predizioni sono scritte in `taskA/risultati/model_*_pred.csv`.

### 5. Valutazione

```bash
python evaluation.py --config ../config_valence.yaml    # val
python evaluation.py --config ../config_valence_test.yaml  # test ufficiale
# Idem per arousal e dominance.
```

Genera i file `model_*_results.txt` con **MAE** e **Pearson r** per ogni dimensione.

### 6. Comparativa Task A

```bash
python comparativa_taskA.py
```

- Stampa una tabella con MAE e Pearson r per Valence, Arousal, Dominance.
- Salva il grafico in `taskA/risultati/comparativa_taskA.png`.

---

## Task B: regressione multi-target (VAD congiunto)

Un unico modello BERT che predice **V, A e D** insieme.

### 1. Preparazione dati

Dalla **root del progetto**:

```bash
python taskB/code/prepare_dataset.py
```

- Legge `taskB/data/Development set.csv`.
- Crea `taskB/data/train.csv` e `taskB/data/val.csv` (80/20).

### 2. Training

Si può eseguire da qualsiasi directory; gli script usano path relativi a `taskB/` tramite `_paths.py`.

```bash
python taskB/code/train.py
```

- Modello salvato in `taskB/risultati/model.pth`.
- Loss di train e validation salvate in `train_losses.txt` e `val_losses.txt`.

### 3. Predizioni sul test

```bash
python taskB/code/predict.py
```

- Legge `taskB/data/Test set.csv` e salva le predizioni in `taskB/risultati/my_predictions.csv` (colonne `V_pred`, `A_pred`, `D_pred`).

### 4. Valutazione

```bash
python taskB/code/evaluation.py
```

- Confronta `my_predictions.csv` con `Test set - Gold labels.csv` (merge su `id` o `text`).
- Scrive **MAE** e **Pearson r** per V, A, D in `taskB/risultati/risultati.txt`.

### 5. Visualizzazione loss e risultati

```bash
# Grafico train/val loss
python taskB/code/plot_loss.py

# Tabella risultati in console
python taskB/code/tabella_taskB.py
```

### 6. Comparativa Task A vs Task B

```bash
python taskB/code/comparativa_AB.py
```

- Legge i risultati di Task A (`taskA/risultati/model_*_results.txt`) e Task B (`taskB/risultati/risultati.txt`).
- Stampa una tabella e salva `taskB/risultati/comparativa_AB.png`.

---

## Pipeline completa

Sequenza consigliata per riprodurre tutto dall’inizio:

```bash
# 1. Ambiente
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Dati
python taskA/code/prepare_taskA_datasets.py
python taskB/code/prepare_dataset.py

# 3. Task A: train → predict → evaluation (per V, A, D)
cd taskA/code
for dim in valence arousal dominance; do
  python train.py --config ../config_${dim}.yaml
  python predict.py --config ../config_${dim}.yaml
  python evaluation.py --config ../config_${dim}.yaml
done
python comparativa_taskA.py
cd ../..

# 4. Task B: train → predict → evaluation
python taskB/code/train.py
python taskB/code/predict.py
python taskB/code/evaluation.py
python taskB/code/plot_loss.py
python taskB/code/comparativa_AB.py
```

---

## Spiegazione del codice

### Task A

#### `model.py` — BertForSingleRegression

- **Base**: `AutoModel.from_pretrained("dbmdz/bert-base-italian-cased")`.
- **Layer congelati**: i primi 10 layer di BERT (`layer.0` … `layer.9`) hanno `requires_grad=False` per ridurre overfitting e tempo di training; si aggiornano solo gli ultimi 2 layer e il regressore.
- **Dropout**: 0.3 sull’output del token `[CLS]`.
- **Regressore**: livello lineare `hidden_size (768) → 1`; in uscita si predice una sola dimensione (V, A o D).
- **Forward**: `input_ids` e `attention_mask` → BERT → `last_hidden_state[:, 0, :]` (CLS) → dropout → regressore → `squeeze(-1)` per avere shape `(batch,)`.

#### `train.py` — Dataset e training

- **EmoITA_Dataset**: legge un CSV con colonne `text` e (opzionale) una colonna target (es. `V`). Tokenizza con il tokenizer BERT, `max_length` e padding; le etichette sono **normalizzate** in [0, 1] con `target / 5.0`.
- **train_epoch / eval_epoch**: un’epoca di training con backprop e una di sola validazione; loss **SmoothL1Loss**, ottimizzatore **Adam** (lr da config).
- **Early stopping**: se la validation loss non migliora per **3 epoche** consecutive, il training si interrompe; viene salvato sempre il modello con validation loss minima (`model_save_path` nel config).

#### `predict.py`

- Usa lo stesso tokenizer e `max_len` del training; il dataset ha solo `text` (nessuna label).
- Carica il modello da `model_save_path`, esegue la forward e **denormalizza** le predizioni moltiplicando per 5.
- Scrive un CSV con le colonne originali del test più una colonna `{target_col}_pred`.

#### `evaluation.py`

- Legge il CSV delle predizioni e il CSV con le gold; allinea per indice (stesso numero di righe).
- Calcola **MAE** (sklearn) e **Pearson r** (scipy) tra gold e predizioni e scrive i risultati in `model_*_results.txt`.

#### `prepare_taskA_datasets.py`

- Carica il Development set, rimuove righe con `text`, V, A o D mancanti.
- Split 80% train, 20% validation (seed fisso).
- Per ogni dimensione (valence, arousal, dominance) salva `train_*.csv`, `val_*.csv`.
- Dal “Test set - Gold labels” crea i file `test_*.csv` per la valutazione finale.

---

### Task B

#### `_paths.py`

- Calcola la directory `taskB/` come parent di `code/`.
- La funzione `path(*parts)` restituisce il path assoluto di un file sotto `taskB/`, così gli script funzionano indipendentemente dalla directory di esecuzione.

#### `model.py` — BertForRegression

- Stessa base BERT italiano; congelamento dei primi 10 layer (compatibile con naming `encoder.layer.*` e `layer.*`).
- **Regressore**: `hidden_size → 3` (V, A, D). Output shape `(batch, 3)`.
- Forward: CLS → dropout → regressore; nessuno squeeze.

#### `train.py`

- **EmoITA_Dataset**: colonne `text`, `V`, `A`, `D`; etichette normalizzate con `V/5`, `A/5`, `D/5`.
- Stessa logica di train/val e early stopping del Task A; in più **scrive** train e validation loss per ogni epoca in `train_losses.txt` e `val_losses.txt`.

#### `predict.py`

- Carica `config.yaml` e `risultati/model.pth`, legge il test set (solo `text`), produce predizioni (batch, 3), denormalizza moltiplicando per 5.
- Salva un CSV con `text` (e eventuali colonne originali) più `V_pred`, `A_pred`, `D_pred`.

#### `evaluation.py`

- Merge tra file di predizioni e gold su colonna `id` o `text`; se non possibile, allineamento per posizione (con controllo lunghezze).
- Per V, A, D: MAE e Pearson r (con `safe_pearsonr` per vettori costanti); risultati in `risultati.txt`.

#### `plot_loss.py`

- Legge `train_losses.txt` e `val_losses.txt` (formato `epoch,loss`) e disegna un grafico epoca vs loss (train e val).

#### `comparativa_AB.py`

- Parsa i file `model_*_results.txt` di Task A e `risultati.txt` di Task B.
- Costruisce una tabella e un grafico (MAE e Pearson r) per confrontare Task A e Task B sulle tre dimensioni.

---

## Configurazione

### Task A — Esempio `config_valence.yaml`

```yaml
dataset: taskA/data/train_valence.csv    # training
testset: taskA/data/val_valence.csv      # validation (o test_valence.csv per test ufficiale)
target_col: V                            # V, A o D
model_save_path: taskA/risultati/model_valence.pth

tokenizer:
  name: dbmdz/bert-base-italian-cased
  # max_len opzionale; default 128 in train/predict

training:
  epochs: 8
  batch_size: 16
  learning_rate: 0.00002
  seed: 42
```

Per il **test ufficiale** si usa ad es. `config_valence_test.yaml` con `testset: taskA/data/test_valence.csv`.

### Task B — `config.yaml`

```yaml
dataset:
  train: "data/train.csv"
  val: "data/val.csv"
  test: "data/Test set.csv"
  gold: "data/Test set - Gold labels.csv"

predictions_path: "risultati/my_predictions.csv"
results_path: "risultati/risultati.txt"

tokenizer:
  name: "dbmdz/bert-base-italian-cased"
  max_len: 64

params:
  batch_size: 16
  lr: 0.00003
  num_epochs: 8
```

I path sotto `dataset` e per `predictions_path` / `results_path` sono relativi a `taskB/` quando si usa `_paths.path()`.

---

## Risultati e valutazione

- **Metriche**: per ogni dimensione (V, A, D) vengono riportati **MAE** (in scala 1–5) e **Pearson r** (correlazione con le gold).
- **Task A**: tre file `model_*_results.txt` e grafico `comparativa_taskA.png`.
- **Task B**: un file `risultati.txt` e grafico `comparativa_AB.png` (confronto con Task A).
- I file `.pth` non sono presenti; dopo il training si trovano in `taskA/risultati/` e `taskB/risultati/`.

---

## Risultati ottenuti

Di seguito i risultati ottenuti sui set di valutazione/test dopo training ed evaluation (valori riproducibili con la [pipeline completa](#-pipeline-completa)).

### Task A — Tre modelli (single-target)

| Dimensione | MAE | Pearson r |
|------------|-----|-----------|
| Valence (V) | 0.364 | 0.632 |
| Arousal (A) | 0.310 | 0.557 |
| Dominance (D) | 0.287 | 0.541 |

### Task B — Un modello (multi-target)

| Dimensione | MAE | Pearson r |
|------------|-----|-----------|
| Valence (V) | 0.323 | 0.711 |
| Arousal (A) | 0.301 | 0.589 |
| Dominance (D) | 0.281 | 0.598 |

### Discussione

- **Cosa abbiamo ottenuto**: correlazioni **Pearson r** nella fascia **0.54–0.71**; **MAE** tra **0.28 e 0.36** (su scala 1–5). Entrambi gli approcci raggiungono correlazioni moderate–buone con le gold labels e errori assoluti medi contenuti.
- **Pattern**: Valence tende a MAE leggermente più alto nel Task A; nel Task B si osservano correlazioni più alte su Valence. Dominance ha MAE più basso in entrambi i task.
- **Trade-off**: Task A usa tre modelli specializzati (tre training, file separati per dimensione); Task B usa un solo modello multi-output (un training, un unico file di risultati). I risultati dipendono da dataset, iperparametri e split; non si dichiara un “modello migliore” — i due approcci sono entrambi validi e riproducibili.

I grafici comparativi (`comparativa_taskA.png`, `comparativa_AB.png`) visualizzano questi risultati; i valori numerici sono in `taskA/risultati/model_*_results.txt` e `taskB/risultati/risultati.txt`.

---

## Grafici e risultati

Dopo aver eseguito training, evaluation e gli script comparativi, il progetto genera i seguenti grafici. Qui sotto sono mostrati come riferimento.

### Task A — Comparativa Valence, Arousal, Dominance

MAE e Pearson r per le tre dimensioni (un modello per dimensione). Generato con `python taskA/code/comparativa_taskA.py`.

<img width="3000" height="1200" alt="comparativa_taskA" src="https://github.com/user-attachments/assets/df4a98a5-ae75-4396-9ebd-202eec1cdcf3" />



### Task A vs Task B — Confronto metriche

Confronto diretto tra l’approccio a tre modelli (Task A) e l’approccio a un solo modello (Task B). Generato con `python taskB/code/comparativa_AB.py`.

<img width="3600" height="1500" alt="comparativa_AB" src="https://github.com/user-attachments/assets/fa455cbb-f799-4896-907a-a501791559ae" />




---

## Riferimenti

- **BERT italiano**: [dbmdz/bert-base-italian-cased](https://huggingface.co/dbmdz/bert-base-italian-cased)
- **EmoITA**: dataset EmoITA per emozioni in italiano (VAD)
- **PyTorch**: [pytorch.org](https://pytorch.org/)
- **Hugging Face Transformers**: [huggingface.co/transformers](https://huggingface.co/transformers/)

