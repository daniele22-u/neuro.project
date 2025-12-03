# Fine-tuning e Inferenza con Foundation Models

Questo progetto fornisce una pipeline completa per:
✅ **Fine-tuning** di foundation models (SAM, MedSAM) su tutto il dataset ASOCA + ImageCAS
✅ **Calcolo del Dice score medio** durante e dopo il fine-tuning
✅ **Inferenza pura** senza training

## 🎯 Caratteristiche Principali

### 1. Fine-tuning Completo
- Fine-tune su **tutto il dataset** (ASOCA + ImageCAS)
- Training con **Dice Loss + BCE Loss**
- **Monitoraggio Dice score** durante training
- **Validazione** ogni epoca
- **Salvataggio automatico** del modello migliore
- **Grafici** di training loss e Dice score

### 2. Inferenza Pura
- Modalità **inference-only** (no training)
- Supporto per modelli **pretrained** o **fine-tuned**
- **Calcolo Dice medio** sul test set
- Statistiche complete (mean, std, min, max)
- Salvataggio risultati per analisi successive

### 3. Dataset ASOCA + ImageCAS
- **Supporto completo** per la struttura dataset fornita
- **Split automatico** train/val/test (70/15/15)
- **Caching** per performance ottimali
- **Augmentations 2D** (flip, rotation, noise, shift)
- **Preprocessing** automatico (crop, resize, normalization)

## 📁 File Principali

```
neuro.project/
├── train_foundation_models.py    # Script principale training/inference
├── dataset_asoca_cas.py           # Dataset loader ASOCA + ImageCAS
├── example_finetuning.py          # Esempi d'uso completi
├── FINETUNING_GUIDE.md           # Guida dettagliata (italiano)
├── README_FINETUNING.md          # Questo file
└── foundation_models_pipeline.py  # Foundation models (SAM, MedSAM, etc.)
```

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
pip install -r requirements.txt
```

### 2. Configurazione Dataset

Modifica i path in `dataset_asoca_cas.py`:

```python
BASE_DIR_ASOCA = "/path/to/your/ASOCA"
BASE_DIR_CAS = "/path/to/your/ImageCAS/Data"
```

### 3. Fine-tuning

```bash
# Fine-tune MedSAM per 10 epoche
python train_foundation_models.py --mode train --model medsam --epochs 10

# Fine-tune SAM
python train_foundation_models.py --mode train --model sam --epochs 5
```

**Output:**
- Checkpoint in `checkpoints/`
- Modello migliore: `checkpoints/best_model.pth`
- **Dice score medio finale** stampato a schermo

### 4. Inferenza (senza training)

```bash
# Inferenza con MedSAM pretrained
python train_foundation_models.py --mode inference --model medsam

# Inferenza con modello fine-tuned
python train_foundation_models.py --mode inference --model medsam \
    --checkpoint checkpoints/best_model.pth
```

**Output:**
- Predizioni in `inference_results/`
- **Dice score medio** sul test set
- Statistiche complete

## 📊 Esempio Output

### Durante Fine-tuning

```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
STARTING FINE-TUNING
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
Model: MEDSAM
Epochs: 10
Samples per epoch: 100
Device: cuda

============================================================
Epoch 1/10
============================================================
Training: 100%|████████████| 100/100 [05:23<00:00, loss=0.4235, dice=0.7231]

📊 Epoch 1 Summary:
   Train Loss: 0.4235
   Train Dice: 0.7231
   Val Dice:   0.7456 (on 50 samples)
   💾 New best model saved! (Dice: 0.7456)

...

✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
FINE-TUNING COMPLETED
✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

Final Training Dice: 0.8123
Best Validation Dice: 0.7892
```

### Durante Inferenza

```
🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
RUNNING INFERENCE
🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
Model: MEDSAM
Mode: Inference only (no training)
Device: cuda

Inference: 100%|████████████| 20/20 [03:45<00:00]

📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊
INFERENCE RESULTS
📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊
Number of samples: 20
Average Dice Score: 0.7845
Std Dice Score: 0.0856
Min Dice Score: 0.6234
Max Dice Score: 0.9012

💾 Results saved to: inference_results/inference_medsam.pth
```

## 💻 Uso Programmatico

### Esempio Completo: Fine-tuning + Validazione

```python
from dataset_asoca_cas import DatasetMerged_2d
from train_foundation_models import FoundationModelTrainer

# 1. Carica dataset
train_dataset = DatasetMerged_2d(split='train', img_side=256)
val_dataset = DatasetMerged_2d(split='val', img_side=256)

# 2. Inizializza trainer
trainer = FoundationModelTrainer(
    model_type='medsam',
    learning_rate=1e-4,
    output_dir='checkpoints'
)

# 3. Fine-tune
results = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10,
    samples_per_epoch=100
)

# 4. Mostra risultati
print(f"Dice Finale: {results['train_dice_scores'][-1]:.4f}")
print(f"Miglior Dice Val: {max(results['val_dice_scores']):.4f}")

# 5. Cleanup
trainer.cleanup()
```

### Esempio: Inferenza Pura

```python
from dataset_asoca_cas import DatasetMerged_2d
from train_foundation_models import FoundationModelTrainer

# 1. Carica test dataset
test_dataset = DatasetMerged_2d(split='test', img_side=256)

# 2. Inizializza trainer
trainer = FoundationModelTrainer(model_type='medsam')

# 3. (Opzionale) Carica checkpoint fine-tuned
trainer.load_checkpoint('checkpoints/best_model.pth')

# 4. Esegui inferenza
results = trainer.inference(dataset=test_dataset)

# 5. Mostra risultati
print(f"Dice Medio: {results['avg_dice']:.4f}")

# 6. Cleanup
trainer.cleanup()
```

## 🎓 Esempi Interattivi

Il file `example_finetuning.py` contiene 3 esempi completi:

```bash
# Esempio 1: Fine-tuning
python example_finetuning.py --example 1

# Esempio 2: Inferenza pura
python example_finetuning.py --example 2

# Esempio 3: Confronto SAM vs MedSAM
python example_finetuning.py --example 3

# Tutti gli esempi
python example_finetuning.py --all
```

O usa il menu interattivo:

```bash
python example_finetuning.py
```

## 📋 Parametri CLI

### train_foundation_models.py

```bash
python train_foundation_models.py [OPTIONS]

Opzioni:
  --mode {train,inference}     Modalità: train o inference (default: train)
  --model {sam,medsam}         Modello foundation (default: medsam)
  --epochs INT                 Numero epoche (default: 10)
  --samples-per-epoch INT      Samples per epoca (default: 100)
  --learning-rate FLOAT        Learning rate (default: 1e-4)
  --checkpoint PATH            Path checkpoint per inference/resume
  --output-dir PATH            Directory output (default: checkpoints)
```

### Esempi:

```bash
# Fine-tuning base
python train_foundation_models.py --mode train --model medsam

# Fine-tuning avanzato
python train_foundation_models.py \
    --mode train \
    --model medsam \
    --epochs 20 \
    --samples-per-epoch 200 \
    --learning-rate 5e-5 \
    --output-dir my_checkpoints

# Inferenza con pretrained
python train_foundation_models.py --mode inference --model medsam

# Inferenza con checkpoint
python train_foundation_models.py \
    --mode inference \
    --model medsam \
    --checkpoint checkpoints/best_model.pth
```

## 🔧 Componenti Principali

### FoundationModelTrainer

Classe principale per training e inference:

```python
trainer = FoundationModelTrainer(
    model_type='medsam',        # 'sam' o 'medsam'
    device='cuda',              # 'cuda' o 'cpu'
    learning_rate=1e-4,
    output_dir='checkpoints'
)

# Training
results = trainer.train(
    train_dataset=...,
    val_dataset=...,
    epochs=10
)

# Inference
results = trainer.inference(
    dataset=...,
    save_results=True
)

# Cleanup
trainer.cleanup()
```

### DatasetMerged_2d

Dataset loader per ASOCA + ImageCAS:

```python
dataset = DatasetMerged_2d(
    split='train',              # 'train', 'val', 'test'
    img_side=256,               # Dimensione output
    use_cache=True,             # Abilita caching
    max_cache_size=8            # Max items in cache
)

# Get batch per training
img_batch, lab_batch = dataset.get(minibatch_size=4, out_side=256)

# Get sample per validation/test
data = dataset._get_val_test()
image = data['image']  # [S, 1, H, W]
label = data['label']  # [S, 1, H, W]
```

## 📚 Documentazione Completa

Per informazioni dettagliate, consulta:

- **FINETUNING_GUIDE.md** - Guida completa in italiano
- **example_finetuning.py** - Esempi pratici commentati
- **foundation_models_pipeline.py** - Documentazione modelli
- **dataset_asoca_cas.py** - Struttura dataset

## 🎯 Features Implementate

✅ **Fine-tuning su tutto il dataset**
- Support ASOCA (Normal + Diseased)
- Support ImageCAS
- Split automatico train/val/test

✅ **Dice score medio alla fine del fine-tuning**
- Calcolo durante training
- Calcolo su validation set
- Report finale dettagliato

✅ **Inferenza pura senza training**
- Modalità inference-only
- Support pretrained models
- Support fine-tuned checkpoints

✅ **Loss functions**
- Dice Loss
- Binary Cross-Entropy Loss
- Combined Loss

✅ **Monitoring e logging**
- Progress bars (tqdm)
- Epoch summaries
- Best model tracking
- Training curves

✅ **Checkpointing**
- Salvataggio automatico
- Best model selection
- Resume training support

✅ **Data augmentation**
- Horizontal/vertical flip
- Rotation 90°
- Small shifts
- Noise injection
- Intensity shift

✅ **Preprocessing**
- HU window normalization
- Heart region detection
- Automatic cropping
- Resize to fixed size

## 🔬 Metriche

### Dice Score

```
Dice = (2 * |Prediction ∩ Ground Truth|) / (|Prediction| + |Ground Truth|)
```

- Range: [0, 1]
- 0 = nessuna sovrapposizione
- 1 = sovrapposizione perfetta

### Loss Functions

**Dice Loss:**
```
Dice Loss = 1 - Dice Score
```

**Binary Cross-Entropy:**
```
BCE = -[y*log(p) + (1-y)*log(1-p)]
```

**Combined Loss:**
```
Total Loss = BCE + Dice Loss
```

## 🐛 Risoluzione Problemi

### GPU Out of Memory

```bash
# Riduci samples per epoca
python train_foundation_models.py --samples-per-epoch 50

# Usa immagini più piccole
# Modifica in dataset_asoca_cas.py
dataset = DatasetMerged_2d(img_side=128)
```

### Dataset Non Trovato

1. Verifica i path in `dataset_asoca_cas.py`
2. Controlla la struttura del dataset
3. Verifica i permessi di lettura

### Slow Training

1. Abilita caching: `use_cache=True`
2. Aumenta `max_cache_size`
3. Riduci `val_every` per validare meno spesso
4. Usa GPU più potente

## 📞 Supporto

Per problemi o domande:

1. Consulta **FINETUNING_GUIDE.md**
2. Verifica gli esempi in **example_finetuning.py**
3. Controlla la configurazione dataset
4. Apri una issue su GitHub

## 🏆 Best Practices

1. **Inizia con poche epoche** (3-5) per testare
2. **Monitora validation Dice** per evitare overfitting
3. **Usa learning rate basso** (1e-4 o 1e-5)
4. **Salva checkpoint regolarmente**
5. **Valida su dati mai visti** durante training
6. **Usa GPU** per training veloce
7. **Abilita caching** per dataset grandi

## 📊 Benchmark

Tempi stimati (GPU NVIDIA A100):

| Operazione | Tempo |
|------------|-------|
| 1 epoca training (100 samples) | ~5-7 min |
| Validation (10 samples) | ~2-3 min |
| Inferenza (1 sample) | ~10-15 sec |
| Fine-tuning completo (10 epoche) | ~60-90 min |

## 🎉 Conclusione

Questo progetto fornisce una soluzione completa per:
- ✅ Fine-tuning di foundation models
- ✅ Calcolo Dice score medio
- ✅ Inferenza pura senza training

Tutto integrato con il dataset ASOCA + ImageCAS!

---

**Happy fine-tuning! 🚀**
