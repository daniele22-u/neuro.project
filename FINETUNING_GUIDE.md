# Guida al Fine-tuning e Inferenza con Foundation Models

Questa guida spiega come utilizzare i foundation models (SAM, MedSAM) per:
1. **Fine-tuning** sul dataset completo ASOCA + ImageCAS
2. **Inferenza pura** senza training
3. **Calcolo del Dice score medio** durante e dopo il training

## 📋 Requisiti

### Dataset
Il dataset deve seguire questa struttura:

```
ASOCA/
├── Normal/
│   ├── CTCA/               # Immagini CT normali (.nrrd)
│   ├── Annotations/        # Annotazioni normali (.nrrd)
│   └── Testset_Normal/     # Test set senza annotazioni
└── Diseased/
    ├── CTCA/               # Immagini CT malate (.nrrd)
    ├── Annotations/        # Annotazioni malate (.nrrd)
    └── Testset_Diseased/   # Test set senza annotazioni

ImageCAS/Data/
├── 1.img.nii.gz           # Immagini CAS
├── 1.label.nii.gz         # Label CAS
├── 2.img.nii.gz
├── 2.label.nii.gz
└── ...
```

### Configurazione Paths
Modifica i path in `dataset_asoca_cas.py`:

```python
BASE_DIR_ASOCA = "/path/to/your/ASOCA"
BASE_DIR_CAS = "/path/to/your/ImageCAS/Data"
```

## 🚀 Utilizzo Rapido

### 1. Fine-tuning completo

```bash
# Fine-tune MedSAM per 10 epoche
python train_foundation_models.py --mode train --model medsam --epochs 10

# Fine-tune SAM per 5 epoche
python train_foundation_models.py --mode train --model sam --epochs 5 --learning-rate 1e-5
```

**Output:**
- Checkpoint salvati in `checkpoints/`
- Modello migliore: `checkpoints/best_model.pth`
- Grafici di training: `checkpoints/training_curves.png`
- **Dice score medio** stampato alla fine del training

### 2. Inferenza pura (senza training)

```bash
# Inferenza con MedSAM pretrained
python train_foundation_models.py --mode inference --model medsam

# Inferenza con modello fine-tuned
python train_foundation_models.py --mode inference --model medsam --checkpoint checkpoints/best_model.pth
```

**Output:**
- Predizioni salvate in `inference_results/`
- **Dice score medio** sul test set
- Statistiche dettagliate (min, max, std)

## 📊 Dettagli delle Funzionalità

### Fine-tuning

Il processo di fine-tuning:

1. **Carica il modello pretrained** (SAM o MedSAM)
2. **Imposta il modello in modalità training**
3. **Itera sul dataset** usando la funzione `get()` per ottenere batch
4. **Calcola le loss:**
   - Binary Cross-Entropy Loss
   - Dice Loss
   - Combined Loss = BCE + Dice
5. **Calcola il Dice score** per monitorare le performance
6. **Salva il checkpoint** quando il Dice score migliora
7. **Valida sul validation set** ogni epoca

**Parametri configurabili:**

```bash
python train_foundation_models.py \
    --mode train \
    --model medsam \
    --epochs 20 \
    --samples-per-epoch 200 \
    --learning-rate 1e-4 \
    --output-dir my_checkpoints
```

### Inferenza

Il processo di inferenza:

1. **Carica il modello** (pretrained o fine-tuned)
2. **Imposta il modello in modalità eval**
3. **Itera sul test set**
4. **Per ogni sample:**
   - Estrae una slice centrale
   - Genera la predizione
   - Calcola il Dice score (se label disponibile)
5. **Calcola statistiche:**
   - Dice medio
   - Deviazione standard
   - Min/Max Dice
6. **Salva i risultati**

### Calcolo del Dice Score

Il Dice score viene calcolato come:

```python
Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
```

Dove:
- X = predizione binarizzata (threshold 0.5)
- Y = ground truth
- |·| = numero di pixel

**Range:** 0 (peggiore) - 1 (perfetto)

## 💻 Uso Programmatico

### Esempio 1: Fine-tuning Completo

```python
from dataset_asoca_cas import DatasetMerged_2d
from train_foundation_models import FoundationModelTrainer

# Carica i dataset
train_dataset = DatasetMerged_2d(split='train', img_side=256)
val_dataset = DatasetMerged_2d(split='val', img_side=256)

# Inizializza il trainer
trainer = FoundationModelTrainer(
    model_type='medsam',
    learning_rate=1e-4,
    output_dir='checkpoints'
)

# Fine-tune
results = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10,
    samples_per_epoch=100,
    val_every=1,
    save_every=2
)

# Risultati
print(f"Dice Finale (Train): {results['train_dice_scores'][-1]:.4f}")
print(f"Miglior Dice (Val): {max(results['val_dice_scores']):.4f}")

# Cleanup
trainer.cleanup()
```

### Esempio 2: Inferenza Pura

```python
from dataset_asoca_cas import DatasetMerged_2d
from train_foundation_models import FoundationModelTrainer

# Carica test dataset
test_dataset = DatasetMerged_2d(split='test', img_side=256)

# Inizializza trainer
trainer = FoundationModelTrainer(
    model_type='medsam',
    output_dir='inference_results'
)

# Opzionale: carica checkpoint fine-tuned
# trainer.load_checkpoint('checkpoints/best_model.pth')

# Esegui inferenza
results = trainer.inference(
    dataset=test_dataset,
    save_results=True
)

# Risultati
print(f"Dice Medio: {results['avg_dice']:.4f}")
print(f"Std: {np.std(results['all_dice_scores']):.4f}")

# Cleanup
trainer.cleanup()
```

### Esempio 3: Training Step Singolo

```python
# Per controllo fine-grained del training
trainer = FoundationModelTrainer(model_type='medsam')

# Get a sample
img_batch, lab_batch = train_dataset.get(minibatch_size=1, out_side=256)
img = img_batch[0]  # [1, H, W]
gt = lab_batch[0]   # [1, H, W]

# Single training step
metrics = trainer.train_step(img, gt, boxes=None)

print(f"Loss: {metrics['loss']:.4f}")
print(f"Dice: {metrics['dice_score']:.4f}")
```

## 📈 Monitoraggio Training

Durante il training, vengono stampate informazioni dettagliate:

```
================================================================
Epoch 1/10
================================================================
Training: 100%|███████████| 100/100 [05:23<00:00, loss=0.4235, dice=0.7231]

📊 Epoch 1 Summary:
   Train Loss: 0.4235
   Train Dice: 0.7231

🔍 Running validation...
Validating: 100%|███████████| 10/10 [02:15<00:00]
   Val Dice:   0.7456 (on 50 samples)
   💾 New best model saved! (Dice: 0.7456)
   💾 Checkpoint saved: checkpoints/checkpoint_epoch_1.pth
```

Alla fine del training:

```
✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
FINE-TUNING COMPLETED
✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

Final Training Dice: 0.8123
Best Validation Dice: 0.7892
```

## 🔧 Parametri Avanzati

### Trainer Configuration

```python
trainer = FoundationModelTrainer(
    model_type='medsam',        # 'sam' o 'medsam'
    device='cuda',              # 'cuda' o 'cpu'
    learning_rate=1e-4,         # Learning rate
    output_dir='checkpoints'    # Directory per checkpoint
)
```

### Training Configuration

```python
results = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10,                  # Numero di epoche
    samples_per_epoch=100,      # Samples per epoca
    val_every=1,                # Valida ogni N epoche
    save_every=2                # Salva checkpoint ogni N epoche
)
```

### Dataset Configuration

```python
dataset = DatasetMerged_2d(
    split='train',              # 'train', 'val', 'test'
    img_side=256,               # Dimensione immagini
    use_cache=True,             # Cache per speedup
    max_cache_size=8            # Max items in cache
)
```

## 📁 Struttura Output

```
checkpoints/
├── best_model.pth              # Miglior modello (best val Dice)
├── checkpoint_epoch_1.pth      # Checkpoint epoca 1
├── checkpoint_epoch_2.pth      # Checkpoint epoca 2
└── training_curves.png         # Grafici loss e Dice

inference_results/
├── inference_medsam.pth        # Risultati inferenza MedSAM
└── inference_sam.pth           # Risultati inferenza SAM
```

### Formato Checkpoint

```python
checkpoint = {
    'epoch': 5,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'dice_score': 0.7892
}
```

### Formato Risultati Inferenza

```python
results = {
    'predictions': [
        {
            'id': 'test-MERGED-Normal_1',
            'prediction': np.array(...),  # [1, H, W]
            'dice': 0.8234
        },
        ...
    ],
    'avg_dice': 0.7654,
    'all_dice_scores': [0.8234, 0.7123, ...]
}
```

## 🎯 Best Practices

### 1. Fine-tuning

- **Inizia con poche epoche** (3-5) per testare
- **Usa learning rate basso** (1e-4 o 1e-5) per foundation models
- **Monitora validation Dice** per evitare overfitting
- **Salva checkpoint regolarmente** per recovery

### 2. Inferenza

- **Usa modelli fine-tuned** per performance migliori
- **Test su dati mai visti** per valutazione onesta
- **Analizza slice-by-slice** per dataset 3D
- **Salva predizioni** per analisi post-hoc

### 3. Dataset

- **Verifica paths** prima di iniziare
- **Controlla disponibilità GPU** per speedup
- **Usa caching** per dataset ripetitivi
- **Balance train/val split** (70/15/15)

## 🐛 Troubleshooting

### Out of Memory (GPU)

```python
# Riduci samples per epoca
trainer.train(samples_per_epoch=50)

# Usa immagini più piccole
dataset = DatasetMerged_2d(img_side=128)
```

### Dataset Non Trovato

```python
# Verifica paths in dataset_asoca_cas.py
BASE_DIR_ASOCA = "/correct/path/to/ASOCA"
BASE_DIR_CAS = "/correct/path/to/ImageCAS"
```

### Slow Training

```python
# Abilita cache
dataset = DatasetMerged_2d(use_cache=True, max_cache_size=16)

# Riduci validation frequency
trainer.train(val_every=5)
```

## 📚 Riferimenti

- **SAM**: [Segment Anything Model](https://segment-anything.com/)
- **MedSAM**: [Medical SAM](https://github.com/bowang-lab/MedSAM)
- **ASOCA**: ASOCA Challenge Dataset
- **ImageCAS**: ImageCAS Dataset

## 💡 Esempi Completi

Vedi `example_finetuning.py` per esempi completi:

```bash
# Esempio 1: Fine-tuning
python example_finetuning.py --example 1

# Esempio 2: Inferenza
python example_finetuning.py --example 2

# Esempio 3: Confronto SAM vs MedSAM
python example_finetuning.py --example 3

# Tutti gli esempi
python example_finetuning.py --all
```

## 📞 Supporto

Per problemi o domande:
1. Controlla questa guida
2. Verifica i file di esempio
3. Controlla la configurazione del dataset
4. Apri una issue su GitHub

---

**Buon fine-tuning! 🎯**
