# 🚀 Quick Start: Fine-tuning Foundation Models

## In 3 passi

### 1️⃣ Configura il Dataset

Modifica i path in `dataset_asoca_cas.py`:

```python
# Linee 9-10
BASE_DIR_ASOCA = "/path/to/your/ASOCA"
BASE_DIR_CAS = "/path/to/your/ImageCAS/Data"
```

### 2️⃣ Installa le Dipendenze

```bash
pip install -r requirements.txt
```

### 3️⃣ Esegui!

#### Fine-tuning (con Dice medio alla fine):

```bash
python train_foundation_models.py --mode train --model medsam --epochs 10
```

**Output:**
```
✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
FINE-TUNING COMPLETED
✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

Final Training Dice: 0.8123  ← DICE MEDIO!
Best Validation Dice: 0.7892
```

#### Inferenza Pura (senza training):

```bash
python train_foundation_models.py --mode inference --model medsam
```

**Output:**
```
📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊
INFERENCE RESULTS
📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊📊
Average Dice Score: 0.7845  ← DICE MEDIO!
Std Dice Score: 0.0856
```

## ✨ Fatto!

Hai implementato con successo:
- ✅ Fine-tuning su tutto il dataset ASOCA + ImageCAS
- ✅ Dice score medio alla fine del processo
- ✅ Inferenza pura senza training

## 📚 Documentazione Completa

- **FINETUNING_GUIDE.md** - Guida dettagliata
- **README_FINETUNING.md** - Features complete
- **example_finetuning.py** - Esempi pratici

## 🆘 Problemi?

1. Dataset non trovato? Verifica i path in `dataset_asoca_cas.py`
2. GPU out of memory? Riduci `--samples-per-epoch`
3. Domande? Consulta `FINETUNING_GUIDE.md`

---

**Happy fine-tuning! 🎯**
