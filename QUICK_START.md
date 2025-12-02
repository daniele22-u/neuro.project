# 🚀 Quick Start - Foundation Models Pipeline

Guida rapida per iniziare ad usare la pipeline di segmentazione con modelli foundation.

## 📋 Prerequisiti

- Python 3.8+
- CUDA-capable GPU (raccomandato, ma funziona anche su CPU)
- 8GB+ RAM (16GB+ raccomandato con GPU)

## ⚡ Installazione Veloce

```bash
# 1. Clona il repository (se non già fatto)
git clone https://github.com/daniele22-u/neuro.project.git
cd neuro.project

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. (Opzionale) Login Hugging Face per alcuni modelli
huggingface-cli login
```

## 🎯 Uso Base

### Opzione 1: Run All (Più Semplice)

Esegui tutti i modelli con dati di esempio:

```bash
python run_all_foundation_models.py
```

Questo eseguirà:
- ✅ SAM (Segment Anything)
- ✅ MedSAM (Medical SAM)
- ✅ CLIP (Feature extraction)
- ✅ ViT (Vision Transformer)

I risultati saranno salvati in `results/foundation_models_results.png`

### Opzione 2: Con La Tua Immagine

```bash
python run_all_foundation_models.py --image path/to/your/image.png
```

### Opzione 3: Codice Python

```python
from foundation_models_pipeline import FoundationModelsPipeline
import torch

# Carica la tua immagine (formato: [C, H, W])
image = torch.rand(1, 256, 256)  # Sostituisci con la tua immagine

# Crea e esegui pipeline
pipeline = FoundationModelsPipeline(device="cuda")
results = pipeline.run_all(image=image)

# Visualizza
pipeline.visualize_results(image, save_path="results.png")
```

## 📚 Esempi Completi

### Esempio 1: Integrazione con Dataset ASOCA

```bash
python example_integration.py
```

Questo mostra come integrare con il dataset esistente da `Neuro_Dani-4.py`.

### Esempio 2: Solo MedSAM

```python
from foundation_models_pipeline import MedSAMModel

# Inizializza
medsam = MedSAMModel(device="cuda")
medsam.load()

# Predizione con bounding box
boxes = [[50, 50, 200, 200]]  # [x_min, y_min, x_max, y_max]
mask = medsam.predict(image, boxes=boxes)

# Cleanup
medsam.unload()
```

### Esempio 3: Analisi con CLIP

```python
from foundation_models_pipeline import CLIPViTModel

clip = CLIPViTModel(device="cuda")
clip.load()

result = clip.predict(
    image, 
    text_prompt="coronary artery"
)

print(f"Similarity: {result['similarity']:.4f}")
clip.unload()
```

## 🎨 Struttura Output

Dopo l'esecuzione, troverai:

```
results/
├── foundation_models_results.png    # Visualizzazione comparativa
├── asoca_foundation_models_example.png  # Esempio ASOCA
└── [altri risultati...]
```

## 🔧 Parametri Comuni

### Bounding Boxes (per SAM/MedSAM)

```python
# Singolo box
boxes = [[x_min, y_min, x_max, y_max]]

# Box intorno al centro dell'immagine
H, W = 256, 256
boxes = [[W//4, H//4, 3*W//4, 3*H//4]]
```

### Text Prompts (per CLIP)

Esempi di prompt efficaci:
- `"coronary artery"` - arterie coronarie
- `"medical anatomical structure"` - strutture anatomiche
- `"blood vessel in CT scan"` - vasi sanguigni in CT
- `"cardiac tissue"` - tessuto cardiaco

## 💡 Tips & Tricks

### 1. Out of Memory?

Riduci la dimensione dell'immagine:

```python
import torch.nn.functional as F
image = F.interpolate(image.unsqueeze(0), size=(128, 128), mode='bilinear')
image = image.squeeze(0)
```

### 2. CPU invece di GPU

Il codice fallback automaticamente su CPU se CUDA non è disponibile:

```python
pipeline = FoundationModelsPipeline(device="cpu")
```

### 3. Esegui Solo Alcuni Modelli

```python
pipeline = FoundationModelsPipeline()

# Solo SAM e MedSAM
pipeline.run_sam(image)
pipeline.run_medsam(image)
```

### 4. Salva Maschere di Segmentazione

```python
results = pipeline.run_all(image)

# Salva maschere
torch.save(results['sam'], 'sam_mask.pt')
torch.save(results['medsam'], 'medsam_mask.pt')
```

## 🐛 Problemi Comuni

### ImportError: No module named 'transformers'

```bash
pip install transformers accelerate
```

### CUDA out of memory

Riduci batch size o dimensione immagine, oppure usa CPU:

```python
pipeline = FoundationModelsPipeline(device="cpu")
```

### Model download failed

Assicurati di avere accesso a internet e fai login su Hugging Face:

```bash
huggingface-cli login
```

## 📖 Documentazione Completa

Per documentazione dettagliata, consulta:
- **README_FOUNDATION_MODELS.md** - Documentazione completa in italiano
- **example_integration.py** - Esempi di integrazione
- **foundation_models_pipeline.py** - Documentazione inline nel codice

## 🎓 Prossimi Passi

1. ✅ Esegui `python run_all_foundation_models.py` per testare
2. ✅ Prova con le tue immagini mediche
3. ✅ Integra con il tuo dataset ASOCA
4. ✅ Personalizza i modelli e i parametri
5. ✅ Sperimenta con diversi text prompts per CLIP

## 💬 Supporto

Per problemi o domande:
1. Controlla la documentazione completa in README_FOUNDATION_MODELS.md
2. Guarda gli esempi in example_integration.py
3. Apri una issue su GitHub

---

**Buona segmentazione! 🎯🔬**
