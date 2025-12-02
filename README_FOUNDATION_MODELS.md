# Foundation Models Segmentation Pipeline

Pipeline GPU-based per la segmentazione di immagini mediche utilizzando modelli foundation (SAM, MedSAM, CLIP, ViT).

## 🎯 Caratteristiche

- **Esecuzione sequenziale**: I modelli vengono eseguiti uno dopo l'altro con pulizia automatica della memoria GPU
- **Gestione memoria ottimizzata**: Dopo ogni modello, la memoria GPU viene liberata prima di caricare il successivo
- **Supporto multipli modelli**:
  - **SAM** (Segment Anything Model) - Segmentazione generale
  - **MedSAM** - Segmentazione specializzata per immagini mediche
  - **CLIP** - Estrazione features e analisi basata su testo
  - **ViT** (Vision Transformer) - Modelli transformer generici per feature extraction
- **Pipeline "run all"**: Esegui tutti i modelli con un singolo comando
- **Visualizzazione risultati**: Genera automaticamente visualizzazioni comparative

## 📦 Installazione

### 1. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 2. Verifica CUDA (opzionale ma raccomandato)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 🚀 Utilizzo

### Opzione 1: Run All (Esecuzione Completa)

Esegui tutti i modelli foundation in sequenza:

```bash
python run_all_foundation_models.py
```

Questo script:
1. Crea dati di esempio (se non forniti)
2. Esegue SAM, MedSAM, CLIP e ViT in sequenza
3. Pulisce la memoria GPU tra ogni modello
4. Genera visualizzazioni dei risultati
5. Salva i risultati nella cartella `results/`

### Opzione 2: Con Immagine Custom

```bash
python run_all_foundation_models.py --image path/to/your/image.png
```

### Opzione 3: Utilizzo Programmatico

```python
from foundation_models_pipeline import FoundationModelsPipeline
import torch

# Crea pipeline
pipeline = FoundationModelsPipeline(device="cuda")

# Carica la tua immagine (formato: [C, H, W])
image = torch.rand(1, 256, 256)  # Esempio con immagine random

# Opzionale: definisci bounding boxes per segmentazione
boxes = [[50, 50, 200, 200]]  # [x_min, y_min, x_max, y_max]

# Esegui tutti i modelli
results = pipeline.run_all(
    image=image,
    boxes=boxes,
    text_prompt="coronary artery"
)

# Visualizza risultati
pipeline.visualize_results(image, save_path="my_results.png")
```

### Opzione 4: Esegui Modelli Singolarmente

```python
from foundation_models_pipeline import (
    SAMModel, 
    MedSAMModel, 
    CLIPViTModel, 
    GenericViTModel
)

# Esempio con MedSAM
medsam = MedSAMModel(device="cuda")
medsam.load()

# Predizione
mask = medsam.predict(image, boxes=boxes)

# Pulizia memoria
medsam.unload()
```

## 📁 Struttura File

```
neuro.project/
├── foundation_models_pipeline.py   # Classi base e pipeline
├── run_all_foundation_models.py    # Script principale run-all
├── segmentation.py                 # Loss functions e metriche
├── requirements.txt                # Dipendenze Python
├── README_FOUNDATION_MODELS.md     # Questa documentazione
└── results/                        # Output (creato automaticamente)
    └── foundation_models_results.png
```

## 🔧 Modelli Supportati

### SAM (Segment Anything Model)

```python
from foundation_models_pipeline import SAMModel

sam = SAMModel(device="cuda")
sam.load()
mask = sam.predict(image, boxes=[[x0, y0, x1, y1]])
sam.unload()
```

**Caratteristiche**:
- Segmentazione general-purpose
- Supporta prompt: boxes, points
- Modello: `facebook/sam-vit-base`

### MedSAM (Medical SAM)

```python
from foundation_models_pipeline import MedSAMModel

medsam = MedSAMModel(device="cuda")
medsam.load()
mask = medsam.predict(image, boxes=[[x0, y0, x1, y1]])
medsam.unload()
```

**Caratteristiche**:
- Specializzato per immagini mediche
- Pre-trained su dataset medicali
- Modello: `flaviagiammarino/medsam-vit-base`

### CLIP (Contrastive Language-Image Pre-training)

```python
from foundation_models_pipeline import CLIPViTModel

clip = CLIPViTModel(device="cuda")
clip.load()
result = clip.predict(image, text_prompt="coronary artery")
print(f"Similarity: {result['similarity']}")
clip.unload()
```

**Caratteristiche**:
- Analisi immagini basata su testo
- Feature extraction multi-modale
- Modello: `openai/clip-vit-base-patch32`

### ViT (Vision Transformer)

```python
from foundation_models_pipeline import GenericViTModel

vit = GenericViTModel(device="cuda")
vit.load()
features = vit.predict(image)
print(f"Features shape: {features['last_hidden_state'].shape}")
vit.unload()
```

**Caratteristiche**:
- Transformer generico per visione
- Feature extraction di alta qualità
- Modello: `google/vit-base-patch16-224`

## 💾 Gestione Memoria GPU

Il pipeline implementa una gestione ottimale della memoria:

```python
def cleanup_memory():
    """Pulisce memoria GPU e CPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
```

Questa funzione viene chiamata automaticamente:
- Dopo ogni modello nel `run_all()`
- Quando si chiama `model.unload()`
- Alla fine dell'esecuzione

## 📊 Output e Risultati

### Formato Risultati

Il metodo `pipeline.run_all()` ritorna un dizionario:

```python
{
    'sam': torch.Tensor,        # Maschera di segmentazione [1, H, W]
    'medsam': torch.Tensor,     # Maschera di segmentazione [1, H, W]
    'clip': {                   # Features e similarity
        'image_features': torch.Tensor,
        'text_features': torch.Tensor,
        'similarity': float
    },
    'vit': {                    # Features estratte
        'last_hidden_state': torch.Tensor,
        'pooler_output': torch.Tensor
    }
}
```

### Visualizzazione

La visualizzazione automatica mostra:
- Immagine originale
- Segmentazione SAM (overlay rosso)
- Segmentazione MedSAM (overlay blu)

## 🔍 Integrazione con Dataset Esistente

Per integrare con il codice esistente in `Neuro_Dani-4.py`:

```python
from Neuro_Dani-4 import DatasetMerged_2d
from foundation_models_pipeline import FoundationModelsPipeline

# Carica dataset
dataset = DatasetMerged_2d(split='val', img_side=256)

# Inizializza pipeline
pipeline = FoundationModelsPipeline(device="cuda")

# Loop sul dataset
for patient_data in dataset:
    img = patient_data['image']  # Assumendo formato [S, 1, H, W]
    
    # Prendi una slice
    slice_img = img[img.shape[0] // 2]  # Slice centrale [1, H, W]
    
    # Esegui pipeline
    results = pipeline.run_all(slice_img)
    
    # Visualizza
    pipeline.visualize_results(slice_img)
```

## ⚙️ Configurazione Avanzata

### Cambiare Modello

```python
# Usa un ViT diverso
vit = GenericViTModel(
    device="cuda",
    model_name="google/vit-large-patch16-224"
)
```

### Personalizzare Pipeline

```python
class CustomPipeline(FoundationModelsPipeline):
    def run_all(self, image, **kwargs):
        # Esegui solo SAM e CLIP
        self.run_sam(image)
        self.run_clip(image)
        return self.results
```

## 🐛 Troubleshooting

### CUDA Out of Memory

Se ricevi errori di memoria:

```python
# Riduci dimensione batch o immagine
image = F.interpolate(image.unsqueeze(0), size=(128, 128))
```

### Modelli non disponibili

Alcuni modelli richiedono autenticazione Hugging Face:

```bash
# Login Hugging Face
huggingface-cli login
```

### Import errors

Assicurati che tutte le dipendenze siano installate:

```bash
pip install --upgrade -r requirements.txt
```

## 📝 Esempi Completi

### Esempio 1: Segmentazione Coronarie

```python
from foundation_models_pipeline import FoundationModelsPipeline
import torch

# Carica immagine CT coronaria
image = load_coronary_ct_image()  # [1, 256, 256]

# Pipeline
pipeline = FoundationModelsPipeline(device="cuda")

# Esegui con prompt specifico
results = pipeline.run_all(
    image=image,
    text_prompt="coronary artery vessel"
)

# Salva maschere
torch.save(results['medsam'], 'coronary_mask.pt')
```

### Esempio 2: Batch Processing

```python
from pathlib import Path
from foundation_models_pipeline import MedSAMModel

medsam = MedSAMModel(device="cuda")
medsam.load()

# Processa multiple immagini
for img_path in Path("images/").glob("*.png"):
    image = load_image(img_path)
    mask = medsam.predict(image)
    save_mask(mask, f"masks/{img_path.stem}_mask.png")

medsam.unload()
```

## 📚 Riferimenti

- **SAM**: [Segment Anything (Meta AI)](https://segment-anything.com/)
- **MedSAM**: [Medical SAM](https://github.com/bowang-lab/MedSAM)
- **CLIP**: [OpenAI CLIP](https://openai.com/research/clip)
- **ViT**: [Vision Transformer (Google)](https://github.com/google-research/vision_transformer)

## 📄 Licenza

Questo codice è fornito come parte del progetto neuro.project.

## 🤝 Contributi

Per domande o problemi, apri una issue su GitHub.

---

**Buona segmentazione! 🎯**
