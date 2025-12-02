# 📘 Guida Kaggle - Foundation Models Pipeline

Guida completa per utilizzare la pipeline di foundation models su Kaggle Notebooks.

## 🎯 Setup Completo su Kaggle

### Step 1: Crea un Dataset con i File Python

1. **Prepara i File**

   I file necessari da caricare come dataset:
   ```
   - segmentation.py
   - foundation_models_pipeline.py
   - run_all_foundation_models.py
   - example_integration.py
   ```

2. **Crea il Dataset su Kaggle**

   - Vai su [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Clicca su "New Dataset"
   - Carica tutti i 4 file Python sopra elencati
   - Dai un nome al dataset: `foundation-models-segmentation`
   - Descrizione: "Foundation models pipeline for medical image segmentation"
   - Imposta come **Public** (o Private se preferisci)
   - Clicca "Create"

3. **Copia il Path del Dataset**

   Una volta creato, il path sarà qualcosa tipo:
   ```
   /kaggle/input/foundation-models-segmentation
   ```

### Step 2: Crea un Notebook Kaggle

1. **Nuovo Notebook**
   
   - Vai su [Kaggle Notebooks](https://www.kaggle.com/notebooks)
   - Clicca "New Notebook"
   - Scegli "GPU" come acceleratore (raccomandato)
   - Settings → Accelerator → GPU P100 o GPU T4

2. **Aggiungi il Dataset al Notebook**
   
   - Nel pannello destro, clicca "Add data"
   - Cerca il tuo dataset: `foundation-models-segmentation`
   - Clicca "Add"

### Step 3: Setup nel Notebook

#### Cella 1: Installazione Dipendenze

```python
# Installa le dipendenze necessarie
!pip install -q transformers accelerate sentencepiece
!pip install -q SimpleITK scikit-image

import sys
import os

# Verifica che il dataset sia montato
dataset_path = '/kaggle/input/foundation-models-segmentation'
print("✓ Dataset path:", dataset_path)
print("✓ Files nel dataset:", os.listdir(dataset_path))
```

#### Cella 2: Import dei Moduli

```python
# Aggiungi il path del dataset al sys.path
import sys
sys.path.insert(0, '/kaggle/input/foundation-models-segmentation')

# Importa i moduli della pipeline
import segmentation
import foundation_models_pipeline as fmp
from foundation_models_pipeline import (
    FoundationModelsPipeline,
    SAMModel,
    MedSAMModel,
    CLIPViTModel,
    GenericViTModel,
    cleanup_memory
)

print("✅ Moduli importati con successo!")
```

#### Cella 3: Verifica GPU

```python
import torch

# Verifica CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 4: Esegui la Pipeline

#### Opzione A: Con Dati di Esempio

```python
# Crea dati di esempio
import torch
import numpy as np

H, W = 256, 256
image = torch.zeros(1, H, W, dtype=torch.float32)

# Aggiungi strutture simulate
y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
center_y, center_x = H // 2, W // 2
radius = 60
dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
image[0, dist < radius] = 0.8

# Aggiungi rumore
noise = torch.randn(1, H, W) * 0.05
image = torch.clamp(image + noise, 0, 1)

print(f"✓ Immagine creata: {image.shape}")

# Inizializza pipeline
pipeline = FoundationModelsPipeline(device=device)

# ⚠️ IMPORTANTE: Usa mode="test" per evitare data leakage
results = pipeline.run_all(
    image=image,
    boxes=None,  # NO boxes in test mode!
    text_prompt="medical anatomical structure",
    mode="test"  # TEST MODE - No data leakage
)

# Visualizza risultati
pipeline.visualize_results(image, save_path="results.png")
```

#### Opzione B: Con Dataset ASOCA (se disponibile)

```python
# Se hai il dataset ASOCA su Kaggle
from pathlib import Path

# Path al dataset ASOCA (modifica secondo il tuo setup)
asoca_path = Path("/kaggle/input/asoca-dataset")

if asoca_path.exists():
    # Carica un'immagine dal dataset
    # (implementa il caricamento secondo la struttura del tuo dataset)
    
    # Esempio con SimpleITK
    import SimpleITK as sitk
    
    img_path = asoca_path / "Normal" / "CTCA" / "Normal_1.nrrd"
    sitk_img = sitk.ReadImage(str(img_path))
    img_np = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    
    # Normalizzazione
    img_np = np.clip(img_np, -200.0, 800.0)
    img_np = (img_np + 200.0) / 1000.0
    img_np = np.clip(img_np, 0.0, 1.0)
    
    # Prendi una slice centrale
    mid_slice = img_np.shape[0] // 2
    slice_img = torch.from_numpy(img_np[mid_slice]).unsqueeze(0)  # [1, H, W]
    
    # Resize a 256x256
    import torch.nn.functional as F
    slice_img = F.interpolate(
        slice_img.unsqueeze(0),
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    print(f"✓ Slice caricata: {slice_img.shape}")
    
    # Esegui pipeline in TEST MODE
    pipeline = FoundationModelsPipeline(device=device)
    results = pipeline.run_all(
        image=slice_img,
        boxes=None,  # NO boxes in test mode
        text_prompt="coronary artery",
        mode="test"  # TEST MODE
    )
    
    pipeline.visualize_results(slice_img, save_path="asoca_results.png")
else:
    print("⚠️ Dataset ASOCA non trovato")
```

### Step 5: Salva i Risultati

```python
# Salva le maschere di segmentazione
if 'sam' in results:
    torch.save(results['sam'], 'sam_mask.pt')
    print("✓ SAM mask salvata")

if 'medsam' in results:
    torch.save(results['medsam'], 'medsam_mask.pt')
    print("✓ MedSAM mask salvata")

# Salva features CLIP
if 'clip' in results:
    torch.save(results['clip']['image_features'], 'clip_features.pt')
    print("✓ CLIP features salvate")

# Salva features ViT
if 'vit' in results:
    torch.save(results['vit']['last_hidden_state'], 'vit_features.pt')
    print("✓ ViT features salvate")

print("\n✅ Tutti i risultati salvati!")
```

## 🔥 Template Notebook Completo

Ecco un template completo da copiare-incollare:

```python
# =============================================================================
# CELLA 1: SETUP E INSTALLAZIONE
# =============================================================================

!pip install -q transformers accelerate sentencepiece SimpleITK scikit-image

import sys
sys.path.insert(0, '/kaggle/input/foundation-models-segmentation')

import torch
import numpy as np
from foundation_models_pipeline import FoundationModelsPipeline, cleanup_memory

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device: {device}")

# =============================================================================
# CELLA 2: CREA DATI DI ESEMPIO
# =============================================================================

H, W = 256, 256
image = torch.zeros(1, H, W)

y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
center_y, center_x = H // 2, W // 2

# Struttura centrale
dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
image[0, dist < 60] = 0.8

# Strutture periferiche
for i in range(3):
    angle = i * np.pi * 2 / 3
    cx = int(center_x + 40 * np.cos(angle))
    cy = int(center_y + 40 * np.sin(angle))
    dist = torch.sqrt((y - cy)**2 + (x - cx)**2)
    image[0, dist < 20] = 0.6

# Rumore
image += torch.randn(1, H, W) * 0.05
image = torch.clamp(image, 0, 1)

print(f"✓ Image shape: {image.shape}")

# =============================================================================
# CELLA 3: ESEGUI PIPELINE (TEST MODE - NO DATA LEAKAGE)
# =============================================================================

pipeline = FoundationModelsPipeline(device=device)

# ⚠️ IMPORTANTE: mode="test" per evitare data leakage
results = pipeline.run_all(
    image=image,
    boxes=None,  # NO boxes in test mode
    text_prompt="medical structure",
    mode="test"  # TEST MODE
)

# =============================================================================
# CELLA 4: VISUALIZZA E SALVA RISULTATI
# =============================================================================

pipeline.visualize_results(image, save_path="foundation_models_results.png")

# Analisi risultati
print("\n📊 RISULTATI:")
print(f"SAM coverage: {(results['sam'] > 0.5).float().mean().item()*100:.2f}%")
print(f"MedSAM coverage: {(results['medsam'] > 0.5).float().mean().item()*100:.2f}%")
print(f"CLIP similarity: {results['clip']['similarity']:.4f}")

# Cleanup finale
cleanup_memory()
print("\n✅ COMPLETATO!")
```

## ⚙️ Modalità di Esecuzione: Train vs Test

### Training/Validation Mode (con Ground Truth)

```python
# ⚠️ USA SOLO durante training/validation
# Quando hai accesso alle label ground truth
results = pipeline.run_all(
    image=image,
    boxes=[[x_min, y_min, x_max, y_max]],  # Boxes dalla GT
    text_prompt="coronary artery",
    mode="train"  # o mode="val"
)
```

### Test Mode (SENZA Ground Truth)

```python
# ✅ USA SEMPRE per test/inference
# NO boxes dalla ground truth = NO data leakage
results = pipeline.run_all(
    image=image,
    boxes=None,  # NO boxes in test
    text_prompt="coronary artery",
    mode="test"  # TEST MODE
)
```

## 🎓 Esempio con Batch Processing

```python
# Processing multiplo con cleanup memoria
import torch.nn.functional as F

pipeline = FoundationModelsPipeline(device=device)

# Lista di immagini
images = [torch.rand(1, 256, 256) for _ in range(5)]

all_results = []

for i, img in enumerate(images):
    print(f"\n{'='*60}")
    print(f"Processing image {i+1}/{len(images)}")
    print(f"{'='*60}")
    
    # Esegui in TEST mode
    results = pipeline.run_all(
        image=img,
        boxes=None,  # NO boxes
        mode="test"  # TEST MODE
    )
    
    all_results.append(results)
    
    # Salva visualizzazione
    pipeline.visualize_results(img, save_path=f"result_{i+1}.png")
    
    # Cleanup tra immagini
    cleanup_memory()

print(f"\n✅ Processate {len(images)} immagini!")
```

## 🐛 Troubleshooting Kaggle

### Out of Memory

```python
# Riduci dimensione immagine
image = F.interpolate(
    image.unsqueeze(0),
    size=(128, 128),
    mode='bilinear'
).squeeze(0)
```

### Import Error

```python
# Verifica il path
import os
print(os.listdir('/kaggle/input/foundation-models-segmentation'))

# Verifica sys.path
import sys
print(sys.path)
```

### Modelli non scaricano

```python
# Usa cache Kaggle
import os
os.environ['HF_HOME'] = '/kaggle/working/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/kaggle/working/hf_cache'
```

## 📋 Checklist Pre-Esecuzione

- [ ] Dataset caricato su Kaggle
- [ ] Dataset aggiunto al notebook
- [ ] GPU abilitata nel notebook
- [ ] Dipendenze installate
- [ ] `sys.path.insert()` eseguito
- [ ] Moduli importati correttamente
- [ ] **Mode="test" per evitare data leakage**
- [ ] `boxes=None` in test mode

## 🎯 Best Practices

1. **Sempre usa `mode="test"`** quando fai inference/test
2. **Mai usare boxes dalla GT** durante il test
3. **Fai cleanup** della memoria tra batch
4. **Salva i risultati** prima della fine del notebook
5. **Usa GPU P100/T4** per performance migliori

## 📚 Link Utili

- [Kaggle GPU Quota](https://www.kaggle.com/discussions/product-feedback)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [SAM Model Card](https://huggingface.co/facebook/sam-vit-base)
- [MedSAM Paper](https://arxiv.org/abs/2304.12306)

---

**Buon lavoro su Kaggle! 🚀**
