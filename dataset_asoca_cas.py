# =========================
# CONFIG + PATH + SPLIT
# =========================

import os
import random
import numpy as np
from types import SimpleNamespace

# >>> CAMBIA SOLO QUESTO <<< (se serve)
BASE_DIR_ASOCA = "/content/drive/MyDrive/Colab Notebooks/Neuroengineering/Challenge/ASOCA"
BASE_DIR_CAS   = "/content/drive/MyDrive/Colab Notebooks/Neuroengineering/Challenge/ImageCAS/Data"

# struttura attesa ASOCA:
# BASE_DIR_ASOCA/
#   Normal/CTCA/
#   Normal/Annotations/
#   Diseased/CTCA/
#   Diseased/Annotations/

NORMAL_CT_DIR       = os.path.join(BASE_DIR_ASOCA, "Normal",   "CTCA")
NORMAL_LABEL_DIR    = os.path.join(BASE_DIR_ASOCA, "Normal",   "Annotations")
DISEASED_CT_DIR     = os.path.join(BASE_DIR_ASOCA, "Diseased", "CTCA")
DISEASED_LABEL_DIR  = os.path.join(BASE_DIR_ASOCA, "Diseased", "Annotations")

# Cartelle test (solo immagini, senza label)
TEST_NORMAL_CT_DIR   = os.path.join(BASE_DIR_ASOCA, "Normal",   "Testset_Normal")
TEST_DISEASED_CT_DIR = os.path.join(BASE_DIR_ASOCA, "Diseased", "Testset_Diseased")


def _get_case_ids(ct_folder):
    """ID (nome file senza estensione) da una cartella ASOCA (.nii/.nii.gz/.nrrd/.mha)."""
    if not os.path.isdir(ct_folder):
        return []
    ids = []
    for f in os.listdir(ct_folder):
        name = f.lower()
        if not name.endswith((".nii", ".nii.gz", ".nrrd", ".mha")):
            continue
        base = f
        if base.lower().endswith(".gz"):
            base = os.path.splitext(base)[0]
        base = os.path.splitext(base)[0]
        ids.append(base)
    return sorted(ids)


def _get_full_paths(ct_folder):
    """Lista di path completi ai file di test ASOCA."""
    if not os.path.isdir(ct_folder):
        return []
    paths = []
    for f in os.listdir(ct_folder):
        name = f.lower()
        if not name.endswith((".nii", ".nii.gz", ".nrrd", ".mha")):
            continue
        paths.append(os.path.join(ct_folder, f))
    return sorted(paths)


# --------- ImageCAS: lettura ID --------- #

def _get_cas_ids(cas_folder):
    """
    Cerca file tipo:
      1.img.nii.gz
      1.label.nii.gz
    Ritorna ID come stringhe 'CAS_1', 'CAS_2', ...
    """
    if not os.path.isdir(cas_folder):
        return []
    ids = []
    for f in os.listdir(cas_folder):
        name = f.lower()
        if not name.endswith(".img.nii.gz"):
            continue
        # prendo la parte prima di ".img.nii.gz"
        base = f[: -len(".img.nii.gz")]
        ids.append(f"CAS_{base}")
    # se sono numeri, li ordino numericamente
    def _key(x):
        try:
            return int(x.replace("CAS_", ""))
        except ValueError:
            return x
    return sorted(ids, key=_key)


# ---- train/val: prendo gli ID da CTCA + CAS ----
normal_ids   = _get_case_ids(NORMAL_CT_DIR)       # es. 'Normal_1'
diseased_ids = _get_case_ids(DISEASED_CT_DIR)     # es. 'Diseased_3'
cas_ids      = _get_cas_ids(BASE_DIR_CAS)         # es. 'CAS_1'

all_ids = normal_ids + diseased_ids + cas_ids

# split semplice: 70% train, 15% val, 15% test (interna)
random_seed = 0
random.Random(random_seed).shuffle(all_ids)

n = len(all_ids)
train_ratio = 0.7
val_ratio   = 0.15
n_train = int(train_ratio * n)
n_val   = int(val_ratio * n)

train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train + n_val]

# ---- test ufficiale ASOCA: uso i folder Testset_* ----
test_paths_normal   = _get_full_paths(TEST_NORMAL_CT_DIR)
test_paths_diseased = _get_full_paths(TEST_DISEASED_CT_DIR)
test_ids = test_paths_normal + test_paths_diseased

# Se i folder Testset_* sono vuoti/non esistono, uso il resto come test interno
if len(test_ids) == 0:
    test_ids = all_ids[n_train + n_val :]

# ---- finto "config" ----
config = SimpleNamespace()
config.random = SimpleNamespace(seed=random_seed)

config.dataset = SimpleNamespace()
config.dataset.ASOCA = SimpleNamespace(
    split={
        "train": train_ids,   # es: ["Normal_1", "Diseased_3", "CAS_1", ...]
        "val":   val_ids,
        "test":  test_ids,    # per il test: path completi ai .nrrd (ASOCA) o ID se fallback
    },
    change_image_every_n_samplings=128,
    path_normal_image=NORMAL_CT_DIR,
    path_diseased_image=DISEASED_CT_DIR,
    path_normal_label=NORMAL_LABEL_DIR,
    path_diseased_label=DISEASED_LABEL_DIR,
    path_cas_data=BASE_DIR_CAS,
)

# ---- logging minimo ----
def error(msg: str, return_only: bool = False) -> str:
    text = f"[ERROR] {msg}"
    print(text)
    return text


def debug(msg: str) -> None:
    return  # disattivato


# =========================
# AUGMENTATIONS 2D
# =========================

import torch
from numpy.random import default_rng

try:
    _seed = config.random.seed
except NameError:
    _seed = 0

_rng = default_rng(_seed)


def hflip(img: torch.Tensor, lab: torch.Tensor):
    if _rng.binomial(1, 0.5):
        img = torch.flip(img, dims=[-1])
        lab = torch.flip(lab, dims=[-1])
    return img, lab


def vflip(img: torch.Tensor, lab: torch.Tensor):
    if _rng.binomial(1, 0.5):
        img = torch.flip(img, dims=[-2])
        lab = torch.flip(lab, dims=[-2])
    return img, lab


def rot90(img: torch.Tensor, lab: torch.Tensor):
    if _rng.binomial(1, 0.3):
        k = int(_rng.integers(0, 4))
        if k > 0:
            img = torch.rot90(img, k, dims=(-2, -1))
            lab = torch.rot90(lab, k, dims=(-2, -1))
    return img, lab


def small_shift(img: torch.Tensor, lab: torch.Tensor):
    if _rng.binomial(1, 0.5):
        dx = int(_rng.integers(-10, 10))
        dy = int(_rng.integers(-10, 10))
        img = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))
        lab = torch.roll(lab, shifts=(dy, dx), dims=(-2, -1))
    return img, lab


def light_noise(img: torch.Tensor, lab: torch.Tensor):
    if _rng.binomial(1, 0.5):
        std = 0.01
        noise = torch.normal(
            mean=0.0,
            std=std,
            size=img.shape,
            device=img.device,
        )
        img = img + noise
    return img, lab


def light_intensity_shift(img: torch.Tensor, lab: torch.Tensor):
    if _rng.binomial(1, 0.5):
        shift = float(_rng.normal(0.0, 0.02))
        img = img + shift
    return img, lab


def augment_2d(img: torch.Tensor, lab: torch.Tensor):
    img, lab = hflip(img, lab)
    img, lab = vflip(img, lab)
    img, lab = rot90(img, lab)
    img, lab = small_shift(img, lab)

    img, lab = light_noise(img, lab)
    img, lab = light_intensity_shift(img, lab)

    img = img.clamp(0.0, 1.0)
    return img, lab


# =========================
# PREPROCESSING 3D -> 2D
# =========================

import torch.nn.functional as F
from skimage.measure import label as sk_label, regionprops


def find_heart_bbox_2d(img_t_norm: torch.Tensor,
                       thresh: float = 0.3,
                       min_area: int = 4000,
                       margin: int = 32):
    S, C, H, W = img_t_norm.shape
    assert C == 1

    mid = S // 2
    sl = img_t_norm[mid, 0].cpu().numpy()

    mask = sl > thresh
    lab = sk_label(mask)

    if lab.max() == 0:
        return 0, H, 0, W

    regions = regionprops(lab)
    reg = max(regions, key=lambda r: r.area)
    if reg.area < min_area:
        return 0, H, 0, W

    minr, minc, maxr, maxc = reg.bbox

    minr = max(minr - margin, 0)
    minc = max(minc - margin, 0)
    maxr = min(maxr + margin, H)
    maxc = min(maxc + margin, W)

    h = maxr - minr
    w = maxc - minc
    side = max(h, w)

    cy = (minr + maxr) / 2.0
    cx = (minc + maxc) / 2.0

    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))

    y0 = max(0, min(H - side, y0))
    x0 = max(0, min(W - side, x0))
    y1 = y0 + side
    x1 = x0 + side

    return y0, y1, x0, x1


def crop_and_resize_volume(img_t: torch.Tensor,
                           lab_t: torch.Tensor | None,
                           bbox,
                           out_side: int = 256):
    y0, y1, x0, x1 = bbox

    img_crop = img_t[:, :, y0:y1, x0:x1]
    lab_crop = None
    if lab_t is not None:
        lab_crop = lab_t[:, :, y0:y1, x0:x1]

    if img_crop.shape[-1] != out_side:
        img_crop = F.interpolate(
            img_crop.float(),
            size=(out_side, out_side),
            mode="bilinear",
            align_corners=False,
        ).half()

        if lab_crop is not None:
            lab_crop = F.interpolate(
                lab_crop.float(),
                size=(out_side, out_side),
                mode="nearest",
            ).long()

    return img_crop, lab_crop


# =========================
# DATASET 2D (ASOCA + ImageCAS)
# =========================

from typing import Literal, Tuple
import SimpleITK as sitk


class DatasetMerged_2d(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal['train', 'val', 'test'],
        img_side: int = 512,
        use_cache: bool = True,
        max_cache_size: int = 8,
    ):
        super().__init__()
        self.img_side = img_side
        self.split = split
        self.rng = np.random.default_rng(seed=config.random.seed)

        self.data_ids_pool: list[str] = config.dataset.ASOCA.split[self.split]

        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        self._cache_order: list[str] = []

        match self.split:
            case "train":
                self.change_after_n_sampling = config.dataset.ASOCA.change_image_every_n_samplings
                self.counter = 0
                self.current_patient_id = self.rng.choice(self.data_ids_pool)
                self.image, self.label = self._load_image_label(self.current_patient_id)
            case "val":
                self.current_patient_id = self.data_ids_pool[0]
            case "test":
                self.current_patient_id = self.data_ids_pool[0]
            case _:
                raise ValueError(error(f"'split' must be in ['train', 'val', 'test'], got {self.split}", True))

    # --------- CACHE UTILS --------- #

    def _cache_get(self, patient_id: str):
        if not self.use_cache:
            return None
        return self._cache.get(patient_id, None)

    def _cache_put(self, patient_id: str, img_t: torch.Tensor, lab_t: torch.Tensor | None):
        if not self.use_cache:
            return
        if patient_id in self._cache:
            return
        if len(self._cache_order) >= self.max_cache_size:
            old_id = self._cache_order.pop(0)
            self._cache.pop(old_id, None)
        self._cache[patient_id] = (img_t, lab_t)
        self._cache_order.append(patient_id)

    # --------- LOADING --------- #

    def _load_image_label(self, patient_id: str):
        """
        - train / val:
            * ASOCA: 'Normal_xxx' o 'Diseased_xxx'
            * ImageCAS: 'CAS_1', 'CAS_2', ...
        - test:
            * path completo (ASOCA testset) oppure ID (fallback)

        Ritorna (img_t, lab_t) su CPU.
        """

        cached = self._cache_get(patient_id)
        if cached is not None:
            img_t, lab_t = cached
            return img_t, lab_t

        if self.split == "test" and os.path.sep in patient_id:
            # caso test ASOCA ufficiale: patient_id è un path
            img_path = patient_id
            lab_path = None

        elif self.split in ["train", "val", "test"]:
            # CAS?
            if patient_id.startswith("CAS_"):
                base_id = patient_id.replace("CAS_", "")
                cas_dir = config.dataset.ASOCA.path_cas_data
                img_path = os.path.join(cas_dir, f"{base_id}.img.nii.gz")
                lab_path = os.path.join(cas_dir, f"{base_id}.label.nii.gz")
            else:
                # ASOCA normale/diseased (.nrrd)
                is_normal = ("Normal" in patient_id) or ("normal" in patient_id)

                img_dir = (
                    config.dataset.ASOCA.path_normal_image
                    if is_normal
                    else config.dataset.ASOCA.path_diseased_image
                )
                lab_dir = (
                    config.dataset.ASOCA.path_normal_label
                    if is_normal
                    else config.dataset.ASOCA.path_diseased_label
                )

                img_path = os.path.join(img_dir, patient_id + ".nrrd")
                lab_path = os.path.join(lab_dir, patient_id + ".nrrd")
        else:
            raise ValueError(error(f"Split sconosciuto: {self.split}", True))

        # 3) controlli
        if not os.path.isfile(img_path):
            raise FileNotFoundError(error(f"Immagine non trovata: {img_path}", True))
        if self.split in ["train", "val"] and (lab_path is None or not os.path.isfile(lab_path)):
            raise FileNotFoundError(error(f"Label non trovata per {patient_id}: {lab_path}", True))

        debug(f"Carico immagine da: {img_path}")
        if lab_path is not None:
            debug(f"Carico label da:    {lab_path}")

        # 4) lettura immagine
        sitk_img = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

        # NORMALIZZAZIONE UNICA: finestra cardiaca in HU -> [0,1]
        img_np = np.clip(img_np, -200.0, 800.0)
        img_np = (img_np + 200.0) / 1000.0
        img_np = np.clip(img_np, 0.0, 1.0)

        img_np = img_np.astype(np.float16)
        img_t = torch.from_numpy(img_np).unsqueeze(1)  # (S,1,H,W)

        # label
        if lab_path is not None and os.path.isfile(lab_path):
            sitk_lab = sitk.ReadImage(lab_path)
            lab_np = sitk.GetArrayFromImage(sitk_lab).astype(np.int64)
            lab_t = torch.from_numpy(lab_np).unsqueeze(1)  # (S,1,H,W)
        else:
            lab_t = None

        self._cache_put(patient_id, img_t, lab_t)
        return img_t, lab_t

    # --------- PATCH SAMPLING 2D --------- #

    def get(self, minibatch_size: int = 4, out_side: int = 256):
        imgs = []
        labs = []

        for _ in range(minibatch_size):
            pid = self.rng.choice(self.data_ids_pool)

            img_t, lab_t = self._load_image_label(pid)   # (S,1,H,W)

            bbox = find_heart_bbox_2d(img_t, thresh=0.3, min_area=4000, margin=32)
            img_t_crop, lab_t_crop = crop_and_resize_volume(img_t, lab_t, bbox, out_side=out_side)

            # Handle case where lab_t_crop might be None
            if lab_t_crop is None:
                # Create empty label tensor
                lab_t_crop = torch.zeros_like(img_t_crop).long()
            
            lab_np = lab_t_crop[:, 0].cpu().numpy()
            slice_sums = lab_np.sum(axis=(1, 2))
            pos_slices = np.where(slice_sums > 0)[0]

            if len(pos_slices) > 0 and np.random.rand() < 0.7:
                s = int(np.random.choice(pos_slices))
            else:
                s = int(np.random.randint(0, img_t_crop.shape[0]))

            img_s = img_t_crop[s]
            lab_s = lab_t_crop[s]

            img_s, lab_s = augment_2d(img_s, lab_s)

            imgs.append(img_s)
            labs.append(lab_s)

        img_batch = torch.stack(imgs, dim=0).float()
        lab_batch = torch.stack(labs, dim=0).long()

        return img_batch, lab_batch

    # --------- vecchie API per val/test --------- #

    def _get_train(self, minibatch_size: int = 1):
        img_minibatch = torch.zeros((minibatch_size, 1, self.img_side, self.img_side), dtype=torch.float32)
        lab_minibatch = torch.zeros((minibatch_size, 1, self.img_side, self.img_side), dtype=torch.long)
        for b in range(minibatch_size):
            slice_ = self.rng.integers(low=0, high=self.image.shape[0])
            low_k1 = self.rng.integers(low=0, high=self.image.shape[2]-self.img_side+1)
            low_k2 = self.rng.integers(low=0, high=self.image.shape[3]-self.img_side+1)
            img_ = self.image[slice_:slice_+1, :, low_k1:low_k1+self.img_side, low_k2:low_k2+self.img_side]
            lab_ = self.label[slice_:slice_+1, :, low_k1:low_k1+self.img_side, low_k2:low_k2+self.img_side]
            img_, lab_ = augment_2d(img_, lab_)
            img_minibatch[b] = img_
            lab_minibatch[b] = lab_

        self.counter += minibatch_size
        if self.counter >= self.change_after_n_sampling:
            self.counter = 0
            self.current_patient_id = self.rng.choice(self.data_ids_pool)
            self.image, self.label = self._load_image_label(self.current_patient_id)

        return img_minibatch, lab_minibatch

    def _get_val(self):
        return self._get_val_test()

    def _get_test(self):
        return self._get_val_test()

    def _get_val_test(self) -> dict:
        image, label = self._load_image_label(self.current_patient_id)
        out_id = self.current_patient_id
        if self.split == 'test' and os.path.sep in out_id:
            out_id = os.path.splitext(os.path.basename(out_id))[0]
        
        # Find position in the pool
        try:
            position = self.data_ids_pool.index(self.current_patient_id)
            next_position = (position + 1) % len(self)
            self.current_patient_id = self.data_ids_pool[next_position]
        except ValueError:
            # If current patient ID not found, reset to first
            debug("In DatasetMerged_2d._get_val_test(), patient ID not found in pool, resetting to first.")
            self.current_patient_id = self.data_ids_pool[0]
        
        return {
            "id": f"{self.split}-MERGED-{out_id}",
            "image": image,
            "label": label
        }

    def __len__(self):
        return len(self.data_ids_pool)
