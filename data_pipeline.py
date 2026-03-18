import os
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split

# Config Block
SEED        = 42
IMG_SIZE    = 150
BATCH_SIZE  = 32
TRAIN_PATH  = "/kaggle/input/brain-tumor-mri-dataset/Training"
TEST_PATH   = "/kaggle/input/brain-tumor-mri-dataset/Testing"
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def set_seed(seed: int) -> None:
    """Set seed for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# EDA
def run_eda(train_path: str, class_names: list) -> None:
    """
    Run Exploratory Data Analysis on the training directory.

    Produces:
      - eda_class_distribution.png  (Fig 1 — class balance bar chart)
      - eda_sample_images.png       (Fig 2 — 4x3 image grid)

    Prints dataset statistics and split sizes to the console.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_path = Path(train_path)

    counts    = {}
    all_files = {}
    for cls in class_names:
        cls_dir = train_path / cls
        imgs = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]
        counts[cls]    = len(imgs)
        all_files[cls] = imgs

    total = sum(counts.values())

    print("=" * 52)
    print("  EDA — Brain Tumor MRI Dataset (Training split)")
    print("=" * 52)
    print(f"  Total training images : {total}")
    print()
    print(f"  {'Class':<14} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*14} {'-'*6}  {'-'*6}")
    for cls in class_names:
        pct = counts[cls] / total * 100
        print(f"  {cls:<14} {counts[cls]:>6}  {pct:>5.1f}%")
    print()

    n_val   = 632
    n_train = total - n_val
    n_test  = 703
    print("  Planned split sizes:")
    print(f"    Train : {n_train}")
    print(f"    Val   : {n_val}")
    print(f"    Test  : {n_test}  (from Testing/ folder, not split)")
    print("=" * 52)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax1.bar(
        class_names,
        [counts[c] for c in class_names],
        color=colors,
        edgecolor="white",
        linewidth=0.8
    )

    for bar, cls in zip(bars, class_names):
        h   = bar.get_height()
        pct = h / total * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 15,
            f"{int(h)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Number of Images", fontsize=12)
    ax1.set_title("Class Distribution — Training Set", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, max(counts.values()) * 1.18)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", labelsize=11)
    fig1.tight_layout()
    fig1.savefig("eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved -> eda_class_distribution.png")

    N_SAMPLES = 3
    rng = random.Random(SEED)

    fig2, axes = plt.subplots(
        len(class_names), N_SAMPLES,
        figsize=(N_SAMPLES * 3.2, len(class_names) * 3.2)
    )

    for row_idx, cls in enumerate(class_names):
        samples = rng.sample(all_files[cls], min(N_SAMPLES, len(all_files[cls])))
        for col_idx, img_path in enumerate(samples):
            ax  = axes[row_idx][col_idx]
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")

        axes[row_idx][0].set_yticks([])
        axes[row_idx][0].set_ylabel(
            cls.replace("_", " ").title(),
            fontsize=12, fontweight="bold"
        )

    for col_idx in range(N_SAMPLES):
        axes[0][col_idx].set_title(f"Sample {col_idx + 1}", fontsize=11)

    fig2.suptitle(
        "Sample MRI Images per Class",
        fontsize=15, fontweight="bold", y=1.01
    )
    fig2.tight_layout()
    fig2.savefig("eda_sample_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved -> eda_sample_images.png")


#  Dataset Class & Transforms
class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI images.

    Args:
        samples   : list of (filepath: str | Path, integer_label: int) tuples
        transform : torchvision transform pipeline (or None)
    """

    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_transforms(split: str):
    """
    Return the appropriate torchvision transform pipeline.

    Args:
        split : one of "train", "val", "test"

    Returns:
        torchvision.transforms.Compose

    - ToTensor() scales pixel values to [0, 1].
    - ImageNet Normalize standardizes to zero-mean / unit-std.
    - Augmentations are applied to the training split only.
    """
    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(40),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize,
        ])
    else:  
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            _normalize,
        ])

def to_one_hot(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Convert integer label tensor to one-hot float tensor."""
    return F.one_hot(labels, num_classes).float()

# Data Loading & Splitting
def get_dataloaders(config: dict) -> dict:
    """
    Build and return train / val / test DataLoaders.

    Uses torchvision.datasets.ImageFolder to collect (filepath, label) pairs
    from the Training folder, then applies a stratified 90/10 split.
    The test split is loaded directly from the Testing folder without splitting.

    Args:
        config : dict with keys TRAIN_PATH, TEST_PATH, SEED, BATCH_SIZE, IMG_SIZE

    Returns:
        dict with keys "train", "val", "test" (DataLoaders) and "class_to_idx"
    """
    train_root = config["TRAIN_PATH"]
    test_root  = config["TEST_PATH"]
    seed       = config["SEED"]
    batch_size = config["BATCH_SIZE"]

    folder       = datasets.ImageFolder(root=train_root)
    all_paths    = [s[0] for s in folder.samples]
    all_labels   = [s[1] for s in folder.samples]
    class_to_idx = folder.class_to_idx  # {"glioma":0, "meningioma":1, ...}

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=632,
        stratify=all_labels,
        random_state=seed,
    )

    test_folder  = datasets.ImageFolder(root=test_root)
    test_samples = [(s[0], s[1]) for s in test_folder.samples]

    train_ds = BrainTumorDataset(
        list(zip(train_paths, train_labels)),
        transform=get_transforms("train")
    )
    val_ds = BrainTumorDataset(
        list(zip(val_paths, val_labels)),
        transform=get_transforms("val")
    )
    test_ds = BrainTumorDataset(
        test_samples,
        transform=get_transforms("test")
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"  DataLoaders ready — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    return {
        "train":        train_loader,
        "val":          val_loader,
        "test":         test_loader,
        "class_to_idx": class_to_idx,
    }

# Split Integrity Check
def verify_splits(dataloaders: dict) -> None:
    """
    Verify that train, val, and test splits have no filepath overlap,
    and print per-class image counts for each split.

    Raises:
        AssertionError : if any filepath appears in more than one split.
    """

    def _extract(loader):
        return [(Path(fp), lbl) for fp, lbl in loader.dataset.samples]

    train_items = _extract(dataloaders["train"])
    val_items   = _extract(dataloaders["val"])
    test_items  = _extract(dataloaders["test"])

    train_paths = {fp for fp, _ in train_items}
    val_paths   = {fp for fp, _ in val_items}
    test_paths  = {fp for fp, _ in test_items}

    # Overlap checks
    tv_overlap = train_paths & val_paths
    tt_overlap = train_paths & test_paths
    vt_overlap = val_paths   & test_paths

    assert len(tv_overlap) == 0, (
        f"DATA LEAKAGE: {len(tv_overlap)} file(s) shared between train and val:\n"
        + "\n".join(str(p) for p in list(tv_overlap)[:5])
    )
    assert len(tt_overlap) == 0, (
        f"DATA LEAKAGE: {len(tt_overlap)} file(s) shared between train and test:\n"
        + "\n".join(str(p) for p in list(tt_overlap)[:5])
    )
    assert len(vt_overlap) == 0, (
        f"DATA LEAKAGE: {len(vt_overlap)} file(s) shared between val and test:\n"
        + "\n".join(str(p) for p in list(vt_overlap)[:5])
    )

    print("   No data leakage")

    # Per-class distribution table 
    idx_to_class = {v: k for k, v in dataloaders["class_to_idx"].items()}

    def _class_counts(items):
        c = Counter(lbl for _, lbl in items)
        return {idx_to_class.get(k, k): v for k, v in sorted(c.items())}

    tc = _class_counts(train_items)
    vc = _class_counts(val_items)
    ec = _class_counts(test_items)

    print()
    print(f"  {'Class':<14} {'Train':>6}  {'Val':>6}  {'Test':>6}")
    print(f"  {'-'*14} {'-'*6}  {'-'*6}  {'-'*6}")
    for cls in sorted(set(tc) | set(vc) | set(ec)):
        print(f"  {cls:<14} {tc.get(cls,0):>6}  {vc.get(cls,0):>6}  {ec.get(cls,0):>6}")
    print()

# Main Entry Point
if __name__ == "__main__":
    set_seed(SEED)

    run_eda(TRAIN_PATH, CLASS_NAMES)

    config = {
        "TRAIN_PATH": TRAIN_PATH,
        "TEST_PATH":  TEST_PATH,
        "SEED":       SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "IMG_SIZE":   IMG_SIZE,
    }

    dataloaders = get_dataloaders(config)
    verify_splits(dataloaders)

    print("Pipeline ready. Outputs: eda_class_distribution.png, eda_sample_images.png")
