"""
baseline_vgg16.py — VGG16 Baseline for Brain Tumor MRI Classification
Team: Ermolaeva, Vagin, Shakirzyanov (B23-AI-02, Innopolis University)

Kaggle dataset : masoudnickparvar/brain-tumor-mri-dataset
Run via        : !python baseline_vgg16.py
"""

import os
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report


# =============================================================================
# Config Block
# =============================================================================

CONFIG = {
    "SEED":        42,
    "IMG_SIZE":    150,
    "BATCH_SIZE":  32,
    "LR":          1e-4,
    "EPOCHS":      10,
    "NUM_CLASSES": 4,
    "TRAIN_PATH":  "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Training",
    "TEST_PATH":   "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing",
    "CLASS_NAMES": ["glioma", "meningioma", "notumor", "pituitary"],
}


def set_seed(seed: int) -> None:
    """Set seed for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset Class & Transforms
# =============================================================================

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

    Notes:
        - ToTensor() scales pixel values to [0, 1].
        - ImageNet Normalize standardizes to zero-mean / unit-std.
        - Augmentations are applied ONLY to the training split.
    """
    img_size   = CONFIG["IMG_SIZE"]
    _normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(40),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize,
        ])
    else:  # "val" or "test" — no augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            _normalize,
        ])


def to_one_hot(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Convert integer label tensor to one-hot float tensor."""
    return F.one_hot(labels, num_classes).float()


# =============================================================================
# Data Loading & Splitting
# =============================================================================

def get_dataloaders(config: dict) -> dict:
    """
    Build and return train / val / test DataLoaders.

    Uses torchvision.datasets.ImageFolder to collect (filepath, label) pairs
    from the Training folder, then applies a stratified split reserving 632
    samples for validation. The test split is loaded directly from the Testing
    folder without further splitting.

    Expected split sizes:
        Train : 5,688   Val : 632   Test : 703

    Args:
        config : dict with keys TRAIN_PATH, TEST_PATH, SEED, BATCH_SIZE, IMG_SIZE

    Returns:
        dict with keys "train", "val", "test" (DataLoaders) and "class_to_idx"
    """
    train_root = config["TRAIN_PATH"]
    test_root  = config["TEST_PATH"]
    seed       = config["SEED"]
    batch_size = config["BATCH_SIZE"]

    # ── Collect (filepath, label) pairs via ImageFolder ──────────────────────
    folder       = datasets.ImageFolder(root=train_root)
    all_paths    = [s[0] for s in folder.samples]
    all_labels   = [s[1] for s in folder.samples]
    class_to_idx = folder.class_to_idx   # e.g. {"glioma": 0, "meningioma": 1, ...}

    # ── Stratified train / val split ─────────────────────────────────────────
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=632,
        stratify=all_labels,
        random_state=seed,
    )

    # ── Test set (no splitting) ───────────────────────────────────────────────
    test_folder  = datasets.ImageFolder(root=test_root)
    test_samples = [(s[0], s[1]) for s in test_folder.samples]

    # ── Wrap in BrainTumorDataset ─────────────────────────────────────────────
    train_ds = BrainTumorDataset(
        list(zip(train_paths, train_labels)),
        transform=get_transforms("train"),
    )
    val_ds = BrainTumorDataset(
        list(zip(val_paths, val_labels)),
        transform=get_transforms("val"),
    )
    test_ds = BrainTumorDataset(
        test_samples,
        transform=get_transforms("test"),
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"  DataLoaders ready — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    return {
        "train":        train_loader,
        "val":          val_loader,
        "test":         test_loader,
        "class_to_idx": class_to_idx,
    }


# =============================================================================
# Split Integrity Check
# =============================================================================

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

    # ── Overlap checks ────────────────────────────────────────────────────────
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

    print("  ✓ No data leakage detected")

    # ── Per-class distribution table ──────────────────────────────────────────
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
        print(f"  {cls:<14} {tc.get(cls, 0):>6}  {vc.get(cls, 0):>6}  {ec.get(cls, 0):>6}")
    print()


# =============================================================================
# Model Definition
# =============================================================================

def build_model(num_classes: int = 4):
    """
    Build and return a modified VGG16 model for brain tumor classification.

    - Loads pretrained VGG16 weights.
    - Freezes conv blocks 1–3 (features[0]–features[19]).
    - Leaves conv blocks 4–5 (features[20]+) unfrozen.
    - Replaces the classifier head with a 3-layer MLP.

    Args:
        num_classes : number of output classes (default: 4)

    Returns:
        torch.nn.Module — modified VGG16

    Usage (standalone import):
        from baseline_vgg16 import build_model
        model = build_model()
    """
    model = models.vgg16(pretrained=True)

    # Freeze conv blocks 1–3 (features indices 0–19 inclusive)
    for i in range(20):
        for param in model.features[i].parameters():
            param.requires_grad = False

    # Leave conv blocks 4–5 (features[20]+) unfrozen
    for i in range(20, len(model.features)):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )

    return model


# =============================================================================
# Training
# =============================================================================

def train(model, dataloaders: dict, config: dict, device) -> dict:
    """
    Train the model and save the best checkpoint by validation loss.

    Args:
        model       : PyTorch model to train
        dataloaders : dict returned by get_dataloaders() — must have "train" and "val"
        config      : CONFIG dictionary
        device      : torch.device

    Returns:
        history (dict) with lists: train_loss, val_loss, train_acc, val_acc
    """
    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]
    class_names  = config["CLASS_NAMES"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["LR"],
    )

    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, config["EPOCHS"] + 1):

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc  = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch [{epoch:02d}/{config['EPOCHS']}]  "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  |  "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

        # ── Save best checkpoint by val loss ──────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names":      class_names,
                    "class_to_idx":     {
                        "glioma":     0,
                        "meningioma": 1,
                        "notumor":    2,
                        "pituitary":  3,
                    },
                },
                "vgg16_baseline.pth",
            )
            print(f"   Best checkpoint saved (val_loss={best_val_loss:.4f})")

    return history


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, test_loader, device) -> dict:
    """
    Evaluate the model on the test set and save metrics to metrics.json.

    Computes:
      - Per-class and macro F1-score
      - Macro AUC-ROC (one-vs-rest, softmax probabilities)
      - Full sklearn classification_report (printed to stdout)

    Args:
        model       : trained PyTorch model (best checkpoint already loaded)
        test_loader : DataLoader for the test split
        device      : torch.device

    Returns:
        dict with f1_macro, f1_per_class, auc_roc_macro
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    class_names = CONFIG["CLASS_NAMES"]

    # F1
    f1_macro        = f1_score(all_labels, all_preds, average="macro")
    f1_per_class_arr = f1_score(all_labels, all_preds, average=None)
    f1_per_class    = {
        name: float(score)
        for name, score in zip(class_names, f1_per_class_arr)
    }

    # AUC-ROC (one-vs-rest, macro)
    auc_roc_macro = roc_auc_score(
        all_labels,
        all_probs,
        multi_class="ovr",
        average="macro",
    )

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"  F1 Macro  : {f1_macro:.4f}")
    print(f"  AUC-ROC   : {auc_roc_macro:.4f}")

    metrics = {
        "f1_macro":     float(f1_macro),
        "f1_per_class": f1_per_class,
        "auc_roc_macro": float(auc_roc_macro),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("  ✓ Metrics saved to metrics.json")
    return metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_history(history: dict) -> None:
    """
    Plot train vs. val loss and accuracy curves, saved to training_curves.png.
    Does not call plt.show().

    Args:
        history : dict with keys train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o")
    axes[0].set_title("Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="o")
    axes[1].set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(" Training curves saved to training_curves.png")


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    dataloaders = get_dataloaders(CONFIG)
    verify_splits(dataloaders)

    model = build_model(num_classes=CONFIG["NUM_CLASSES"]).to(device)

    # TODO: Score-CAM hook point — attach hooks here before or after training

    history = train(model, dataloaders, CONFIG, device)

    checkpoint = torch.load("vgg16_baseline.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate(model, dataloaders["test"], device)
    plot_history(history)
