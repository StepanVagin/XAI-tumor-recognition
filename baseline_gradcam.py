import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


CONFIG = {
    "SEED": 42,
    "IMG_SIZE": 150,
    "NUM_CLASSES": 4,
    "CLASS_NAMES": ["glioma", "meningioma", "notumor", "pituitary"],
    "MODEL_PATH": "/kaggle/input/datasets/fuller1te/vgg16-baseline/vgg16_baseline.pth",
    "DATA_PATH": "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset",
}

def set_seed(seed: int) -> None:
    """Ensure reproducibility for random image sampling."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(num_classes: int = 4):
    """Recreate the modified VGG16 architecture used in training."""

    model = models.vgg16(pretrained=False)
    
    # recreate the classifier head exactly as per baseline_vgg16.py
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

def load_checkpoint(filepath: str, device: torch.device):
    """Initialize model and load trained weights."""

    model = build_model(num_classes=CONFIG["NUM_CLASSES"])
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Model successfully loaded from '{filepath}'")

    return model


def get_inference_transform():
    """Returns the normalization and resizing pipeline used during baseline training."""

    return transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def run_baseline_visualization(model, device):
    """
    Main loop to sample images from all classes and generate Grad-CAM overlays.
    Saves the final grid as 'baseline_xai_samples.png'.
    """
    transform = get_inference_transform()
    
    # target the last convolutional layer in VGG16 architecture
    target_layers = [model.features[28]] 
    cam_engine = GradCAM(model=model, target_layers=target_layers)

    fig, axes = plt.subplots(4, 2, figsize=(14, 24))
    for i, cls_name in enumerate(CONFIG["CLASS_NAMES"]):
        folder_path = os.path.join(CONFIG["DATA_PATH"], "Testing", cls_name)
        img_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        sample_path = random.choice(img_paths)
        raw_img = Image.open(sample_path).convert("RGB")
        
        input_tensor = transform(raw_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        pred_label = CONFIG["CLASS_NAMES"][pred_idx]

        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)[0, :]
        
        img_np = np.array(raw_img.resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])))
        img_np = img_np.astype(np.float32) / 255.0
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"Target: {cls_name}", fontsize=12, fontweight='bold')
        axes[i, 0].axis("off")
        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"XAI: Pred {pred_label} ({confidence*100:.2f}%)", fontsize=12)
        axes[i, 1].axis("off")

    plt.suptitle("Baseline XAI (Grad-CAM) Interpretation", fontsize=20, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig("baseline_xai_samples.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved to baseline_xai_samples.png")


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running XAI Baseline Sprint on '{device}'")
    
    model = load_checkpoint(CONFIG["MODEL_PATH"], device)
    
    run_baseline_visualization(model, device)
