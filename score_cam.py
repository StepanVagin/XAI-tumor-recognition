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

from pytorch_grad_cam import GradCAM as LibraryGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =============================================================================
# Config Block
# =============================================================================

CONFIG = {
    "SEED": 42,
    "IMG_SIZE": 150,
    "NUM_CLASSES": 4,
    "CLASS_NAMES": ["glioma", "meningioma", "notumor", "pituitary"],
    "MODEL_PATH": "/kaggle/input/datasets/fuller1te/vgg16-baseline/vgg16_attention.pth",
    "DATA_PATH": "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset",
}

def set_seed(seed: int) -> None:
    """Ensure reproducibility for random image sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# VGG16 + Attention
# =============================================================================

class PaperChannelAttentionHead(nn.Module):
    """
    Channel-level attention after the last conv feature map, as in the paper.

    Given z ∈ R^{B×C×H×W}:
      1. v_i = (1/HW) Σ_{j,k} z_{b,i,j,k}  (global average pooling per channel)
      2. α = softmax(w) with trainable w ∈ R^C (same α for all batch elements)
      3. v_att = v ⊙ α
      4. logits = W v_att + b  (Linear; softmax is applied in CrossEntropyLoss)
    """

    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.attention_logits = nn.Parameter(torch.zeros(num_channels))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, H, W)
        v = z.mean(dim=(2, 3))
        alpha = F.softmax(self.attention_logits, dim=0)
        v_att = v * alpha
        return self.fc(v_att)

class VGG16PaperAttention(nn.Module):
    """VGG16 conv features + paper channel-attention head (no VGG avgpool/classifier)."""
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.vgg16(pretrained=False)
        self.features = backbone.features
        # 512 is the number of channels in the last VGG16 block
        self.head = PaperChannelAttentionHead(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.head(z)

def build_model(num_classes: int = 4):
    """Instantiates the Attention-enhanced architecture."""
    
    return VGG16PaperAttention(num_classes=num_classes)

def load_checkpoint(filepath: str, device: torch.device):
    """Initialize model and load trained weights."""
    
    model = build_model(num_classes=CONFIG["NUM_CLASSES"])
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Model successfully loaded from '{filepath}'")

    return model

# =============================================================================
# Preprocessing Pipeline
# =============================================================================

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

# =============================================================================
# Score-CAM Implementation
# =============================================================================

class ScoreCAM:
    def __init__(self, model, target_layer, batch_size=32):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.activations = None 
        self.hook_handle = self.target_layer.register_forward_hook(self._forward_hook)
        
        # Normalization constants for re-normalizing masked images
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def remove_hook(self):
        self.hook_handle.remove()

    def __call__(self, input_tensor, target_class_idx):
        device = input_tensor.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        
        # 1. Trigger Hook to get activations
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        maps = self.activations[0] 
        num_channels = maps.shape[0]
        B, C, H, W = input_tensor.shape

        # 2. Process Masks
        upsampled_maps = F.interpolate(maps.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)
        max_v = upsampled_maps.view(num_channels, -1).max(dim=1)[0].view(num_channels, 1, 1, 1)
        min_v = upsampled_maps.view(num_channels, -1).min(dim=1)[0].view(num_channels, 1, 1, 1)
        normalized_masks = (upsampled_maps - min_v) / (max_v - min_v + 1e-8)

        # 3. De-normalize input before masking
        raw_input = input_tensor * self.std + self.mean

        # 4. Scoring Engine
        scores = []
        for i in range(0, num_channels, self.batch_size):
            batch_masks = normalized_masks[i : i + self.batch_size]
            
            # Mask the RAW image (0 stays black)
            masked_raw = raw_input * batch_masks
            
            # RE-NORMALIZE for the model
            masked_input = (masked_raw - self.mean) / self.std
            
            with torch.no_grad():
                output = self.model(masked_input)
                probs = F.softmax(output, dim=1)
                scores.append(probs[:, target_class_idx])

        scores = torch.cat(scores)

        # 5. Weighted Summation
        cam = (upsampled_maps * scores.view(num_channels, 1, 1, 1)).sum(dim=0, keepdim=True)
        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().squeeze().numpy()

# =============================================================================
# Comparison Function
# =============================================================================

def compare_xai_methods(model, image_path, target_layer, device):
    """
    Generates side-by-side comparison between Grad-CAM (library) 
    and Score-CAM (manual implementation).
    """
    # 1. Setup
    transform = get_inference_transform()
    raw_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    
    # 2. Get Model Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    pred_label = CONFIG["CLASS_NAMES"][pred_idx]
    true_label = image_path.split('/')[-2]

    # 3. Generate Grad-CAM (Baseline Logic)
    grad_cam_engine = LibraryGradCAM(model=model, target_layers=[target_layer])
    grad_cam_mask = grad_cam_engine(input_tensor=input_tensor, 
                                    targets=[ClassifierOutputTarget(pred_idx)])[0, :]

    # 4. Generate Score-CAM (Our New Logic)
    score_cam_engine = ScoreCAM(model=model, target_layer=target_layer)
    score_cam_mask = score_cam_engine(input_tensor=input_tensor, target_class_idx=pred_idx)
    score_cam_engine.remove_hook()

    # 5. Prepare for Visualization
    bg_img = np.array(raw_img.resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))).astype(np.float32) / 255.0
    
    grad_viz = show_cam_on_image(bg_img, grad_cam_mask, use_rgb=True)
    score_viz = show_cam_on_image(bg_img, score_cam_mask, use_rgb=True)

    # 6. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input MRI\nTarget: {true_label}", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(grad_viz)
    axes[1].set_title(f"Grad-CAM (Baseline)\nPred: {pred_label} ({confidence*100:.1f}%)", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(score_viz)
    axes[2].set_title(f"Score-CAM (Advanced)\nPred: {pred_label} ({confidence*100:.1f}%)", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def run_attention_visualization(model, device):
    """
    Main loop: picks a random test image and displays Grad-CAM vs Score-CAM.
    """
    target_layer = model.features[28]
    test_files = glob.glob(os.path.join(CONFIG["DATA_PATH"], "Testing", "*", "*.jpg"))
    sample_path = random.choice(test_files)
    compare_xai_methods(model, sample_path, target_layer, device)


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    model = load_checkpoint(CONFIG["MODEL_PATH"], device)
    run_attention_visualization(model, device)
