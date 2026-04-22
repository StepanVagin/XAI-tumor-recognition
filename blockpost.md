# Brain Tumor MRI Classification with Score-CAM Explainability

**Team:** Lana Ermolaeva · Stepan Vagin · Ramil Shakirzyanov  
**Course:** XAI — B23-AI-02, Innopolis University  
**Date:** April 22, 2026

---

## 1. Domain of Application

**Healthcare Diagnostics — Multiclass Brain Tumor Detection from MRI**

Brain tumors are classified from MRI scans into four categories:

| Class | Description |
|---|---|
| Glioma | Aggressive tumor arising from glial cells |
| Meningioma | Tumor growing from the meninges (brain lining) |
| Pituitary | Tumor in the pituitary gland |
| No Tumor | Healthy brain scan |

Dataset: **Kaggle Brain Tumor MRI Dataset** (Masoud Nickparvar) — 7,023 labeled MRI scans.

---

## 2. Motivation

Brain tumor diagnosis from MRI is a high-stakes, time-critical task. Radiologists worldwide face increasing workloads, and misdiagnosis carries severe consequences for patients. Deep learning models can match or exceed human-level performance in classification accuracy — but in clinical practice, **a prediction without a justification is not trusted**.

Clinicians need to know: *"What in this scan made the model say glioma?"*

Without this, even a 99%-accurate model cannot be deployed. The field of Explainable AI (XAI) exists to answer this question.

---

## 3. The Real-World Problem

Black-box AI models make life-or-death medical decisions without any interpretable justification. A tumor classification system that cannot show *where* it is looking:

- Cannot be audited for bias (e.g., responding to skull shape instead of the tumor)
- Cannot be trusted by clinicians who are legally responsible for diagnosis
- Gives no feedback to identify systematic failure modes
- May be confidently wrong — our baseline found **98–99% confidence misclassifications**

The goal: build a system that classifies tumors accurately *and* generates faithful explanations of each decision.

---

## 4. The Technical Problem — Inside the Black Box

A deep convolutional neural network (DCNN) like VGG16 transforms a 150×150 MRI image through 13 convolutional layers, producing a 512×9×9 feature tensor before classification. Each of the 512 channels learns a different visual pattern. The final class score is a nonlinear function of millions of parameters.

**The black box problem:** Given only the input image and the output class, there is no direct way to know which pixels drove the decision. The internal activations encode information in a distributed, high-dimensional space that is not human-interpretable.

XAI methods like Grad-CAM and Score-CAM project this internal representation back onto the image as a heatmap — a saliency map showing which regions were most important for the predicted class.

---

## 5. ML Model Being Explained

### Architecture: VGG16 + SoftMax Channel Attention

We use a **VGG16 backbone** extended with a custom **channel-level attention head**, implemented from scratch based on Aiya et al. [1].

```
Input MRI (150×150×3)
     ↓
VGG16 Feature Extractor (13 conv layers, 5 blocks)
     ↓
Feature Map z ∈ ℝ^{B×512×9×9}
     ↓
Global Average Pooling  →  v ∈ ℝ^{512}
     ↓
SoftMax Channel Attention  →  v_att ∈ ℝ^{512}
     ↓
Linear (512 → 4)  →  logits
     ↓
Predicted Class (Glioma / Meningioma / No Tumor / Pituitary)
```

**Transfer learning strategy:** Convolutional blocks 1–3 (layers 0–19) are frozen — they retain ImageNet low-level feature detectors (edges, textures) which transfer well to MRI. Blocks 4–5 (layers 20+) are fine-tuned on the MRI data.

### Attention Mechanism (from paper)

Given feature map $z \in \mathbb{R}^{B \times C \times H \times W}$:

**Step 1 — Global Average Pooling per channel:**
$$v_i = \frac{1}{HW} \sum_{j,k} z_{b,i,j,k}$$

**Step 2 — SoftMax attention weights** (learned parameter $w \in \mathbb{R}^C$):
$$\alpha_i = \frac{e^{w_i}}{\sum_{j=1}^{C} e^{w_j}}$$

**Step 3 — Weighted feature vector:**
$$v_{\text{att}} = v \odot \alpha$$

**Step 4 — Classification:**
$$\text{logits} = W \cdot v_{\text{att}} + b$$

The attention weights $\alpha$ are shared across all inputs in a batch — they are learned channel importance weights, not input-dependent. This is exactly the formulation from Aiya et al. [1].

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 150 × 150 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Epochs | 10 |
| Batch size | 32 |
| Loss | Cross-Entropy |
| Augmentation | Rotation ±40°, translation ±20%, scale 0.8–1.2×, horizontal flip |

---

## 6. Grad-CAM — The Baseline XAI Method

Gradient-weighted Class Activation Mapping (Grad-CAM) [3] uses gradients of the target class score flowing back to the last convolutional layer to produce a coarse saliency map.

### How It Works

**Step 1 — Compute gradient of target class score $y^c$ w.r.t. feature map $A^k$:**
$$\frac{\partial y^c}{\partial A^k_{ij}}$$

**Step 2 — Global average pool the gradients to get channel importance weights:**
$$\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

**Step 3 — Weighted combination of feature maps, followed by ReLU:**
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha^c_k \cdot A^k\right)$$

The ReLU retains only activations that positively influence the class score. The resulting 9×9 map is upsampled to 150×150 and overlaid on the image.

**Implementation:** Applied to `features[28]` — the last ReLU activation in the VGG16 feature extractor.

---

## 7. Grad-CAM Limitations (That Score-CAM Fixes)

### 1. Gradient Noise
Gradients at the last convolutional layer can be noisy or saturated, especially when activations are near the boundary of non-linearities. Small perturbations in input can cause large gradient fluctuations — the saliency map is unstable.

### 2. High-Confidence Misclassifications with Misleading Maps
Our baseline found the model predicting glioma as meningioma with **99.67% confidence** and pituitary as meningioma with **98.09% confidence**. In both cases, the Grad-CAM map activated over a plausible-looking region — meaning the explanation looked reasonable but was wrong. Gradients cannot distinguish *"this activation causes the correct class"* from *"this activation causes the wrong class"*.

### 3. Map Blurring from Global Averaging
Averaging gradients spatially collapses spatial structure. Two channels with opposing gradients at the same location cancel each other out, smoothing the map and obscuring the actual discriminative region.

### 4. Gradient Dependence
Grad-CAM is not applicable to architectures without explicit gradients (e.g., non-differentiable components). More importantly, gradients measure local sensitivity — not the actual contribution of a region to the output score.

---

## 8. Score-CAM — Architecture and Formulas

Score-CAM [2] is a **gradient-free** method. Instead of using gradients as proxy for importance, it directly measures how much each channel's activation mask increases the model's confidence for the target class.

### Algorithm

**Step 1 — Extract activation maps from the target layer:**
$$A^k_l \in \mathbb{R}^{H_f \times W_f}, \quad k = 1, \ldots, C$$

**Step 2 — Upsample and normalize each map to [0, 1] to create masks:**
$$M^k = \frac{A^k_l - \min(A^k_l)}{\max(A^k_l) - \min(A^k_l) + \epsilon}$$

**Step 3 — Mask the input image with each channel's normalized map:**
$$x_k = x_{\text{raw}} \odot M^k_{\uparrow}$$
where $M^k_{\uparrow}$ is bilinearly upsampled to input resolution (150×150).

**Step 4 — Score each channel by how much its mask increases target class confidence above a black-image baseline:**
$$s^c_k = f^c(x_k) - f^c(x_{\text{baseline}})$$

**Step 5 — Clamp and weighted sum:**
$$L^c_{\text{Score-CAM}} = \sum_k \max(s^c_k, 0) \cdot M^k_{\uparrow}$$

No gradients are computed anywhere. Importance is measured by actual confidence change under masking.

---

## 9. What We Changed in Score-CAM

Our implementation differs from the original paper in three key ways, each motivated by empirical observation during development.

### 9.1 Per-Channel Clamping Before Summation
**Original paper:** applies ReLU *after* the weighted sum $\text{ReLU}(\sum_k s^c_k \cdot M^k)$.  
**Our version:** clamps each score to zero *before* summation: `scores = torch.clamp(scores, min=0.0)`.

**Why:** Channels with negative scores (i.e., their mask reduces confidence below baseline) drag the entire weighted sum downward. When this sum is then passed through ReLU, large negative contributions can blank out the map entirely. Clamping per-channel prevents this: channels that hurt confidence contribute nothing, while positive channels are unaffected.

### 9.2 Full-Resolution Upsampling Before Weighting
**Original paper:** weights the 9×9 feature maps first, then upsample the final sum.  
**Our version:** upsamples all 512 masks to 150×150 first, then weights and sums.

**Why:** Weighting low-resolution maps and upsampling at the end collapses fine spatial structure before it can be preserved. Upsampling first retains channel-level spatial detail in the final heatmap.

### 9.3 Black-Image Baseline Subtraction
We compute the model's confidence for the target class on a fully-black image ($x_{\text{baseline}} = \mathbf{0}$) and subtract it from every channel's score:
$$s^c_k = f^c(x_k) - f^c(x_{\text{baseline}})$$

This measures the *net* confidence gain attributable to each channel mask, rather than raw confidence. Without this, channels that produce moderate confidence even on a masked image would be over-weighted.

### 9.4 Batched Scoring Engine
Running 512 forward passes one-by-one is slow. We batch channels in groups of 32 for efficient GPU utilization, processing the entire 512-channel set in 16 forward passes.

---

## 10. Results

### Classification Metrics (VGG16 + Attention, Test Set — 703 images)

| Class | F1-Score |
|---|---|
| Glioma | 0.854 |
| Meningioma | 0.887 |
| No Tumor | 0.945 |
| Pituitary | **0.971** |
| **Macro Average** | **0.914** |
| **Macro AUC-ROC** | **0.987** |

### Score-CAM vs. Grad-CAM — Qualitative Improvements

| Property | Grad-CAM | Score-CAM |
|---|---|---|
| Gradient-free | No | **Yes** |
| Localization sharpness | Coarse / diffuse | **Tighter, tumor-focused** |
| Stability across runs | Variable (gradient noise) | **Deterministic** |
| Handles high-confidence errors | Misleading maps | **Better discrimination** |
| Computation | Single backward pass | 16 batched forward passes |

Score-CAM consistently produces heatmaps that concentrate on the tumor mass rather than surrounding tissue, reducing the risk of a saliency map that looks plausible but points to the wrong region.

---

## 11. Example Showcase

Visual outputs are available in `outputs/`:

| File | Description |
|---|---|
| `outputs/baseline_xai_samples.png` | Grad-CAM overlays for one sample per class (target vs. prediction) |
| `outputs/test_prediction.png` | Grad-CAM on 5 test samples including misclassifications with confidence scores |
| `outputs/training_curves.png` | Train/val loss and accuracy over 10 epochs |
| `outputs/eda_class_distribution.png` | Class balance bar chart for the training set |
| `outputs/eda_sample_images.png` | Sample MRI grid — 3 images per class |

`score_cam.py` generates a 3-panel comparison (original MRI / Grad-CAM / Score-CAM) for any random test image at runtime.

---

## 12. Conclusions

1. **A fine-tuned VGG16 with channel attention achieves strong classification performance** — macro F1 of 0.914 and AUC-ROC of 0.987 on 703 held-out MRI scans, matching the state-of-the-art baseline from Aiya et al.

2. **Grad-CAM is sufficient for correct, high-confidence predictions** but becomes unreliable when the model is wrong. High-confidence misclassifications (98–99%) produced Grad-CAM maps that looked anatomically plausible — meaning the explanation was as wrong as the prediction, with no visual warning.

3. **Score-CAM addresses the root limitation.** By replacing gradients with forward-pass confidence measurements, Score-CAM generates maps grounded in actual model behavior rather than local sensitivity. The result is tighter, more trustworthy localization.

4. **Explainability is not an add-on.** The error analysis only became possible because the XAI layer existed. Without Grad-CAM, the macro F1 would have suggested a near-production-ready model. With it, two critical failure modes — glioma/meningioma confusion at 99% confidence — became visible and diagnosable.

---

## 13. What We Learned

**Data integrity must be enforced explicitly.** Silent data leakage from `ImageFolder`-based pipelines is easy to introduce and hard to detect. We built `verify_splits()` with explicit filepath-overlap assertions from the start, not as an afterthought.

**Partial freezing is the right fine-tuning strategy for small medical datasets.** Freezing the first three VGG16 blocks (0–19) and fine-tuning blocks 4–5 reduced trainable parameters while keeping low-level ImageNet features that transfer well to grayscale MRI. The result: 0.914 macro F1 in 10 epochs on ~6K training images.

**Gradient noise is not a theoretical concern — it is visible.** Error analysis with Grad-CAM showed two misclassifications where the saliency map activated over a perfectly plausible-looking region. The gradient-based explanation gave no indication that the prediction was wrong. This motivated Score-CAM as a necessity, not a curiosity.

**Early per-channel clamping changes the output meaningfully.** Implementing Score-CAM revealed that the paper's formulation (ReLU after summing) can blank the entire heatmap when many channels have negative scores. Moving the clamp to before the sum is a small code change with a large visual difference.

---

## 14. Personal Goals Reached

| Goal | Status |
|---|---|
| Implement a paper's attention mechanism from scratch | Done — `PaperChannelAttentionHead` in `baseline_vgg16.py` |
| Fine-tune a DCNN on a real medical dataset | Done — VGG16 with ImageNet transfer learning |
| Implement a gradient-based XAI method | Done — Grad-CAM via `pytorch-grad-cam` |
| Implement a gradient-free XAI method from scratch | Done — custom `ScoreCAM` class in `score_cam.py` |
| Reproduce a research paper's architecture | Done — channel attention matches Aiya et al. exactly |
| Identify and explain a real model failure mode | Done — high-confidence glioma/meningioma confusion |
| Produce a reproducible, public PyTorch codebase | Done |

---

## 15. Links

- **GitHub Repository:** https://github.com/PLACEHOLDER/PLACEHOLDER *(to be updated)*
- **Dataset:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Kaggle Notebook (training):** *(link to be added)*

---

## 16. References

[1] A. J. Aiya, N. Wani, M. Ramani, A. Kumar, S. Pant, K. Kotecha, A. Kulkarni, and A. Al-Danakh, "Optimized deep learning for brain tumor detection: a hybrid approach with attention mechanisms and clinical explainability," *Sci. Rep.*, vol. 15, no. 1, Art. no. 31386, Aug. 2025, doi: 10.1038/s41598-025-04591-3.

[2] H. Wang, Z. Wang, M. Du, F. Yang, Z. Zhang, S. Ding, P. Mardziel, and X. Hu, "Score-CAM: Score-weighted visual explanations for convolutional neural networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2020, pp. 111–119.

[3] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.

[4] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2015.

[5] M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
