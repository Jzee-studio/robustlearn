# Spectral Reliance Imbalance in Adversarial Training — Landscape Survey

## Research Direction
Adversarial training (AT) suppresses high-frequency perturbations, causing the network to concentrate discriminative power on low-frequency semantic components. This creates a "spectral reliance imbalance" (SRI) that degrades OOD generalization to low-frequency distribution shifts (contrast, brightness, fog, etc.). Can we mitigate SRI via a method compatible with RiFT-based fine-tuning?

## Core Phenomenon

**Spectral Reliance Imbalance (SRI)**: AT models learn to disregard high-frequency features because adversarial perturbations concentrate in high frequencies. The model shifts to relying almost exclusively on low-frequency components (shape, coarse structure). This creates a brittle dependency — the model performs well on clean and adversarial examples but fails catastrophically on low-frequency OOD corruptions.

**Key empirical facts:**
1. AT-trained models show markedly lower softmax confidence on high-frequency components vs. standard models
2. When frequency components are progressively retained via inverse FFT, AT models quickly saturate — high frequencies barely affect predictions
3. AT smooths convolutional kernels, naturally attenuating high-frequency responses throughout the network
4. AT improves robustness to high-frequency corruptions (noise) while REDUCING robustness to low-frequency corruptions (fog, contrast)
5. Certified defenses (randomized smoothing) show 35-75% performance drops on low-frequency OOD corruptions

## Paper Landscape

### Category A: Phenomenon & Diagnosis
1. **Gavrikov & Keuper (CVPR 2024)** — "Can Biases in ImageNet Models Explain Generalization?" — Large-scale study of 48 models; found that spectral biases alone do NOT holistically explain generalization; AT favorably shifts models toward shape bias.
2. **Liao et al. (arXiv 2024)** — "Evaluating Adversarial Robustness in the Spatial Frequency Domain" — SF-CNNs with DCT-based layers show lower frequency components contribute most to robustness.
3. **Kim et al. (WACV 2024)** — "Exploring Adversarial Robustness of ViTs in the Spectral Perspective" — ViTs rely more on phase and low-frequency information; different architectures have different spectral vulnerability profiles.
4. **Chen et al. (arXiv 2025)** — "Intriguing Frequency Interpretation of AT Robustness for CNNs and ViTs" — As high-frequency components increase, performance gap between AT and natural widens; CNNs vs. ViTs have opposite spectral attack profiles.

### Category B: Frequency-Rebalancing Methods
5. **Zhang et al. (Neural Networks 2025)** — HFDR: "Mitigating Low-Frequency Bias: Feature Recalibration and Frequency Attention Regularization" — Separates features into frequency bands, recalibrates high-frequency components. +2.89% on CIFAR-100, +3.09% on ImageNet-1K against AutoAttack.
6. **Bu et al. (ICCV 2023)** — FPCM: "Towards Building More Robust Models with Frequency Bias" — Plug-and-play module that reconfigures frequency components in intermediate features (not raw inputs).
7. **FAA** — "Fidelity-aware Adversarial Training" — Fourier-based decomposition perturbs only non-semantic frequency bands; replaces non-semantic components with target-domain spectral content. +8.6 mIoU on domain adaptation.
8. **Sun et al. (2022)** — FourierMix: Randomly perturbs amplitude and phase in Fourier domain + consistency regularizer. 18-26% improvement on OOD.
9. **Vaish et al. (CVPR 2024)** — AFA: "Fourier-basis Functions to Bridge Augmentation Gap" — Additive Fourier-basis noise; computationally cheap (~2× FLOPs). Improves on ImageNet-C, 3DCC, ImageNet-R.

### Category C: Layer/Module Criticality Methods
10. **Zhu et al. (ICCV 2023)** — RiFT: "Improving Generalization of AT via Robust Critical Fine-Tuning" — Identifies non-robust-critical modules (low MRC) and fine-tunes only those on clean data. ~1.5% improvement in OOD while maintaining robustness. Compatible with TRADES, MART, AWP, SCORE.
11. **Gopal et al. (ICML 2025)** — CLAT: "Criticality Leveraged Adversarial Training" — Identifies robustness-critical layers via feature weakness ratio; fine-tunes only ~5% of parameters. >2% robustness gain.

### Category D: Spectral Theory
12. **Xu, Zhang & Luo (2025)** — "Overview Frequency Principle/Spectral Bias in Deep Learning" — Comprehensive survey of F-Principle (DNNs fit low→high frequencies).
13. **Belfer et al. (JMLR 2024)** — "Spectral Analysis of NTK for Deep Residual Networks" — ResNTK eigenvalues decay as k^(-d); deep ResNTK biased toward even frequencies.
14. **Bowman & Montúfar (2024)** — Shows spectral bias extends beyond training set to test residuals.

## Gaps Identified

### Gap 1: RiFT + Spectral Rebalancing (method-transfer)
RiFT identifies non-robust-critical modules for fine-tuning, but NO existing work asks: do these modules have characteristic spectral biases? Can RiFT's MRC be re-purposed to guide spectral rebalancing? This is a clean method transfer — RiFT provides the "where to intervene" signal, spectral methods provide the "what to do."

### Gap 2: Post-AT Spectral Recalibration (untested assumption)
Most frequency-rebalancing methods (HFDR, FPCM) are integrated DURING adversarial training. The assumption that spectral imbalance must be addressed during AT is UNTESTED. Can we fix spectral imbalance POST-HOC via fine-tuning alone (e.g., RiFT + spectral augmentation)?

### Gap 3: Frequency-Domain Module Criticality (diagnostic)
Nobody has asked: WHY are certain modules "non-robust-critical" in RiFT's sense? Are these modules over-reliant on low frequencies or high frequencies? A diagnostic study linking MRC to spectral response profiles could reveal fundamental structure in AT networks.

### Gap 4: Low-Frequency Augmentation During Fine-Tuning (contradiction)
Standard AT improves high-frequency robustness but worsens low-frequency robustness. Data augmentation (AugMix, AutoAugment) helps both. The contradiction: AT + augmentation partially helps, but post-hoc fine-tuning with targeted low-frequency augmentation is unexplored.

### Gap 5: Spectral Mixup / Amplitude-Phase Perturbation for RiFT (scaling-regime)
FourierMix and AFA work by perturbing Fourier amplitude/phase during training. Can we adapt this for the RiFT fine-tuning phase — specifically perturbing the non-robust-critical modules' inputs/outputs in the frequency domain?

### Gap 6: Gradual Frequency Unfreezing (diagnostic)
During RiFT fine-tuning, can we progressively reintroduce frequency diversity (from low to high) to "walk back" the spectral imbalance?

## Landscape Summary

The field has converged on: (1) AT causes spectral reliance imbalance; (2) frequency-aware training can mitigate this; (3) layer-wise criticality methods (RiFT, CLAT) can improve AT generalization. However, NO work combines these two insights — no one has asked whether layer-wise criticality is fundamentally linked to spectral properties, or whether spectral rebalancing can be achieved via post-hoc fine-tuning rather than during AT.

The key opportunity: RiFT provides a principled way to identify WHICH modules to fine-tune. Frequency-domain augmentation provides WHAT to do during fine-tuning. The combination is unexplored and potentially high-impact.
