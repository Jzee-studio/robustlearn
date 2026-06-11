# Research Idea Report

**Direction**: Mitigating spectral reliance imbalance (SRI) in adversarially trained models to improve OOD generalization, ideally via RiFT-compatible fine-tuning
**Generated**: 2026-06-11
**Ideas evaluated**: ~12 generated → 8 surviving filtering → 5 recommended

---

## Landscape Summary

Adversarial training (AT) is the leading defense against adversarial attacks, but it induces a well-documented failure mode: **spectral reliance imbalance (SRI)**. Because adversarial perturbations concentrate in high frequencies, AT-trained models learn to disregard high-frequency features and shift their discriminative reliance toward low-frequency components (shape, coarse structure). This creates a brittle dependency — the model performs well on clean and adversarially perturbed inputs but fails dramatically on low-frequency OOD corruptions like fog, contrast shifts, and brightness changes.

Recent work has attacked this problem from two independent directions. **Frequency-rebalancing methods** (HFDR, FPCM, FAA, FourierMix, AFA) intervene during training to force the model to use the full frequency spectrum, typically by manipulating features in the Fourier domain or applying frequency-aware augmentation. **Layer-criticality methods** (RiFT, CLAT) observe that not all layers contribute equally to robustness, and that fine-tuning only specific "non-robust-critical" modules can recover generalization without sacrificing robustness.

The critical gap: **no existing work connects these two insights**. Nobody has asked whether a module's robustness criticality is fundamentally linked to its spectral response profile, or whether spectral rebalancing can be achieved post-hoc via targeted fine-tuning of specific modules. RiFT provides a principled "where to intervene" signal; frequency-domain methods provide "what to do." Their combination is an unexplored design space with high potential.

All five recommended ideas are novel to the best of our knowledge — they sit at the intersection of module criticality and spectral analysis, a frontier that both communities have not yet crossed.

---

## Recommended Ideas (ranked)

### Idea 1: Fourier-RiFT — Spectral Augmentation During Robust Critical Fine-Tuning

- **Method (what we actually do)**:
  1. Train a baseline AT model (e.g., TRADES on ResNet-18 / CIFAR-100).
  2. Run RiFT's MRC computation to identify the single non-robust-critical module (lowest MRC score).
  3. During RiFT fine-tuning of that module, apply Fourier-domain perturbations to the module's input features: randomly swap amplitude spectra between different images within a mini-batch while preserving phase, or add Gaussian noise to amplitude components.
  4. Train with a combined loss: standard cross-entropy + optional consistency regularizer (encouraging the module to produce similar features across Fourier-perturbed and unperturbed versions of the same input).
  5. Apply RiFT's weight interpolation to find the optimal trade-off.

- **Hypothesis**: Injecting spectral diversity specifically into non-robust-critical modules forces them to process a broader range of frequency components, reducing the model's over-reliance on low-frequency cues. Because these modules have "redundant" robustness capacity (per RiFT), the spectral perturbation won't degrade adversarial robustness.

- **Minimum experiment**: CIFAR-100, ResNet-18 + TRADES. Compare: (a) standard RiFT, (b) Fourier-RiFT with amplitude perturbation, (c) Fourier-RiFT with phase perturbation, (d) Fourier-RiFT with full amplitude+phase swap. Measure AutoAttack robustness + CIFAR-100-C (focusing on contrast/brightness/fog/frost). Single seed, ~2 GPU-hours.

- **Expected outcome**: Positive: Fourier-RiFT improves CIFAR-100-C scores by 2-4% over standard RiFT on low-frequency corruptions, with no degradation in AutoAttack robustness. Negative: spectral augmentation during fine-tuning provides no benefit over standard clean-data fine-tuning, suggesting the spectral imbalance must be addressed during AT training, not post-hoc.

- **Novelty**: 8/10 — closest work: RiFT (ICCV 2023) does clean-data fine-tuning only; FourierMix (2022) and AFA (CVPR 2024) apply Fourier augmentation during training, not post-hoc fine-tuning. The combination is entirely novel.

- **Feasibility**: Low cost. RiFT code is open-source; Fourier augmentation is ~50 lines of PyTorch. CIFAR-100 experiments run in hours on a single RTX 3090.

- **Risk**: LOW — grounded in two independently validated methods. Worst case: spectral augmentation provides no additional benefit over clean-data fine-tuning, which is still a publishable finding (it tells us that post-hoc spectral intervention is insufficient).

- **Contribution type**: Empirical finding + new method

- **Pilot result**: SKIPPED (needs GPU; estimated <2h on single GPU)

- **Reviewer's likely objection**: "Fourier augmentation during fine-tuning is not fundamentally different from applying Fourier augmentation during training — what's the unique insight from combining it with RiFT?" Counter: the insight is that spectral imbalance is module-specific, not global; targeting augmentation to specific modules is more efficient and preserves robustness better than global spectral augmentation.

- **Why we should do this**: It's the most direct test of the central hypothesis — that combining RiFT's module selectivity with spectral augmentation yields benefits beyond either approach alone. Positive or negative, the result advances our understanding of whether spectral imbalance can be fixed post-hoc.

---

### Idea 2: Spectral Entropy Regularization for RiFT Fine-Tuning

- **Method (what we actually do)**:
  1. Train baseline AT model and identify non-robust-critical module via RiFT's MRC.
  2. During fine-tuning, compute the 2D FFT of the module's output feature maps.
  3. Treat the normalized power spectrum as a probability distribution over frequencies. Compute its entropy: H = -Σ p(f) log p(f). Low entropy = over-concentration on narrow frequency bands (the AT pathology).
  4. Add a regularization term to the fine-tuning loss: L = L_CE - λ * H_spectral, where λ controls the strength. Maximizing spectral entropy encourages the module to use frequencies more uniformly.
  5. Weight interpolation as in standard RiFT.

- **Hypothesis**: Non-robust-critical modules in AT models exhibit low spectral entropy (concentrated on low frequencies). Maximizing spectral entropy during fine-tuning redistributes the module's discriminative capacity across the frequency spectrum, improving robustness to both high-frequency attacks and low-frequency corruptions.

- **Minimum experiment**: Same setup as Idea 1. Compare different λ values (0.01, 0.1, 1.0). Measure spectral entropy before/after fine-tuning as a diagnostic. ~2 GPU-hours.

- **Expected outcome**: Positive: spectral entropy increases after fine-tuning, and CIFAR-100-C scores improve proportionally to the entropy gain. Negative: spectral entropy maximization conflicts with the classification objective, causing clean accuracy to drop without improving corruption robustness.

- **Novelty**: 9/10 — spectral entropy as a regularization target for robustness is novel. Closest work: HFDR uses frequency attention, but as an architectural module during AT, not as a regularization term during fine-tuning. No prior work uses spectral entropy as a direct optimization target.

- **Feasibility**: Very low cost. FFT + entropy computation is trivially added to any training loop.

- **Risk**: MEDIUM — spectral entropy maximization might conflict with discriminative learning. The regularization strength λ is a critical hyperparameter that may require tuning.

- **Contribution type**: New method + diagnostic

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "Maximizing spectral entropy might amplify noise frequencies that are genuinely uninformative." Counter: the fine-tuning is applied to non-robust-critical modules, which RiFT has already shown can be modified without harming robustness. The modules' features still pass through the rest of the frozen network, which provides a natural filter.

- **Why we should do this**: It's the most principled approach — directly optimizing what we care about (spectral balance) rather than indirectly encouraging it through augmentation. It also produces a clean diagnostic signal (entropy curves) that can be analyzed regardless of whether the method works.

---

### Idea 3: Contrastive Low-Frequency Invariance Fine-Tuning (CLFIT)

- **Method (what we actually do)**:
  1. Train baseline AT model, identify non-robust-critical module via MRC.
  2. During fine-tuning, for each mini-batch, create two views of each image: (a) clean original, (b) low-frequency corrupted version (randomly apply contrast shift, brightness change, or Gaussian blur).
  3. Fine-tune the non-robust-critical module with a combined loss: L = L_CE(clean) + α * L_contrastive(f(clean), f(corrupted)), where L_contrastive is an InfoNCE-style loss that encourages the module's features to be invariant to low-frequency corruptions.
  4. The contrastive term explicitly trains the module to produce similar representations regardless of low-frequency shifts.
  5. Weight interpolation as in standard RiFT.

- **Hypothesis**: AT models fail on low-frequency corruptions because their features change dramatically under these shifts (despite the shifts being perceptually mild). Enforcing feature-level invariance to low-frequency corruptions in non-robust-critical modules teaches the network that these corruptions are semantically irrelevant, directly addressing the OOD failure mode.

- **Minimum experiment**: CIFAR-100, ResNet-18 + TRADES. Compare: (a) standard RiFT (clean fine-tuning), (b) CLFIT with only contrast shift, (c) CLFIT with contrast + brightness + blur. Measure CIFAR-100-C (especially low-frequency corruptions: contrast, brightness, fog, frost). ~3 GPU-hours.

- **Expected outcome**: Positive: CLFIT significantly outperforms standard RiFT on low-frequency CIFAR-100-C corruptions (3-5% improvement), especially contrast and brightness. Negative: contrastive invariance conflicts with discriminative fine-tuning, causing clean accuracy degradation without OOD improvement.

- **Novelty**: 7/10 — contrastive learning for invariance is well-established, but applying it specifically to low-frequency corruptions during module-targeted fine-tuning is novel. Closest work: AugMix uses Jensen-Shannon consistency loss on predictions, not feature-level contrastive loss on specific modules.

- **Feasibility**: Moderate. Requires generating corrupted views during training, but the corruptions are simple and fast (torchvision transforms).

- **Risk**: LOW-MEDIUM — the contrastive loss is well-understood and likely to produce some invariance. The main risk is that feature-level invariance may not translate to prediction-level robustness.

- **Contribution type**: New method

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "Why contrastive loss instead of simpler consistency regularization?" Answer: contrastive loss operates in feature space and explicitly pushes apart representations of different images while pulling together representations of the same image under different corruptions. This is stronger than output-space consistency and specifically targets the feature-level spectral imbalance.

- **Why we should do this**: Directly addresses the core pathology — sensitivity to low-frequency shifts. The contrastive framing also provides a clean ablation pathway (which corruptions matter most?).

---

### Idea 4: Adaptive Frequency Curriculum During RiFT Fine-Tuning

- **Method (what we actually do)**:
  1. Train baseline AT model, identify non-robust-critical module.
  2. Design a frequency curriculum: start fine-tuning with data augmented by low-frequency corruptions only (contrast, brightness — the corruptions AT models are supposedly robust to), then progressively introduce higher-frequency corruptions (Gaussian noise, impulse noise) over the course of fine-tuning epochs.
  3. The rationale: AT models already handle low frequencies; start there to maintain stability, then gradually "stretch" toward higher frequencies.
  4. Use a frequency-band mixing coefficient β(t) that increases from 0 (pure low-frequency) to 1 (uniform frequency mixture) over training.
  5. Weight interpolation as in RiFT.

- **Hypothesis**: A gradual frequency curriculum prevents the catastrophic interference that might occur if high-frequency signals are abruptly reintroduced during fine-tuning. By "walking" the module from its low-frequency comfort zone toward spectral balance, the curriculum achieves better final performance than uniform frequency mixing.

- **Minimum experiment**: CIFAR-100, ResNet-18 + TRADES. Compare: (a) standard RiFT, (b) uniform frequency mixing during fine-tuning, (c) low→high curriculum, (d) high→low curriculum (reverse control). ~3 GPU-hours.

- **Expected outcome**: Positive: the low→high curriculum outperforms uniform mixing, and the reverse curriculum performs worst (validating the theoretical rationale). Negative: curriculum ordering doesn't matter — all frequency mixing approaches perform similarly, suggesting the benefit comes from frequency diversity, not ordering.

- **Novelty**: 8/10 — curriculum learning is established, but frequency-domain curricula for robustness fine-tuning are novel. No prior work has explored the order in which frequency bands should be reintroduced during post-AT fine-tuning.

- **Feasibility**: Low cost. Curriculum scheduling is a simple addition to the training loop.

- **Risk**: MEDIUM — the benefit of curriculum learning is often marginal in practice. May require careful tuning of the curriculum schedule.

- **Contribution type**: Empirical finding

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "Is the curriculum effect large enough to justify the added complexity over uniform frequency mixing?" Must demonstrate non-trivial improvement.

- **Why we should do this**: Even a negative result (curriculum doesn't matter) is informative — it tells us spectral rebalancing is about total frequency exposure, not ordering, which simplifies future method design.

---

### Idea 5: Spectral Anatomy of Module Robustness Criticality (Diagnostic Study)

- **Method (what we actually do)**:
  1. Train AT models (TRADES, PGD-AT) on CIFAR-100 with ResNet-18/34.
  2. For each module, compute: (a) MRC via RiFT's method, (b) spectral response profile: pass the model images with bandpass-filtered frequency content and measure each module's activation change across frequency bands, (c) effective receptive field in the frequency domain.
  3. Correlate MRC with spectral metrics: spectral entropy of features, high-frequency response ratio, frequency localization width, amplitude/phase sensitivity ratio.
  4. Test the key hypothesis: non-robust-critical modules (low MRC) exhibit pathological spectral profiles — extremely low spectral entropy or severe low-frequency bias.
  5. If confirmed, propose a "spectral-MRC" metric that combines both for better module selection.

- **Hypothesis**: Module robustness criticality is partially explained by spectral properties. Non-robust-critical modules are those that became most severely low-frequency biased during AT. This would provide a mechanistic explanation for RiFT's effectiveness and enable better module selection criteria.

- **Minimum experiment**: CIFAR-100, ResNet-18, 2-3 AT methods. For each module, compute spectral response profiles using bandpass-filtered inputs (low-pass, band-pass, high-pass). Correlate with MRC. Generate spectral response heatmaps per module. ~5 GPU-hours (offline analysis, no training needed beyond the base AT models).

- **Expected outcome**: Positive: strong negative correlation between MRC and spectral entropy (r < -0.7), confirming the spectral basis of module criticality. This opens a new diagnostic tool for AT model analysis. Negative: weak or no correlation, suggesting MRC and spectral bias are independent axes of variation — useful but less exciting.

- **Novelty**: 9/10 — no prior work has connected layer-wise robustness criticality to spectral response profiles. The diagnostic question "why are some modules non-robust-critical?" has never been asked.

- **Feasibility**: Low cost for the diagnostic. Mostly analysis code on pre-trained models.

- **Risk**: MEDIUM — purely diagnostic studies are harder to publish at top venues unless the finding is striking. Consider pairing with a method idea (e.g., Idea 1 or 2) in the same paper.

- **Contribution type**: Diagnostic / empirical finding

- **Pilot result**: SKIPPED

- **Reviewer's likely objection**: "Correlation is not causation — even if you find MRC-spectral correlations, this doesn't prove spectral bias causes non-robust-criticality." Must use causal interventions: if we artificially increase a module's spectral entropy, does its MRC change?

- **Why we should do this**: It provides the mechanistic foundation for Ideas 1-4. A strong positive finding here would catalyze a new sub-field at the intersection of module criticality and spectral analysis. Even a null result is informative.

---

## Bonus Ideas (not fully developed, for reference)

### Idea B1: Dual-Teacher Frequency Distillation
Use a standard-trained model as a "frequency diversity teacher" alongside the frozen AT model as a "robustness teacher." During RiFT fine-tuning, distill from both — the AT teacher provides low-frequency stability, the standard teacher provides high-frequency feature recovery. Higher compute cost but potentially stronger results.

### Idea B2: Amplitude-Phase Decoupling Analysis
Fourier-transform features of non-robust-critical modules. Fine-tune by swapping amplitude spectra (containing style/texture info) with standard model features while preserving phase (structure). Test whether amplitude or phase is the bottleneck in AT models' spectral imbalance.

### Idea B3: Frequency-Aware MRC (FA-MRC)
Replace RiFT's MRC metric with a frequency-weighted variant. Instead of measuring worst-case weight perturbation impact uniformly, measure it separately for low-frequency and high-frequency inputs. This could identify modules that are critical specifically for low-frequency robustness vs. high-frequency robustness.

---

## Eliminated Ideas

| Idea | Reason eliminated |
|------|-------------------|
| "Apply HFDR during RiFT fine-tuning" | HFDR requires architectural changes (feature disentanglement module) that must be trained during AT; incompatible with post-hoc fine-tuning |
| "Train a separate spectral augmentation network" | Requires > 1 week GPU time; too expensive for initial exploration |
| "Apply FourierMix globally during fine-tuning" | Less novel than module-targeted approach; FourierMix is already published and global application doesn't leverage RiFT's key insight |
| "Meta-learning for frequency adaptation" | Too speculative; unclear minimum viable experiment |

---

## Suggested Execution Order

1. **Start with Idea 5 (Spectral Anatomy)** — ~1 day of analysis on existing AT checkpoints. This provides the diagnostic foundation and can be written up as a short findings paper even if standalone. The results directly inform how to design Ideas 1-4.

2. **Run Idea 1 (Fourier-RiFT) and Idea 2 (Spectral Entropy Reg) in parallel** — ~1-2 days each on CIFAR-100. These are the two most promising method ideas, and they explore complementary mechanisms (data augmentation vs. explicit optimization).

3. **If Ideas 1-2 show positive signal, run Idea 3 (CLFIT) and Idea 4 (Curriculum)** — These are refinements that can be added to the best-performing base method.

4. **Combine insights into a unified paper** — The narrative arc: diagnostic (Idea 5) → method (Ideas 1-2) → refinements (Ideas 3-4) is a strong paper structure.

---

## Next Steps

- [ ] Obtain or download pre-trained AT checkpoints (TRADES, PGD-AT on CIFAR-100) for Idea 5 analysis
- [ ] Implement FFT-based spectral analysis utilities
- [ ] Set up RiFT codebase (already open-source: github.com/microsoft/robustlearn)
- [ ] Run Idea 5 diagnostic → evaluate whether spectral-MRC correlation exists
- [ ] Based on Idea 5 results, prioritize Ideas 1-4
- [ ] Full experiment with multi-seed validation for the top method
