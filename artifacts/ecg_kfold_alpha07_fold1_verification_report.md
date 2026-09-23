# Controlled Verification Report: Fold 1 with Sampling Exponent $\alpha = 0.70$

## Executive Summary

We executed the approved controlled verification experiment evaluating **Intermediate Frequency Balancing ($\alpha = 0.70$)** on **DS1 Fold 1** (18 training records, 4 validation records) using `GenericHybrid1DBiCNNGRU` (257,541 params) initialized from scratch.

### **Key Scientific Finding**
- **Sampling Proportions:** At $\alpha = 0.70$, training batches are sampled at **49.20% Normal, 15.08% SVEB, 23.05% VEB, 12.67% F**.
- **Normal (N) Recall:** Reached **94.66%** at Epoch 5 (compared to $99.33\%$ at $\alpha=0.50$ and $86.28\%$ at $\alpha=1.00$).
- **SVEB/A (Class 1) Recall:** At $\alpha = 0.70$, SVEB recall remained at **0.49% (1 / 204 beats)**, showing that an intermediate value of $\alpha = 0.70$ is insufficient to overcome the dominant Normal gradient, whereas $\alpha = 1.00$ activated SVEB recall to **73.04%**.

---

## 3-Way Side-by-Side Comparison: $\alpha = 0.50$ vs $\alpha = 0.70$ vs $\alpha = 1.00$ (Fold 1)

| Parameter / Metric | Benchmark ($\alpha = 0.50$) | Intermediate ($\alpha = 0.70$) | Equal Weighting ($\alpha = 1.00$) |
| :--- | :---: | :---: | :---: |
| **Active Target Sampling Proportion (N : SVEB : VEB : F)** | $65.5\% : 9.1\% : 18.5\% : 6.8\%$ | **$49.2\% : 15.1\% : 23.1\% : 12.7\%$** | **$25.0\% : 25.0\% : 25.0\% : 25.0\%$** |
| **Best Validation Epoch** | Epoch 2 | **Epoch 5** | Epoch 3 |
| **Active Accuracy (Classes 0–3)** | **91.35%** | 86.95% | 79.93% |
| **Active Macro F1-Score** | 41.83% | 37.97% | **50.28%** |
| **Minority Macro F1-Score** | 22.94% | 18.61% | **36.48%** |
| **Normal (N) Recall** | **99.33%** | 94.66% | 86.28% |
| **Normal Recall Penalty ($T=96.50\%$)** | **0.0000** | 0.0922 | 0.5112 |
| **SVEB/A (Class 1) Recall** | 0.00% (0 beats) | 0.49% (1 beat) | **73.04% (149 beats)** |
| **SVEB/A Precision & F1** | 0.00% / 0.00% | 2.08% / 0.79% | **34.49% / 46.86%** |
| **VEB/PVC (Class 2) Recall** | **92.75% (691 beats)** | 86.98% (648 beats) | 64.43% (480 beats) |
| **F/VT (Class 3) Recall** | 0.00% (0 beats) | 0.00% (0 beats) | **5.17% (25 beats)** |
| **Final CV Objective Score** | **0.4766** | 0.3428 | 0.0222 |

---

## Per-Class Confusion Matrix ($\alpha = 0.70$, Fold 1 Validation Set)

```
                  Predicted Class
           N       SVEB     VEB      F       Q
True N    7,332        9     403      2      0    (Total: 7,746 N)
True S       29        1     174      0      0    (Total:   204 SVEB)
True V       59       38     648      0      0    (Total:   745 VEB)
True F       99        0     385      0      0    (Total:   484 F)
```

---

## Sampler & Training Audit Summary

1. **Theoretical vs Empirical Sampling Match:**
   - Theoretical target per active class: **49.20% N, 15.08% SVEB, 23.05% VEB, 12.67% F**.
   - Empirical sampled counts (42,269 beats): N=49.13%, SVEB=14.85%, VEB=23.31%, F=12.70%, Q=0.00%.
2. **Q Class (Class 4) Isolation:**
   - Raw Q count: 8 beats ($0.02\%$).
   - Sampler weight: $w_4 = 0.00000000$.
   - Sampled Q beats: **0 beats** ($0.00\%$). Loss `ignore_index=4` active. Q 100% excluded.
3. **Execution Runtime & Memory:**
   - Elapsed Time: **328.72 seconds** (~5.48 minutes).
   - Peak RAM Footprint: **221.25 MB** (Clean, lightweight).

---

## Research Takeaways & Optuna Search Space Insights

1. **Non-Linear Threshold Activation of Minority Classes:**
   - $\alpha = 0.50$ ($9.1\%$ SVEB batches) $\implies$ SVEB Recall = $0.00\%$
   - $\alpha = 0.70$ ($15.1\%$ SVEB batches) $\implies$ SVEB Recall = $0.49\%$
   - $\alpha = 1.00$ ($25.0\%$ SVEB batches) $\implies$ SVEB Recall = **73.04%**
2. **Optuna Hyperparameter Search Relevance:**
   - Exploring $\alpha \in [0.0, 1.0]$ alongside `learning_rate` and `weight_decay` in Optuna is essential because learning rate and optimizer dynamics directly interact with class sampling probabilities to determine decision boundary placement.
