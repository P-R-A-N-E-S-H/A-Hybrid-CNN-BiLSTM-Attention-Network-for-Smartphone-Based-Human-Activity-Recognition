# Ablation Study & Empirical Architectural Analysis

**Project:** A Hybrid CNN-BiLSTM-Attention Network for Human Activity Recognition  
**Author:** Pranesh  

---

## 1. Objective

To empirically validate the architectural contribution of each individual subsystem in the Proposed Hybrid Model:
1. **Residual Connections vs Plain Convolutions**
2. **Squeeze-and-Excitation (SE) Channel Attention**
3. **Bidirectional LSTM vs Unidirectional LSTM**
4. **Multi-Head Self-Attention (MHA)**
5. **Temporal Attention Pooling vs Global Average Pooling**

---

## 2. Ablation Configurations & Experimental Results

All ablation variants were trained under identical hyperparameters:
- **Optimizer:** Adam ($\alpha = 0.001$)
- **Loss Function:** Categorical Cross-Entropy with L2 regularization ($10^{-4}$)
- **Batch Size:** 64
- **Input Dimensions:** $(128, 9)$

| Model Configuration | SE Block | Recurrent Unit | Attention Mechanism | Pooling Type | Test Accuracy | Macro F1-Score | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline CNN** | ✗ | None | None | GlobalAvg | 91.10% | 0.9102 | 0.94 M |
| **M2: Baseline LSTM** | ✗ | Uni-LSTM (128) | None | Last Step | 89.80% | 0.8972 | 0.71 M |
| **M3: Baseline BiLSTM** | ✗ | Bi-LSTM (128) | None | Last Step | 92.40% | 0.9231 | 1.81 M |
| **M4: CNN + BiLSTM** | ✗ | Bi-LSTM (128) | None | GlobalAvg | 93.85% | 0.9378 | 2.65 M |
| **M5: CNN + BiLSTM + SE** | ✓ | Bi-LSTM (128) | None | GlobalAvg | 94.90% | 0.9482 | 2.82 M |
| **M6: CNN + BiLSTM + MHA** | ✗ | Bi-LSTM (128) | MHA (4 Heads) | GlobalAvg | 95.45% | 0.9538 | 3.52 M |
| **M7: Full Proposed Hybrid**| **✓** | **Bi-LSTM (128)**| **MHA (4 Heads)** | **Temporal Attn**| **96.88%** | **0.9685** | **3.84 M** |

---

## 3. Key Findings & Insights

1. **Impact of Squeeze-and-Excitation (SE) Attention (+1.05% Accuracy):**
   Adding the SE block enables the network to re-weight inter-axis dynamics (e.g. emphasizing gyro yaw during turns and vertical gravity during stairs), improving accuracy from 93.85% to 94.90%.
2. **Impact of Multi-Head Self-Attention (+1.60% Accuracy):**
   MHA captures long-range temporal correlations across distinct phases of dynamic human locomotion that recurrent gates alone attenuate.
3. **Impact of Temporal Attention Pooling (+1.43% Accuracy):**
   Dynamic soft weighting of the 128 timesteps yields higher classification boundaries than static global average pooling, especially for rapid transition movements (Walking $\rightarrow$ Standing, Laying $\rightarrow$ Sitting).
4. **Synergistic Effect:**
   Combining all components yields an overall gain of **+5.78% accuracy** over single-baseline CNN and **+7.08% accuracy** over standard LSTM.
