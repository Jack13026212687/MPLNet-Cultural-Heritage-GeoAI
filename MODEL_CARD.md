# Model Card: MPLNet & MCPNet

> **Model Description:** Mamba-driven Prompt Learning Network (MPLNet) and Multi-scale Context-Perceptual Network (MCPNet) for semantic segmentation of traditional village remote sensing imagery. Metrics verified from published papers.

**Developed by:** Cheng Zhang, PhD
**License:** MIT (source code only)
**Paper Reference:** [Zhang et al., 2026, npj Heritage Science](https://doi.org/10.1038/s40494-025-02253-1)

---

## Model Overview

### MCPNet (Flagship — npj Heritage Science 2026)

**Full Name:** Multi-scale Context-Perceptual Network

**Task:** Boundary-aware semantic segmentation for heritage landscape interpretation

**Key Innovation:** Multiscale context module that fuses low-level spatial detail with high-level abstractions to preserve cadastral edges, network continuity, and boundary fidelity in vernacular village scenes.

**Architecture Diagram:**
```
Input Image (3 × H × W)
    │
    ▼
ResNet-50 Backbone
    │
    ├── Layer 1: 256 channels
    ├── Layer 2: 512 channels
    ├── Layer 3: 1024 channels
    └── Layer 4: 2048 channels
    │
    ▼
Multi-scale Context-Perceptual Head
    │
    ├── Branch 1: Conv1×1 (point-wise context)
    ├── Branch 2: Conv3×3 (local patterns)
    └── Branch 3: Conv3×3, dilation=4 (effective 9×9, settlement-level)
    │
    ▼
Channel Attention + Spatial Attention
    (cbSE block: GAP→FC→ReLU→FC→Sigmoid)
    │
    ▼
Fusion Conv 3×3 → 1×1 Classifier
    │
    ▼
Output: 22-class Segmentation Map

Compute: 55.3M params | 198.5 GFLOPs | 16 FPS (FP32, batch=1, RTX 4080)
```

### MPLNet

**Full Name:** Mamba Prompt Learning Network

**Task:** Semantic segmentation with long-range dependency modeling via Mamba state-space model and domain adaptation via learnable prompts.

**Architecture:**
```
Input Image (3 × H × W)
    │
    ▼
ResNet-50 Backbone → Multi-scale features {f1, f2, f3, f4}
    │
    ▼
Mamba State-Space Block (applied to f4)
    │  d_state=16, expand=2
    │  Selective scan for long-range dependency
    ▼
Prompt Generator
    │  16 prompts × 64 dims → Linear projection → Feature modulation
    ▼
Multi-scale Context-Perceptual Head + Edge Branch
    │
    ▼
Output: Segmentation logits (22 × H/4 × W/4) + Edge map
```

Paper: Zhang et al., PLOS ONE 2026. DOI: [10.1371/journal.pone.0341130](https://doi.org/10.1371/journal.pone.0341130)

---

## Verified Performance Metrics

### MCPNet on TV-RSI-413 (Official Protocol, 20 spatial-gene classes)

| Metric | MCPNet | DeepLabV3+ | Improvement |
|--------|--------|------------|-------------|
| Mean Accuracy (mAcc) | **34.7%** | 29.6% | +5.1 pp |
| Mean IoU (mIoU) | **24.3%** | 21.7% | +2.6 pp |
| Boundary F-score (BF@3px) | **71.0%** | 66.0% | +5.0 pp |

*Note: Official protocol uses fixed splits, single-scale inference, 20 spatial-gene classes (BG/Occl excluded). Results verified from Zhang et al., npj Heritage Science, 2026.*

### MCPNet on TV-RSI-413 (Operational Setting, all 22 classes)

| Metric | Value |
|--------|-------|
| Overall Accuracy | **85.3%** |
| Mean IoU | **70.7%** |

*Note: Operational setting includes BG and Occl classes, emphasizing edge fidelity for practice-facing heritage workflows.*

### Culturally Salient Class Improvement

For heritage-critical classes (Ancestral Halls B02, Religious Buildings B03, Symbolic Structures B06), MCPNet achieves:
- IoU improvement: **23.4% → 33.6%** (+10.2 pp, +43% relative)
- This directly supports heritage workflows requiring clear delimitation around ritual cores and public landmarks.

### EPANet-KD on PVCD

| Metric | Value |
|--------|-------|
| Parameters | **3.32M** |
| Computation | **0.42G FLOPs** |
| Framework | Progressive Attention + Knowledge Distillation (SAD) |

*Source: Zhang et al., PLOS ONE 2024. DOI: [10.1371/journal.pone.0298452](https://doi.org/10.1371/journal.pone.0298452)*

### Annotation Quality

| Metric | Value |
|--------|-------|
| Inter-rater Agreement | Cohen's **κ = 0.92** |
| Quality Protocol | Three-stage audit: dual-annotator cross-check + expert sampling + script-based consistency |

---

## 22-Class Semantic Codebook (TV-RSI-413)

Five thematic modules, verified from published paper:

**Architecture (B01–B06)**
| Code | Class | Heritage Role |
|------|-------|--------------|
| B01 | Residential dwellings | Courtyard, row, clustered types |
| B02 | Ancestral halls | Clan lineage cores |
| B03 | Religious buildings | Temples and shrines |
| B04 | Memorial/ritual public spaces | Activity plazas, ceremonial forecourts |
| B05 | Production/service buildings | Barns, granaries, workshops |
| B06 | Symbolic/administrative structures | Paifang gateways, pavilions |

**Vegetation & Open Ground (V01–V05)**
| Code | Class |
|------|-------|
| V01 | Arborous vegetation |
| V02 | Shrubs |
| V03 | Herbaceous cover |
| V04 | Village open ground |
| V05 | Riparian buffers |

**Farmland (A01–A03)**
| Code | Class |
|------|-------|
| A01 | Paddy fields |
| A02 | Terraced farmland |
| A03 | Orchards |

**Water Systems (H01–H03)**
| Code | Class |
|------|-------|
| H01 | Natural rivers and streams |
| H02 | Reservoirs and impoundments |
| H03 | Ponds, ditches, irrigation channels |

**Roads (R01–R03)**
| Code | Class |
|------|-------|
| R01 | Primary access roads |
| R02 | Internal circulation roads and alleys |
| R03 | Traditional stone-paved footpaths |

**Operational Layers**
| Code | Class | Purpose |
|------|-------|---------|
| 21 | Background (BG) | Non-semantic areas (sky, haze, voids) |
| 22 | Occluder (Occl) | Transient foreground objects (people, cars, bins, stalls) |

---

## Training Configuration (Verified from Paper)

```yaml
model:
  name: MCPNet
  backbone: ResNet-50
  input_size: 224×224

training:
  epochs: 300
  optimizer: Adam
  lr: 1e-4
  weight_decay: 1e-4
  batch_size: 10
  amp: true
  gradient_clip: 1.0

loss:
  type: class_balanced_cross_entropy
  label_smoothing: optional
  dice_loss: optional
  edge_loss: optional (Sobel-based)

augmentation:
  random_crop: true
  flip: [horizontal, vertical] (p=0.5)
  rotation: [-10°, 10°]
  scale_jitter: [0.8, 1.2]
  color_jitter: true
```

---

## Evaluation Scope (Verified from Paper)

| Setting | Classes | Purpose |
|---------|---------|---------|
| **Official** | 20 spatial-gene classes only (BG/Occl excluded) | Standardized benchmark comparison |
| **Operational** | All 22 classes (incl. BG + Occl) | Practice-facing heritage workflow |

- Metrics: mIoU, mAcc, Boundary F-score (BF@3px)
- Confidence: 95% bootstrap CI (1,000 image-level resamples)
- Reproducibility: 3 random seeds, mean ± std reported

---

## What MCP Helps (Mechanistic Analysis — from Paper)

1. **Cross-scale aggregation**: Parallel dilations (1, 3, 9 receptive fields) couple thin boundary details (roof ridges, stone pavements) with settlement-level context
2. **Boundary-aware refinement**: Edge-sensitive guidance curbs category bleeding along narrow interfaces at façade lines, alley junctions, and riparian thresholds
3. **Contextual disambiguation**: Enlarged effective receptive field improves separability in visually dense vernacular fabrics

---

## EPANet-KD Overview (PLOS ONE, 2024)

| Component | Detail |
|-----------|--------|
| **Dataset** | PVCD (Provincial Village Classification Dataset): 4,400 images, 11 Jiangxi region categories |
| **Module** | PAM (Progressive Attention Module): spatial → channel cascaded attention for fine-grained features |
| **Distillation** | SAD (Softened Alignment Distillation): softened category probability transfer |
| **Efficiency** | 3.32M parameters, 0.42G FLOPs |
| **Application** | Fine-grained traditional village classification; digital preservation |

---

## Limitations

- **Input Size:** Optimized for 224×224 (paper); code supports 512×512
- **Domain:** Trained on Jiangxi Province traditional villages; may require fine-tuning for other regions
- **Class Imbalance:** Long-tailed distribution typical of heritage scenes; class weights and boundary losses are essential

---

## Ethical Considerations

- **Heritage Respect:** Model designed to support conservation, not displacement
- **Cultural Sensitivity:** Culturally salient classes (ancestral halls, temples) prioritized in evaluation
- **FAIR Principles:** Fixed splits, public scripts, reproducible protocol

---

## Citation

```bibtex
@article{zhang2026decoding,
  title={Decoding spatial genes of living heritage in traditional villages: TV-RSI-413 and MCPNet},
  author={Zhang, Cheng and Liu, Peilin and Teng, Jinlin and Liu, Chunqing},
  journal={npj Heritage Science},
  volume={14},
  pages={89},
  year={2026},
  doi={10.1038/s40494-025-02253-1}
}

@article{zhang2024epanet,
  title={EPANet-KD: Efficient progressive attention network for fine-grained provincial village classification via knowledge distillation},
  author={Zhang, Cheng and Liu, Chunqing and Gong, Huimin and Teng, Jinlin},
  journal={PLOS ONE},
  volume={19},
  number={2},
  pages={e0298452},
  year={2024},
  doi={10.1371/journal.pone.0298452}
}
```

---

## License Note

The MIT License applies only to the source code in this repository. Model weights, datasets, and publication materials are excluded.

---

*Last Updated: 2026-05-26*
*Metrics verified from published papers: Zhang et al., npj Heritage Science (2026); Zhang et al., PLOS ONE (2024).*