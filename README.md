# MPLNet-Cultural-Heritage-GeoAI

**Mamba Prompt Learning for Cultural Heritage GeoAI and Traditional Village Spatial Gene Analysis**  
**面向传统村落保护的文化遗产 GeoAI、Mamba 提示学习与空间基因解析框架**

**Cheng Zhang, PhD / 张成博士**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

This repository presents a Cultural Heritage GeoAI research package developed by Cheng Zhang. It connects remote sensing semantic segmentation, Mamba prompt learning, traditional village spatial gene analysis, ecological suitability evaluation, and planning-oriented heritage conservation.

It is designed as a reproducible starting point for postdoctoral collaboration, joint publications, grant proposals, comparative heritage datasets, and AI-assisted landscape planning research.

### Research Asset Overview

| Research Asset | Evidence in This Repository | Collaboration Value |
|----------------|------------------------------|---------------------|
| Flagship Publication | npj Heritage Science 2026, DOI: 10.1038/s40494-025-02253-1 | Living heritage spatial gene framework |
| AI Models | MPLNet / MCPNet / Mamba prompt learning code | Remote sensing semantic segmentation |
| Verified Performance | MCPNet: mAcc 34.7%, mIoU 24.3% (Official) / Acc 85.3%, mIoU 70.7% (Operational) | +5.1 pp mAcc over DeepLabV3+ |
| Dataset Pipeline | TV-RSI-413: 413 villages, 2,478 scenes, 22 classes, κ=0.92 | Domain-specific Cultural Heritage GeoAI data |
| Patent Assets | 2 AI patents filed + 4 landscape patents granted | Method-system translation potential |
| Supplementary Dataset | PVCD: 4,400 images, 11 region categories | Fine-grained village classification |
| EPANet-KD | 3.32M params, 0.42G FLOPs knowledge distillation model | Efficient model deployment |
| Planning Translation | Heritage conservation, ecological zoning, micro-restoration, parcelisation | Planning-oriented heritage governance |

> **Metrics verified from published papers.** Zhang et al., npj Heritage Science (2026); Zhang et al., PLOS ONE (2024, 2025, 2026).

---

## 1. Hero Summary

This repository presents a complete **Cultural Heritage GeoAI research package**, including:

- **MPLNet & MCPNet**: Semantic segmentation frameworks for traditional village remote sensing imagery
- **TV-RSI-413 Dataset**: 413 villages × 2,478 annotated scenes × 22 semantic classes
- **Patent Assets**: AI methods for heritage-oriented semantic segmentation (applications filed)
- **Translation Pipeline**: From spatial gene analysis to landscape planning applications

---

## 2. Research Motivation

Traditional villages in China contain irreplaceable **spatial genes**—patterns of built form, land use, and ecological arrangement that encode centuries of traditional ecological wisdom. However, these villages face rapid transformation due to urbanization and rural revitalization pressures.

**Core Challenge:** Standard CNN architectures struggle with:
- Long-range spatial dependencies in unstructured rural environments
- Boundary segmentation noise between culturally significant elements
- Domain shift when transferring across morphologically distinct regions

**Our Approach:** Integrate Mamba state-space models and prompt learning with multi-scale context perception to address these challenges while maintaining computational efficiency.

---

## 3. Key Contributions

### 3.1 Novel Architectures

| Model | Innovation | Key Capability |
|-------|------------|----------------|
| **MPLNet** | Mamba + Prompt Learning + Boundary-Aware Segmentation | Long-range dependency modeling |
| **MCPNet** | Multi-scale Context-Perceptual Head + ResNet-50 | Boundary-aware segmentation |

### 3.2 Heritage Dataset

- **TV-RSI-413**: Dataset for traditional village semantic segmentation
- 413 traditional villages from Jiangxi Province
- 2,478 expert-annotated high-resolution scenes
- 22 semantic classes including culturally specific elements (temple, ancestral hall, well, pond)
- High annotation quality demonstrated through inter-rater agreement

### 3.3 Patent Portfolio

- 2 AI/GeoAI patent applications on Mamba prompt learning for semantic segmentation
- 10+ total patents covering AI methods and landscape architecture devices

---

## 4. Repository Structure

```
MPLNet-Cultural-Heritage-GeoAI/
│
├── models/                    # Core network architectures
│   ├── mplnet.py             # Mamba-driven Prompt Learning Network
│   ├── mcpnet.py             # Multi-scale Context-Perceptual Network
│   └── layers/               # Custom layers (Edge-aware loss, boundary refinement)
│
├── datasets/                  # Data pipeline
│   ├── tv_rsi_413.py        # TV-RSI-413 dataset loader
│   └── transforms.py        # Data augmentation transforms
│
├── configs/                   # YAML configuration files
│   ├── mplnet_tv_rsi_413.yaml
│   └── mcpnet_tv_rsi_413.yaml
│
├── tools/                     # Training and evaluation
│   ├── train.py             # Full training pipeline with TensorBoard
│   └── evaluate.py          # Comprehensive evaluation metrics
│
├── demo/                      # Quick inference demo
│   ├── inference.py         # Single-image inference script
│   └── README.md            # Demo usage instructions
│
├── sample_input/             # Sample images for demo
├── sample_output/            # Demo output directory
│
├── images/                    # Paper figures (55 images from 4 papers)
├── papers/                    # Publication PDFs & DOI references
│
├── MODEL_ARCHITECTURE_OVERVIEW.md  # Mermaid architecture diagrams
├── POSTDOC_PORTFOLIO.md           # Postdoctoral research portfolio
├── SCI_EVIDENCE_MATRIX.md         # Publication evidence matrix
├── IMAGE_GALLERY.md               # Complete figure gallery
├── MODEL_CARD.md                  # Model documentation
├── DATASET_CARD.md                # Dataset documentation
│
├── requirements.txt
└── README.md
```
├── MODEL_CARD.md             # Model documentation
├── DATASET_CARD.md           # Dataset documentation
│
├── requirements.txt
└── README.md
```

---

## 5. Core Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cultural Heritage GeoAI Pipeline              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Remote       │    │   Model       │    │   Spatial     │
│  Sensing      │───▶│   Training    │───▶│   Gene        │
│  Imagery      │    │   (MPLNet/    │    │   Analysis    │
│  (413 villages)│    │   MCPNet)    │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                                                    │
                                                    ▼
                          ┌───────────────────────────────────────┐
                          │   Heritage Conservation &            │
                          │   Landscape Planning Translation     │
                          └───────────────────────────────────────┘
```

---

## 6. Flagship Publication & Verified Performance

> **Zhang C, Liu P, Teng J, et al.** (2026). Decoding spatial genes of living heritage in traditional villages: TV-RSI-413 and MCPNet. *npj Heritage Science*, 14: 89.

[![DOI: 10.1038/s40494-025-02253-1](https://img.shields.io/badge/DOI-10.1038/s40494--025--02253--1-blue)](https://doi.org/10.1038/s40494-025-02253-1)

**Research Contributions:**
- TV-RSI-413: First dataset for traditional village semantic segmentation (413 villages, 2,478 scenes, 22 classes, κ = 0.92)
- MCPNet: Multi-scale context-perceptual architecture with boundary-aware refinement
- Three data sources: Ground DSLR + Low-altitude oblique UAV + Open street-view
- Five thematic modules: Architecture, Vegetation, Farmland, Water Systems, Roads
- Explicit Occluder class (Occl) for transient foreground objects — prevents mislabelling

### Verified Performance (from published paper)

| Metric | MCPNet | DeepLabV3+ | Advantage |
|--------|--------|------------|-----------|
| mAcc (Official, 20 classes) | **34.7%** | 29.6% | **+5.1 pp** |
| mIoU (Official, 20 classes) | **24.3%** | 21.7% | **+2.6 pp** |
| BF@3px (Boundary F-score) | **71.0%** | 66.0% | **+5.0 pp** |
| Culturally salient classes (B02/B03/B06) IoU | **33.6%** | 23.4% | **+10.2 pp (+43%)** |

| Metric | Value (Operational, all 22 classes) |
|--------|--------------------------------------|
| Overall Accuracy | **85.3%** |
| Mean IoU | **70.7%** |

### Efficiency

| Metric | MCPNet |
|--------|--------|
| Parameters | **55.3M** |
| GFLOPs | **198.5** |
| Inference | **16 FPS** (FP32, RTX 4080, batch=1) |

---

## 7. Visual Evidence from Papers / 论文视觉证据

### Flagship Paper Key Figures (npj Heritage Science 2026)

| MCPNet Architecture | Per-class Performance | Segmentation Results |
|:---:|:---:|:---:|
| ![MCPNet Arch](images/heritage_page8_img1.jpeg) | ![Heatmap](images/heritage_page10_img1.jpeg) | ![Results](images/heritage_page13_img1.jpeg) |
| *Multi-scale Context-Perceptual Head* | *IoU & Accuracy across 22 classes* | *MCPNet vs. DeepLabV3+ comparison* |

| Dataset Overview | Class Distribution | Cross-regional Transfer |
|:---:|:---:|:---:|
| ![Dataset](images/heritage_page6_img1.png) | ![Distribution](images/heritage_page10_img2.jpeg) | ![Transfer](images/heritage_page15_img1.jpeg) |
| *413 villages, 2,478 scenes, κ=0.92* | *Rank matrix across methods* | *Zero-shot to Hunan Province* |

### Related Papers Key Figures

| MPLNet (PLOS ONE 2026) | Ecological Suitability (PLOS ONE 2025) | EPANet-KD (PLOS ONE 2024) |
|:---:|:---:|:---:|
| ![MPLNet](images/zhongguan_page5_img1.png) | ![Ecological](images/hongguan_page8_img1.png) | ![EPANet](images/weiguan_page10_img1.jpeg) |
| *Mamba prompt learning architecture* | *Spatial suitability distribution* | *Classification performance* |

> **Complete figure gallery:** [IMAGE_GALLERY.md](IMAGE_GALLERY.md) — 55 figures across 4 papers with captions and research significance.

---

## 7.1 Related Publications

**Note:** Journal-level metrics (impact factor, JCR quartile, citations) should be verified from official indexing databases.

| Year | Work | Journal | DOI |
|------|------|---------|-----|
| 2026 | TV-RSI-413 and MCPNet: Decoding spatial genes of living heritage | npj Heritage Science | [10.1038/s40494-025-02253-1](https://doi.org/10.1038/s40494-025-02253-1) |
| 2026 | MPLNet: Mamba prompt learning network | PLOS ONE | [10.1371/journal.pone.0341130](https://doi.org/10.1371/journal.pone.0341130) |
| 2025 | Ecological suitability evaluation | PLOS ONE | [10.1371/journal.pone.0332375](https://doi.org/10.1371/journal.pone.0332375) |
| 2024 | EPANet-KD: Knowledge distillation | PLOS ONE | [10.1371/journal.pone.0298452](https://doi.org/10.1371/journal.pone.0298452) |
| 2022 | Yanfang village conservation | Open J. Social Sciences | [10.4236/jss.2022.105001](https://doi.org/10.4236/jss.2022.105001) |

**Patent Portfolio:** See [papers/README.md](papers/README.md) for patent application details.

---

## 8. Publication-supported Research System

The repository is supported by a publication sequence that connects macro-scale ecological suitability, meso-scale remote sensing semantic segmentation, micro-scale village classification, and cross-scale spatial gene interpretation.
本仓库由一组连续论文支撑，形成从宏观生态适宜性、中观遥感语义分割、微观村落分类到跨尺度空间基因解析的研究链条。

| Research Layer | Supporting Publication | Main Method | Repository Role |
|----------------|---------------------|-------------|----------------|
| Macro ecological suitability | PLOS ONE 2025 ecological suitability paper | AI multi-model integration | Site selection and environmental mechanism |
| Meso remote sensing segmentation | PLOS ONE 2026 MPLNet paper | Mamba prompt learning | Remote sensing semantic segmentation |
| Micro / classification | PLOS ONE 2024 EPANet-KD paper | Knowledge distillation | Fine-grained village classification |
| Cross-scale spatial genes | npj Heritage Science 2026 paper | TV-RSI-413 and MCPNet | Flagship Cultural Heritage GeoAI framework |
| Domain foundation | Open J. Social Sciences 2022 paper | Ecological wisdom and village conservation | Landscape architecture grounding |

**Key Insight:** This is not isolated technical experiments, but a continuous research system from ecological environment to AI model to heritage planning.

---

## 9. Dataset and Data Pipeline

### TV-RSI-413 Dataset

| Property | Value |
|----------|-------|
| Traditional Villages | 413 (Jiangxi Province) |
| Annotated Scenes | 2,478 |
| Semantic Classes | 22 |
| Annotation Quality | κ = 0.92 |
| Image Resolution | High-resolution TIF/PNG |

### Class Categories

**Built Environment:** Building, Traditional Structure, Modern Building, Temple, Ancestral Hall, Road, Bridge, Fence, Well

**Natural Elements:** Water, Pond, Mountain, Forest, Grass, Vegetation

**Land Use:** Farmland, Cultivated Land, Bare Soil, Construction Land

**Cultural Heritage:** Cultural Heritage Site, Ancestral Hall, Temple

---

## 10. Installation

### Requirements

```bash
# Python >= 3.8
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
pytorch-lightning>=2.0.0
tensorboard>=2.13.0
rasterio>=1.3.0
geopandas>=0.13.0
scikit-learn>=1.3.0
```

### Quick Install

```bash
git clone https://github.com/Jack13026212687/MPLNet-Cultural-Heritage-GeoAI.git
cd MPLNet-Cultural-Heritage-GeoAI
pip install -r requirements.txt
```

### Mamba Support (Optional)

```bash
pip install mamba-ssm
```

---

## 11. Quick Demo

Run semantic segmentation on a single image with a few lines:

```bash
# Quick inference (illustrative — uses random backbone weights)
python demo/inference.py \
    --model mcpnet \
    --image sample_input/your_image.jpg \
    --output sample_output/result.png

# With trained weights (available upon request)
python demo/inference.py \
    --model mcpnet \
    --image sample_input/your_image.jpg \
    --checkpoint checkpoints/best_model.pth
```

> **Demo documentation:** [demo/README.md](demo/README.md)
>
> **Trained weights:** Available upon reasonable academic request (854238019@qq.com). Not included in this public repository.
>
> **Full dataset:** Available via Baidu Netdisk. Contact the author for access.

---

## 12. Training & Evaluation

```bash
# Train MPLNet on TV-RSI-413
python tools/train.py \
    --model mplnet \
    --data_root ./data/TV-RSI-413 \
    --epochs 100 \
    --batch_size 16 \
    --img_size 512

# Evaluate
python tools/evaluate.py \
    --model mplnet \
    --checkpoint ./checkpoints/best_model.pth \
    --data_root ./data/TV-RSI-413 \
    --split test \
    --save_predictions
```

---

## 13. Model Architecture

Complete architecture diagrams (Mermaid) are available in [MODEL_ARCHITECTURE_OVERVIEW.md](MODEL_ARCHITECTURE_OVERVIEW.md).

### Quick Reference

| Model | Architecture | Key Innovation | Paper |
|-------|-------------|----------------|-------|
| **MCPNet** | ResNet-50 + Multi-scale Context-Perceptual Head | 3-branch dilation (1, 3, 9 receptive fields) + CBAM attention | npj Heritage Science 2026 |
| **MPLNet** | ResNet-50 + Mamba SSM + Prompt Generator + MCP Head | Mamba state-space model for long-range dependency + prompt learning for domain adaptation | PLOS ONE 2026 |
| **EPANet-KD** | Lightweight CNN + PAM + SAD Distillation | Progressive Attention Module + Softened Alignment Distillation | PLOS ONE 2024 |

### Verified Metrics

| Metric | MCPNet | vs DeepLabV3+ |
|--------|--------|---------------|
| mAcc (Official) | 34.7% | +5.1 pp |
| mIoU (Official) | 24.3% | +2.6 pp |
| Acc (Operational) | 85.3% | — |
| Params | 55.3M | — |
| Inference | 16 FPS | — |

> See [MODEL_CARD.md](MODEL_CARD.md) for full specifications and the 22-class semantic codebook.

---
## 14. Data Download / 数据下载

The TV-RSI-413 dataset and PVCD dataset are available for academic research.

> **Baidu Netdisk / 百度网盘:**
> 链接请通过邮件获取 | Link available via email request.
>
> **Trained model weights (.pth):** Available upon reasonable academic request.
>
> **Contact:** 854238019@qq.com

> **Note:** Large data files and model checkpoints are NOT stored in this GitHub repository. Please use the links above or contact the author directly.

---

## 15. For Prospective PIs and Collaborators

### Collaboration Value

This repository demonstrates capabilities suitable for joint postdoctoral projects:

1. **Reproducible AI Pipeline:** Well-structured code with training/evaluation/demo infrastructure
2. **Domain-Specific Dataset:** TV-RSI-413 with expert annotations (κ = 0.92)
3. **Patent Foundation:** Filed patents on core methods
4. **Publication Track:** Multiple journal articles in Cultural Heritage GeoAI
5. **Heritage Application:** Clear translation from analysis to conservation planning

### Potential Collaboration Directions

- Foundation model adaptation for heritage landscape interpretation
- Spatial gene atlas construction for rural landscape typology
- AI-assisted heritage conservation and rural revitalization planning
- Cross-regional transfer learning for traditional villages

---

## 16. Citation

If this work is useful for your research, please cite:

```bibtex
@article{zhang2026decoding,
  title={Decoding spatial genes of living heritage in traditional villages: TV-RSI-413 and MCPNet},
  author={Zhang, Cheng and Liu, P and Teng, J and et al.},
  journal={npj Heritage Science},
  volume={14},
  pages={89},
  year={2026},
  doi={10.1038/s40494-025-02253-1}
}
```

---

## 17. License and Data Availability

**License Scope:** The MIT License applies only to original source code in this repository. Publications, datasets, images, patent documents, third-party materials, and model weights are excluded unless explicitly stated.

> **Journal Information:** Journal-level metrics (impact factor, JCR quartile, citations) should be verified from official indexing databases before public release.

### Data Availability

The TV-RSI-413 dataset will be made available upon final publication.

---

## 18. Contact

| Channel | Information |
|---------|-------------|
| **Email** | 854238019@qq.com |
| **GitHub** | [Jack13026212687](https://github.com/Jack13026212687) |
| **Location** | Nanchang, Jiangxi, China |

---

## 19. Acknowledgments

This repository is maintained as part of postdoctoral research in Cultural Heritage GeoAI.

---

*This repository is maintained by Cheng Zhang as part of his postdoctoral research portfolio.*
