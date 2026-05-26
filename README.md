# 🏛️ MPLNet & TV-RSI-413: Decoding Spatial Genes of Living Heritage via GeoAI

<div align="center">

# 🏛️ Cultural-Heritage-GeoAI: TV-RSI-413, MCPNet & MPLNet
**Decoding Spatial Genes of Living Heritage via Advanced Geospatial AI**

[![Paper: npj Heritage Science](https://img.shields.io/badge/Paper-npj_Heritage_Science_(Q1)-blue.svg)](#)
[![Paper: PLOS ONE](https://img.shields.io/badge/Paper-PLOS_ONE_(Q2)-green.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> **Official Implementation** of the JCR Q1 paper *"Decoding spatial genes of living heritage in traditional villages: TV-RSI-413 and MCPNet"* and the Mamba-driven semantic segmentation framework *"MPLNet"*.

This repository provides an end-to-end Geospatial AI (GeoAI) pipeline designed specifically for the complex spatial morphology of cultural landscapes. By integrating **Multi-scale Context-Perceptual (MCP) modules** and **Mamba prompt learning** with traditional ecological wisdom, this framework solves the long-sequence dependency and edge-segmentation noise issues commonly found in standard CNN architectures when processing unstructured rural/heritage remote sensing imagery.

---

## 🌟 Key Features & Unfair Advantages

* **⚡ Boundary-Aware Architectures (MCPNet & MPLNet):** Utilizes an encoder coupled with a Multi-scale Context-Perceptual (MCP) head to fuse low-level spatial detail with high-level abstractions. **MCPNet operates highly efficiently with 55.3 M parameters, 198.5 GFLOPs, and an inference speed of 16 FPS.**
* **🗺️ First-of-its-kind Dataset (TV-RSI-413):** Open access to a high-resolution, multi-modal remote sensing image dataset containing **2,478 expert-annotated scenes from 413 traditional Jiangxi villages**.
* **🌍 Extreme Granularity:** Each image is pixel-labelled into **22 distinct classes** (with a high inter-rater agreement of Cohen's $\kappa=0.92$), effectively separating culturally significant spatial genes from transient foreground occluders.
* **🚀 Zero-Shot Generalization:** Demonstrated 100% success rate and 18.48 FPS inference speed when transferred zero-shot to morphologically distinct traditional villages in Hunan Province.

---

## 📂 Repository Structure

```text
Cultural-Heritage-GeoAI/
├── configs/                 # YAML configuration files for MCPNet and MPLNet
├── datasets/                # Data loaders and transforms for TV-RSI-413
│   ├── tv_rsi_413.py
│   └── transforms.py
├── models/                  # Core network architectures
│   ├── mcpnet.py            # ResNet-50 backbone + MCP Head
│   ├── mplnet.py            # Mamba Prompt Learning Network
│   └── layers/              # Custom attention and edge-aware loss modules
├── tools/                   # Scripts for training, evaluation, and inference
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── demo/                    # Sample imagery for quick testing
├── requirements.txt         # Environment dependencies
└── README.md
