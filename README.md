# 🏛️ MPLNet & TV-RSI-413: Decoding Spatial Genes of Living Heritage via GeoAI

[![Paper: npj Heritage Science](https://img.shields.io/badge/Paper-npj_Heritage_Science_(Q1)-blue.svg)](#)
[![Paper: PLOS ONE](https://img.shields.io/badge/Paper-PLOS_ONE_(Q2)-green.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Implementation** of the 2026 JCR Q1 paper *"Decoding spatial genes of living heritage in traditional villages: TV-RSI-413 and MCPNet"* and the Mamba-based semantic segmentation framework *"MPLNet"*.

This repository provides an end-to-end Geospatial AI (GeoAI) pipeline designed specifically for the complex spatial morphology of cultural landscapes. By integrating **Mamba prompt learning** with traditional ecological wisdom, this framework solves the long-sequence dependency and edge-segmentation noise issues commonly found in standard CNN architectures when processing unstructured rural/heritage remote sensing imagery.

---

## 🌟 Key Features & Unfair Advantages

*   **⚡ Mamba-Driven Architecture (MPLNet):** Utilizes linear-time sequence modeling (Mamba) combined with prompt learning modules to achieve high-precision semantic segmentation on heterogeneous remote sensing data. 
*   **🗺️ First-of-its-kind Dataset (TV-RSI-413):** Open access to a high-resolution, multi-modal remote sensing image dataset covering 413 traditional villages, complete with pixel-level spatial gene annotations.
*   **🌍 Climate & Ecological Suitability (EPANet-KD):** Includes knowledge-distilled lightweight models deployed for macro-scale ecological adaptability mapping and multi-model AI integration.

---

## 📊 The TV-RSI-413 Dataset

The lack of specialized geospatial data is the biggest bottleneck in digital heritage conservation. **TV-RSI-413** bridges this gap.

*   **Scale:** High-resolution optical imagery across 413 diverse traditional villages in Jiangxi, China.
*   **Annotations:** Fine-grained, pixel-level masks classifying core spatial genes (e.g., historical waterways, traditional road networks, vernacular settlement footprints, ecological buffers).
*   **Download:** [Insert link to Google Drive / OneDrive / Zenodo data hosting here]

*(Insert a high-quality visualization image here showing the original satellite image vs. your model's perfect segmentation mask. Use code: `![Dataset Preview](path_to_image.png)`)*

---

## 🧠 Model Architecture: MPLNet & MCPNet

Unlike generic vision transformers, our architectures are specifically explicitly prompted to recognize the "ecological wisdom" and morphological rules of traditional human settlements. 

*(Insert your best 2D architectural diagram of the Mamba Prompt Learning Module here. Ensure it has English labels. Use code: `![Architecture](path_to_image.png)`)*

### Performance Benchmark
| Model Architecture | Back-bone | mIoU (%) | F1-Score | Inference Time (ms) |
| :--- | :--- | :--- | :--- | :--- |
| Standard CNN (ResNet50) | CNN | XX.X | XX.X | XX.X |
| Vision Transformer (ViT) | Transformer | XX.X | XX.X | XX.X |
| **MPLNet (Ours)** | **Mamba** | **+X.X** | **+X.X** | **Fastest** |

---

## 🚀 Quick Start & Installation

We have containerized the environment to ensure zero-friction deployment for GeoAI researchers and landscape planners.

```bash
# 1. Clone the repository
git clone [https://github.com/YourUsername/MPLNet-Cultural-Heritage-GeoAI.git](https://github.com/YourUsername/MPLNet-Cultural-Heritage-GeoAI.git)
cd MPLNet-Cultural-Heritage-GeoAI

# 2. Create a Conda environment
conda create -n geoai_mamba python=3.9
conda activate geoai_mamba

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt

# 4. Run inference on a sample village image
python tools/inference.py --config configs/mplnet_base.yaml --image demo/village_sample.tif
