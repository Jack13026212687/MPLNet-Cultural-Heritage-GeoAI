# Dataset Card: TV-RSI-413 & PVCD

> **Dataset Description:** Traditional Village Remote Sensing Image dataset (TV-RSI-413) and Provincial Village Classification Dataset (PVCD) for semantic segmentation, spatial gene analysis, and fine-grained classification of traditional villages.

**Curated by:** Cheng Zhang, PhD
**License:** To be determined upon publication
**Paper References:** [Zhang et al., npj Heritage Science, 2026](https://doi.org/10.1038/s40494-025-02253-1) | [Zhang et al., PLOS ONE, 2024](https://doi.org/10.1371/journal.pone.0298452)

---

## Dataset Overview

### TV-RSI-413 (Semantic Segmentation Dataset)

| Property | Value |
|----------|-------|
| Traditional Villages | **413** (all officially designated villages in Jiangxi Province) |
| Annotated Scenes | **2,478** (6 scenes per village) |
| Semantic Classes | **22** (20 spatial-gene + BG + Occluder) |
| Annotation Quality | Cohen's **κ = 0.92** |
| Data Sources | 3 complementary: Ground DSLR + Low-altitude oblique UAV + Open street-view |
| Geographic Coverage | Jiangxi Province, China (stratified by prefecture, geomorphic zone, vernacular style) |
| Image Format | TIF / PNG |
| Input Resolution | 224×224 (paper), 512×512 (config) |

### Dataset Construction (Verified from Paper)

**Three-Stage Quality Protocol:**
1. Dual-annotator cross-check with disagreement logging
2. Expert sampling review focused on boundary-dense and minority classes
3. Script-based consistency tests (class set, topology plausibility, holes/speckle filters)

**Occluder Class Design:**
- Dedicated Occl class prevents transient foreground objects (people, cars, bicycles, bins, stalls) from contaminating stable spatial-gene labels
- Improves open-set robustness in ground-view deployments

**FAIR Infrastructure:**
- Fixed train/val/test splits (stratified by prefecture, geomorphic zone, vernacular style)
- Public evaluation scripts
- Machine-readable class definitions

---

## 22-Class Semantic Codebook (5 Thematic Modules)

### Architecture Module (B01–B06)
| Code | Class | Heritage Significance |
|------|-------|---------------------|
| B01 | Residential dwellings | Courtyard, row, clustered settlement types |
| B02 | Ancestral halls | Clan lineage cores — culturally salient |
| B03 | Religious buildings | Temples and shrines — culturally salient |
| B04 | Memorial/ritual public spaces | Activity plazas, ceremonial forecourts |
| B05 | Production/service buildings | Barns, granaries, workshops |
| B06 | Symbolic/administrative structures | Paifang gateways, pavilions — culturally salient |

### Vegetation & Open Ground Module (V01–V05)
| Code | Class | Ecological Function |
|------|-------|-------------------|
| V01 | Arborous vegetation | Canopy cover, microclimate regulation |
| V02 | Shrubs | Understory biodiversity |
| V03 | Herbaceous cover | Ground-level vegetation matrix |
| V04 | Village open ground | Courtyards, bare earth, compacted ground |
| V05 | Riparian buffers | Riverbank protective belts — hydro-ecological |

### Farmland Module (A01–A03)
| Code | Class | Agricultural Context |
|------|-------|-------------------|
| A01 | Paddy fields | Lowland hydrology-linked |
| A02 | Terraced farmland | Hilly/mountain terrain adaptation |
| A03 | Orchards | Perennial agricultural systems |

### Water Systems Module (H01–H03) — Hydro-sensitive classes
| Code | Class | Hydrological Role |
|------|-------|------------------|
| H01 | Natural rivers and streams | Primary water network |
| H02 | Reservoirs and impoundments | Water storage infrastructure |
| H03 | Ponds, ditches, irrigation channels | Micro-water management |

### Roads Module (R01–R03) — Connectivity classes
| Code | Class | Connectivity Role |
|------|-------|------------------|
| R01 | Primary access roads | External village connectivity |
| R02 | Internal circulation roads and alleys | Intra-village movement |
| R03 | Traditional stone-paved footpaths | Heritage walking routes — boundary-sensitive |

### Operational Layers
| Code | Class | Evaluation Scope |
|------|-------|-----------------|
| 21 | Background (BG) | Non-semantic areas (distant sky, haze, textureless voids) |
| 22 | Occluder (Occl) | Non-gene, non-background transient objects |

---

## Evaluation Protocols (Verified from Paper)

### Official Protocol
- **Scope:** 20 spatial-gene classes (BG/Occl excluded)
- **Inference:** Single-scale, fixed split
- **Metrics:** mIoU, mAcc, BF@3px, Bootstrap 95% CI
- **Purpose:** Standardized benchmark comparison

### Operational Protocol
- **Scope:** All 22 classes (BG + Occl included)
- **Emphasis:** Edge fidelity for practice-facing heritage workflows
- **Metrics:** Overall Accuracy, mIoU

---

## Dataset Applications (Five Purposes — Verified from Paper)

1. **Standardized Benchmark:** Fine-grain semantic segmentation in boundary-dense, long-tailed class settings typical of vernacular interiors
2. **Spatial Gene Quantification:** Pixel-level quantification supporting digital archiving, cultural-value assessment, and micro-scale restoration planning
3. **Planning Reference:** Semantic vectors for landscape renewal, green-infrastructure optimization, hydrological planning, and parcel-level cluster management
4. **Imbalance-Aware Learning Testbed:** Rigorous setting for reweighting, augmentation, topology-preserving losses, and domain adaptation
5. **Open-Set Robustness:** Dedicated Occl class prevents false positives on heritage elements in ground-view deployments

---

## PVCD (Classification Dataset — EPANet-KD Paper)

| Property | Value |
|----------|-------|
| Dataset Name | Provincial Village Classification Dataset (PVCD) |
| Images | **4,400** |
| Region Categories | 11 (Fuzhou, Ganzhou, Ji'an, Jingdezhen, Jiujiang, Nanchang, etc.) |
| Data Sources | Web crawling + manual arrangement |
| Application | Fine-grained traditional village classification |
| Model Used | EPANet-KD (3.32M params, 0.42G FLOPs) |

---

## Data Pipeline

### Loading (from repository code)

```python
from datasets import TVRSI413Dataset, get_dataloader

dataset = TVRSI413Dataset(
    root="./data/TV-RSI-413",
    split="train",
    img_size=(512, 512),
    transforms=train_transforms,
)
```

### Transforms

```python
from datasets import get_train_transforms, get_val_transforms

# Training: Resize + Flip + Rotate + ColorJitter + Normalize
train_transforms = get_train_transforms(img_size=(512, 512))

# Validation: Resize + Normalize
val_transforms = get_val_transforms(img_size=(512, 512))
```

---

## Ethical Considerations

- **Living Heritage:** Dataset represents ongoing community practice — intended for conservation
- **FAIR Compliance:** Fixed splits, public scripts, machine-readable class definitions
- **Cultural Sensitivity:** Heritage-critical classes (B02/B03/B06) explicitly tracked and evaluated

---

## Access

Dataset access information will be provided upon final publication.

**Contact:** 854238019@qq.com

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

*Last Updated: 2026-05-26*
*Dataset details verified from published papers.*