# Model Architecture Overview / 模型架构总览

> **Cheng Zhang, PhD / 张成博士**
>
> This document presents the complete architecture of MPLNet and MCPNet with Mermaid diagrams. All architectural details are verified from published papers and repository source code.

---

## 1. MCPNet: Multi-scale Context-Perceptual Network

**Paper:** Zhang et al., npj Heritage Science, 2026  
**DOI:** [10.1038/s40494-025-02253-1](https://doi.org/10.1038/s40494-025-02253-1)  
**Code:** `models/mcpnet.py`

### Architecture Diagram

```mermaid
flowchart TB
    subgraph Input["Input / 输入"]
        A["Remote Sensing Image<br/>遥感影像<br/>3 × 512 × 512"]
    end

    subgraph Stem["Stem / 茎干"]
        S1["Conv 7×7, stride=2<br/>BatchNorm + ReLU"]
        S2["MaxPool 3×3, stride=2"]
    end

    subgraph Encoder["ResNet-50 Encoder / 编码器"]
        L1["Layer 1<br/>3× Bottleneck<br/>256 ch | H/4×W/4"]
        L2["Layer 2<br/>4× Bottleneck<br/>512 ch | H/8×W/8"]
        L3["Layer 3<br/>6× Bottleneck<br/>1024 ch | H/16×W/16"]
        L4["Layer 4<br/>3× Bottleneck<br/>2048 ch | H/32×W/32"]
    end

    subgraph MCPHead["Multi-scale Context-Perceptual Head / 多尺度上下文感知头"]
        M1["Branch 1: Conv 1×1<br/>Point-wise context<br/>点级上下文"]
        M2["Branch 2: Conv 3×3<br/>Local patterns<br/>局部模式"]
        M3["Branch 3: Conv 3×3, d=4<br/>Effective 9×9 receptive field<br/>等效 9×9 感受野"]
        M4["Channel Attention<br/>GAP → FC → ReLU → FC → Sigmoid<br/>通道注意力机制"]
        M5["Spatial Attention<br/>AvgPool + MaxPool → Conv 7×7 → Sigmoid<br/>空间注意力机制"]
    end

    subgraph Fusion["Fusion & Output / 融合与输出"]
        F1["Feature Concatenation<br/>5C channels"]
        F2["Projection Conv 1×1<br/>256 ch"]
        F3["Refine Conv 3×3<br/>CBR(3×3)"]
        F4["Classifier 1×1<br/>22 classes"]
        F5["2× Upsample + Softmax"]
    end

    A --> S1 --> S2
    S2 --> L1 --> L2 --> L3 --> L4
    L1 -.-> M1
    L1 -.-> M2
    L1 -.-> M3
    L1 --> M4
    L1 --> M5
    M1 --> F1
    M2 --> F1
    M3 --> F1
    M4 --> F1
    M5 --> F1
    F1 --> F2 --> F3 --> F4 --> F5
```

### MCP Module Detail / MCP 模块详解

```mermaid
flowchart LR
    subgraph Input_MCP["Feature Map<br/>C × H × W"]
        X["Encoder Feature"]
    end

    subgraph ChannelAttn["Channel Attention / 通道注意力"]
        CA1["AdaptiveAvgPool2d → 1×1"]
        CA2["AdaptiveMaxPool2d → 1×1"]
        CA3["Shared FC: C → C/r → C"]
        CA4["Sigmoid → Channel Weight"]
    end

    subgraph MultiScale["Multi-scale Pooling / 多尺度池化"]
        MS1["AvgPool 1×1 → Conv 1×1"]
        MS2["AvgPool 3×3 → Conv 1×1"]
        MS3["AvgPool 5×5 → Conv 1×1"]
    end

    subgraph SpatialAttn["Spatial Attention / 空间注意力"]
        SA1["Channel-wise Avg + Max → 2 ch"]
        SA2["Conv 7×7 → 1 ch → Sigmoid"]
    end

    subgraph Output_MCP["Output / 输出"]
        O1["Weighted Feature + Multi-scale Sum"]
        O2["Conv 3×3 → BN → ReLU → C/4 ch"]
    end

    X --> CA1 & CA2
    CA1 & CA2 --> CA3 --> CA4
    X --> MS1 & MS2 & MS3
    CA4 --> O1
    MS1 & MS2 & MS3 --> O1
    X --> SA1 --> SA2
    SA2 --> O1
    O1 --> O2
```

### Verified Performance Metrics

| Metric | MCPNet | DeepLabV3+ | Advantage |
|--------|--------|------------|-----------|
| mAcc (Official, 20 classes) | **34.7%** | 29.6% | **+5.1 pp** |
| mIoU (Official, 20 classes) | **24.3%** | 21.7% | **+2.6 pp** |
| BF@3px | **71.0%** | 66.0% | **+5.0 pp** |
| Acc (Operational, 22 classes) | **85.3%** | — | — |
| mIoU (Operational, 22 classes) | **70.7%** | — | — |

### Computational Cost

| Metric | Value |
|--------|-------|
| Parameters | **55.3M** |
| GFLOPs | **198.5** |
| Inference Speed | **16 FPS** (FP32, batch=1, RTX 4080) |
| Efficiency Ratio | **1.047×** (accuracy per FLOP vs. ABMDRNet) |

---

## 2. MPLNet: Mamba Prompt Learning Network

**Paper:** Zhang et al., PLOS ONE, 2026  
**DOI:** [10.1371/journal.pone.0341130](https://doi.org/10.1371/journal.pone.0341130)  
**Code:** `models/mplnet.py`

### Architecture Diagram

```mermaid
flowchart TB
    subgraph Input_MPL["Input / 输入"]
        IA["Remote Sensing Image<br/>3 × 512 × 512"]
    end

    subgraph Encoder_MPL["ResNet-50 Encoder / 编码器"]
        EL1["Layer 1 | 256 ch"]
        EL2["Layer 2 | 512 ch"]
        EL3["Layer 3 | 1024 ch"]
        EL4["Layer 4 | 2048 ch"]
    end

    subgraph MambaBlock["Mamba SSM Block / Mamba 状态空间模块"]
        MB1["Linear Projection<br/>d_model → 2×d_inner"]
        MB2["Depthwise Conv1d<br/>kernel=3, groups=d_inner"]
        MB3["SiLU Activation"]
        MB4["SSM Core: Selective Scan<br/>Δ = softplus(dt_proj(x))<br/>A = -exp(A_log)<br/>D = learnable skip"]
        MB5["Gating: y * SiLU(z)"]
        MB6["Output Projection<br/>d_inner → d_model"]
    end

    subgraph PromptGen["Prompt Generator / 提示生成器"]
        PG1["Learnable Embeddings<br/>16 prompts × 64 dims"]
        PG2["Linear Projection<br/>64 → 2048 (embed_dim)"]
        PG3["Expand to Batch Size<br/>Feature Modulation"]
    end

    subgraph Decoder_MPL["MCP Head + Edge Branch / 解码器"]
        D1["Multi-scale Feature Fusion<br/>Layer1-4 → MCP Head"]
        D2["Segmentation Output<br/>22 classes | H/4×W/4"]
        D3["Edge Branch<br/>Layer1 → Conv → BN → ReLU → 1 ch"]
        D4["Edge Prediction<br/>Boundary Map"]
    end

    IA --> EL1 --> EL2 --> EL3 --> EL4
    EL4 --> MB1 --> MB2 --> MB3 --> MB4 --> MB5 --> MB6
    MB6 --> PG1
    PG1 --> PG2 --> PG3
    PG3 --> D1
    EL1 --> D1
    EL2 --> D1
    EL3 --> D1
    EL1 --> D3 --> D4
    D1 --> D2
```

### Mamba SSM Block Detail / Mamba 模块详解

```mermaid
flowchart LR
    subgraph Input_MB["Input<br/>B×L×d_model"]
        IX["x"]
    end

    subgraph Project["Linear Projection"]
        P1["in_proj: d_model → 2×d_inner"]
        P2["Split → x_inner, z"]
    end

    subgraph Conv["1D Convolution"]
        C1["Transpose: B×d_inner×L"]
        C2["Depthwise Conv1d<br/>kernel=3, padding=1"]
        C3["Transpose back + SiLU"]
    end

    subgraph SSM["Selective Scan (SSM)"]
        S1["x_proj → Δ, B, C"]
        S2["dt_proj(Δ) → softplus"]
        S3["A = -exp(A_log)"]
        S4["Discretize: ΔA, ΔB"]
        S5["Recurrent scan: h_t = Ā h_{t-1} + B̄ x_t"]
        S6["y_t = C h_t + D x_t"]
    end

    subgraph Gate["Gating & Output"]
        G1["SiLU(z)"]
        G2["Element-wise: y ⊙ SiLU(z)"]
        G3["out_proj: d_inner → d_model"]
    end

    IX --> P1 --> P2 --> C1 --> C2 --> C3 --> S1 --> S2 --> S4
    S3 --> S4 --> S5 --> S6 --> G2
    P2 --> G1 --> G2 --> G3
```

### Loss Function

MPLNet uses a combined objective tailored to long-tailed heritage class distributions:

```mermaid
flowchart LR
    subgraph Losses["Loss Components / 损失组件"]
        L1["L_CE<br/>Class-balanced Cross-Entropy<br/>类别平衡交叉熵<br/>α_c = (1-β)/(1-β^{n_c})"]
        L2["L_Dice<br/>Dice Loss<br/>Dice 损失<br/>λ_dice = optional"]
        L3["L_Edge<br/>Sobel Edge Loss<br/>Sobel 边缘损失<br/>λ_edge = 0.3"]
    end

    subgraph Total["Total Loss / 总损失"]
        TL["L = L_CE + λ_dice·L_Dice + λ_edge·L_Edge"]
    end

    L1 --> TL
    L2 --> TL
    L3 --> TL
```

---

## 3. EPANet-KD: Efficient Progressive Attention Network

**Paper:** Zhang et al., PLOS ONE, 2024  
**DOI:** [10.1371/journal.pone.0298452](https://doi.org/10.1371/journal.pone.0298452)

### Architecture Diagram

```mermaid
flowchart TB
    subgraph Input_EP["Input / 输入"]
        EA["Village Image<br/>224×224×3"]
    end

    subgraph Backbone_EP["Efficient Backbone / 高效骨干"]
        EB1["Lightweight CNN<br/>Feature Extraction"]
    end

    subgraph PAM["Progressive Attention Module (PAM) / 渐进注意力模块"]
        PA1["Spatial Attention<br/>空间注意力<br/>Salient region identification"]
        PA2["Channel Attention<br/>通道注意力<br/>Discriminative feature extraction<br/>beneath salient spatial regions"]
    end

    subgraph Distill["Knowledge Distillation (SAD) / 知识蒸馏"]
        KD1["Teacher Model<br/>Large pre-trained network"]
        KD2["SAD: Softened Alignment Distillation<br/>软化对齐蒸馏<br/>Softened category probability transfer"]
        KD3["Student Model: EPANet<br/>3.32M params | 0.42G FLOPs"]
    end

    subgraph Output_EP["Output / 输出"]
        EO1["11-class Village Classification<br/>11 类村落分类"]
    end

    EA --> EB1 --> PA1 --> PA2
    KD1 --> KD2 --> KD3
    PA2 --> KD3
    KD3 --> EO1
```

### EPANet-KD Efficiency / 效率指标

| Metric | Value |
|--------|-------|
| Parameters | **3.32M** |
| FLOPs | **0.42G** |
| Dataset | PVCD: 4,400 images, 11 Jiangxi region categories |
| Distillation | SAD (Softened Alignment Distillation) |

---

## 4. Method Comparison / 方法对比

```mermaid
flowchart LR
    subgraph Methods["Methods / 方法"]
        A["MCPNet<br/>Multi-scale Context<br/>Perception<br/>多尺度上下文感知<br/>npj Heritage Science 2026"]
        B["MPLNet<br/>Mamba + Prompt<br/>Learning<br/>Mamba 提示学习<br/>PLOS ONE 2026"]
        C["EPANet-KD<br/>Progressive Attention<br/>+ Knowledge Distillation<br/>渐进注意力+知识蒸馏<br/>PLOS ONE 2024"]
    end

    subgraph Scales["Research Scales / 研究尺度"]
        S1["Cross-scale<br/>跨尺度<br/>Spatial Gene Framework"]
        S2["Meso-scale<br/>中观<br/>Semantic Segmentation"]
        S3["Micro-scale<br/>微观<br/>Classification"]
    end

    subgraph Apps["Applications / 应用"]
        AP1["Heritage Conservation<br/>遗产保护"]
        AP2["Rural Planning<br/>乡村规划"]
        AP3["Digital Archiving<br/>数字存档"]
    end

    A --> S1
    B --> S2
    C --> S3
    S1 --> AP1
    S2 --> AP1 & AP2
    S3 --> AP3
```

---

## Citation

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

@article{zhang2026mplnet,
  title={MPLNet: Mamba prompt learning network for semantic segmentation of remote sensing images of traditional villages},
  author={Zhang, Cheng and Liu, P and Teng, J and et al.},
  journal={PLOS ONE},
  volume={21},
  number={2},
  pages={e0341130},
  year={2026},
  doi={10.1371/journal.pone.0341130}
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

*All architectural details verified from published papers and repository source code.*
*Last Updated: 2026-05-26*