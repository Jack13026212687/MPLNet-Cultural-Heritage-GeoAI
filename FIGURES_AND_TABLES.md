# Figures & Tables for Cultural Heritage GeoAI Portfolio
# 文化遗产 GeoAI 博士后资产包 — 图表说明

> **Cheng Zhang, PhD / 张成博士**
>
> This directory contains visual assets extracted and reconstructed from published papers (Zhang et al., 2022–2026) for academic portfolio demonstration.

---

## Figure 1: MPLNet Architecture Overview / MPLNet 架构总览

```mermaid
flowchart TB
    subgraph Input["Input / 输入"]
        A["Remote Sensing Image<br/>遥感影像<br/>512×512×3"]
    end

    subgraph Encoder["ResNet-50 Encoder / 编码器"]
        B1["Layer 1<br/>256 channels<br/>H/4 × W/4"]
        B2["Layer 2<br/>512 channels<br/>H/8 × W/8"]
        B3["Layer 3<br/>1024 channels<br/>H/16 × W/16"]
        B4["Layer 4<br/>2048 channels<br/>H/32 × W/32"]
    end

    subgraph MambaModule["Mamba SSM Block / Mamba 状态空间模块"]
        C1["Conv1d<br/>Depthwise Conv"]
        C2["SSM Core<br/>Selective Scan"]
        C3["SiLU + Linear<br/>Gating Mechanism"]
    end

    subgraph PromptModule["Prompt Generator / 提示生成器"]
        D1["Prompt Embeddings<br/>16 × 64 dims"]
        D2["Prompt Projection"]
        D3["Domain Adaptation<br/>Feature Modulation"]
    end

    subgraph Decoder["MCP Head + Edge Branch / 多尺度上下文感知头"]
        E1["Multi-scale Features<br/>Layer1-4"]
        E2["Fusion Conv<br/>256 channels"]
        E3["Segmentation Output<br/>22 classes"]
        E4["Edge Output<br/>Boundary Map"]
    end

    A --> B1 --> B2 --> B3 --> B4
    B4 --> C1 --> C2 --> C3
    C3 --> D1
    D1 --> D2 --> D3
    D3 --> E1
    B1 --> E1
    B2 --> E1
    B3 --> E1
    E1 --> E2
    E2 --> E3
    B1 --> E4
```

## Figure 2: MCPNet Architecture Overview / MCPNet 架构总览

```mermaid
flowchart TB
    subgraph Input2["Input / 输入"]
        A2["Remote Sensing Image<br/>遥感影像<br/>512×512×3"]
    end

    subgraph Stem["Stem / 茎干"]
        S1["Conv 7×7 + BN + ReLU"]
        S2["MaxPool 3×3"]
    end

    subgraph Encoder2["ResNet-50 Encoder / 编码器"]
        L1["Layer 1<br/>256 ch"]
        L2["Layer 2<br/>512 ch"]
        L3["Layer 3<br/>1024 ch"]
        L4["Layer 4<br/>2048 ch"]
    end

    subgraph MCPModules["Multi-scale Context-Perceptual Modules / 多尺度感知模块"]
        M1["Channel Attention<br/>通道注意力"]
        M2["Spatial Attention<br/>空间注意力"]
        M3["Multi-scale Pooling<br/>1×1, 3×3, 5×5"]
    end

    subgraph Fusion["Fusion + Output / 融合与输出"]
        F1["Feature Fusion<br/>256 → num_classes"]
        F2["Segmentation Map<br/>22 classes"]
    end

    A2 --> S1 --> S2
    S2 --> L1 --> L2 --> L3 --> L4
    L1 --> M1
    L1 --> M2
    L1 --> M3
    M1 --> F1
    M2 --> F1
    M3 --> F1
    L2 --> F1
    L3 --> F1
    L4 --> F1
    F1 --> F2
```

## Figure 3: TV-RSI-413 Dataset Overview / TV-RSI-413 数据集总览

```mermaid
flowchart LR
    subgraph Coverage["Geographic Coverage / 地理覆盖"]
        G1["Jiangxi Province<br/>江西省"]
        G2["413 Traditional Villages<br/>413 个传统村落"]
    end

    subgraph Annotation["Annotation Scheme / 标注体系"]
        AN1["2,478 Expert-annotated Scenes<br/>2,478 个专家标注场景"]
        AN2["22 Semantic Classes<br/>22 个语义类别"]
        AN3["Cohen's κ = 0.92<br/>高标注一致性"]
    end

    subgraph Classes["Class Categories / 类别体系"]
        CL1["Built Environment<br/>建筑环境 (7 classes)"]
        CL2["Water Features<br/>水体 (3 classes)"]
        CL3["Natural Elements<br/>自然要素 (4 classes)"]
        CL4["Land Use<br/>土地利用 (4 classes)"]
        CL5["Cultural Heritage<br/>文化遗产 (2 classes)"]
    end

    G1 --> G2
    G2 --> AN1
    AN1 --> AN2
    AN2 --> AN3
    AN2 --> CL1
    AN2 --> CL2
    AN2 --> CL3
    AN2 --> CL4
    AN2 --> CL5
```

## Figure 4: Research Scope Expansion / 研究范畴扩展图谱

```mermaid
flowchart TB
    subgraph Core["Core GeoAI Capability / 核心 GeoAI 能力"]
        CORE1["Semantic Segmentation<br/>语义分割<br/>MPLNet + MCPNet"]
        CORE2["Spatial Gene Analysis<br/>空间基因解析<br/>TV-RSI-413"]
    end

    subgraph MethodChain["Method Chain / 方法链"]
        METH1["Ecological Suitability<br/>生态适宜性评价<br/>Multi-model AI"]
        METH2["Mamba Prompt Learning<br/>Mamba 提示学习<br/>Long-range dependency"]
        METH3["Knowledge Distillation<br/>知识蒸馏<br/>EPANet-KD"]
    end

    subgraph Application["Application Domains / 应用领域"]
        APP1["Heritage Conservation<br/>遗产保护"]
        APP2["Rural Planning<br/>乡村规划"]
        APP3["Landscape Ecology<br/>景观生态"]
        APP4["Digital Heritage<br/>数字遗产"]
    end

    subgraph Expansion["Postdoc Expansion Directions / 博后扩展方向"]
        EXP1["Foundation Models<br/>基础模型适配<br/>GeoAI + LLM/VFM"]
        EXP2["Digital Twins<br/>数字孪生<br/>Heritage monitoring"]
        EXP3["Cross-cultural Transfer<br/>跨文化迁移<br/>Global heritage sites"]
        EXP4["Climate Resilience<br/>气候韧性<br/>Village adaptation"]
    end

    CORE1 --> METH2
    CORE2 --> METH1
    METH1 --> APP1
    METH2 --> APP1
    METH2 --> APP3
    METH3 --> APP2
    APP1 --> EXP1
    APP2 --> EXP2
    APP3 --> EXP4
    APP4 --> EXP3
```

## Figure 5: Competitive Advantage Matrix / 竞争优势矩阵

| Dimension / 维度 | Typical CV Approach / 普通简历 | This Portfolio / 本资产包 | Advantage / 优势 |
|---|---|---|---|
| **Papers / 论文** | Lists paper titles | Shows paper sequence forming a research system (Macro→Meso→Micro→Cross-scale) | Demonstrates research continuity |
| **Code / 代码** | Links to scattered repos | Complete model zoo with training pipeline, configs, and evaluation scripts | Ready for reproduction |
| **Data / 数据** | Mentions dataset name | Full dataset card with annotation protocol, class definitions, and quality metrics | Verifiable data pipeline |
| **Patents / 专利** | Lists patent numbers | Links patents to specific model components and methods | Method-system translation |
| **Domain / 领域** | States "Landscape Architecture" | Shows AI capability grounded in 2022 heritage conservation foundation paper | AI + Domain legitimacy |
| **Collaboration / 合作** | "I hope to learn..." | Structured research package ready for joint projects and grants | PI-ready collaboration proposal |

## Figure 6: Postdoc Position Matching Map / 博士后岗位匹配图谱

```mermaid
flowchart LR
    subgraph Position1["GeoAI & Remote Sensing"]
        P1A["Hong Kong PolyU<br/>LSGI / RS Group"]
        P1B["NUS Geography<br/>GeoAI Lab"]
        P1C["Wuhan Univ<br/>LIESMARS"]
    end

    subgraph Position2["AI for Cities & Digital Twins"]
        P2A["HKU Urban Analytics<br/>Future Cities Lab"]
        P2B["ETH Zurich<br/>Digital Twins"]
        P2C["Tsinghua<br/>Urban Informatics"]
    end

    subgraph Position3["Heritage & Landscape"]
        P3A["HKU Architecture<br/>Heritage Conservation"]
        P3B["Tongji CAUP<br/>Landscape Planning"]
        P3C["MIT DUSP<br/>Cultural Landscapes"]
    end

    subgraph Position4["Foundation Models & CV"]
        P4A["CUHK MMLab<br/>Computer Vision"]
        P4B["Nanjing Univ<br/>IMAGINE Lab"]
        P4C["Singapore<br/>AI Singapore"]
    end

    subgraph Assets["Cheng Zhang's Assets / 张成资产包"]
        A1["MPLNet + MCPNet<br/>Semantic Segmentation"]
        A2["TV-RSI-413<br/>Heritage Dataset"]
        A3["5 Publications<br/>2022-2026"]
        A4["10+ Patents<br/>AI + Landscape"]
    end

    A1 --> P1A
    A1 --> P1B
    A1 --> P1C
    A1 --> P4A
    A1 --> P4B
    A1 --> P4C
    A2 --> P2A
    A2 --> P2B
    A2 --> P2C
    A2 --> P4A
    A2 --> P4B
    A3 --> P3A
    A3 --> P3B
    A3 --> P3C
    A4 --> P3A
    A4 --> P3B
    A4 --> P4C
```

## Figure 7: Publication Timeline with Method Evolution / 论文时间线与方法演进

| Year | Publication | Method | Research Scale | Capability Built | Application |
|------|-------------|--------|----------------|------------------|-------------|
| 2022 | Yanfang Ancient Village | Ecological wisdom analysis | Village / Case | Heritage conservation domain | Conservation planning |
| 2024 | EPANet-KD | Knowledge distillation | Provincial / Classification | Efficient model deployment | Village typology |
| 2025 | Ecological suitability | AI multi-model integration | Provincial / Suitability | Environmental spatial analysis | Site selection |
| 2026 | MPLNet (Mamba prompt learning) | Mamba SSM + Prompt | Village / Segmentation | Long-range dependency modeling | Remote sensing |
| 2026 | **TV-RSI-413 + MCPNet** | MCP architecture | Cross-scale / Framework | Spatial gene framework | Heritage GeoAI |

---

*Note on PDF Figures:* The original papers (Zhang et al., 2022–2026) contain additional publication-quality figures including semantic segmentation result comparisons, confusion matrices, class distribution statistics, and geographical distribution maps. These should be extracted from the published PDFs and added to the `images/` folder for portfolio display. The Mermaid diagrams above are reconstructions based on code-verified architecture, config-verified class definitions, and author-provided research descriptions.

*关于论文原图：* 原始论文（Zhang et al., 2022-2026）包含了出版级别的图表，包括语义分割结果对比、混淆矩阵、类别分布统计和地理分布图。这些图应从已发表 PDF 中提取后添加到 `images/` 文件夹用于资产包展示。以上 Mermaid 图表是基于代码验证的架构、配置文件验证的类别定义和作者提供的研究描述重建的。