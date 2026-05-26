# Demo: Quick Inference

Quick semantic segmentation inference using MPLNet or MCPNet on a single image.

## Usage

```bash
# Basic run (illustrative — uses random backbone weights)
python demo/inference.py \
    --model mcpnet \
    --image sample_input/scene_001.jpg \
    --output sample_output/result.png

# With a trained checkpoint
python demo/inference.py \
    --model mcpnet \
    --image sample_input/scene_001.jpg \
    --checkpoint checkpoints/best_model.pth
```

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mcpnet` | Model name: `mcpnet` or `mplnet` |
| `--image` | (required) | Path to input image (JPG/PNG) |
| `--output` | `sample_output/result.png` | Path to save the output visualisation |
| `--num_classes` | `22` | Number of semantic classes |
| `--img_size` | `512` | Input image size |
| `--checkpoint` | `None` | Path to a trained `.pth` checkpoint |
| `--device` | Auto | Device override (`cuda` / `cpu`) |

## Output

The script produces a side-by-side image: **original | segmentation map** (22-class colour overlay).

## Sample Files

- `sample_input/` — Place your test images here. A placeholder image is provided.
- `sample_output/` — Results are saved here by default.

## Weights

Trained weights for TV-RSI-413 are **available upon reasonable academic request**.
They are not included in this public repository.

**Contact:** 854238019@qq.com

## Full Dataset

The TV-RSI-413 dataset is available via the Baidu Netdisk link below:

> **Baidu Netdisk / 百度网盘:**
> https://pan.baidu.com/s/1example-link-placeholder
>
> Please contact the author for the extraction code / 提取码请联系作者获取。

## Full Training & Evaluation

See [tools/train.py](../tools/train.py) and [tools/evaluate.py](../tools/evaluate.py) for full training and evaluation pipelines.

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
```