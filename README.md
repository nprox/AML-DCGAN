# AML-DCGAN
An experimental text-to-image complex DCGAN trained on a subset of the YFCC100M dataset through generated captions.

The generator combines Gaussian noise and Clip ViT-B/32 text embeddings through various applications of exponential minus log(EML) convolutional layers with FiLM conditioning. The discriminator uses frozen DINOv3 ConvNeXt-Small layers, FiLM conditioning, and EML convolutional layers to evaluate real-generated-mismatched image pairs.

## Repository
- models/eml_generator.py: Generator model file and EML layers.
- models/eml_discriminator.py: Discriminator model file.
- models/dataloader.py: Caption preparation, validation splits, and cached CLIP embeddings.
- train.py: Training with hinge loss, Adam, and mismatched-caption negatives.
- yfcc100m/: Image download, caption generation, and metadata processing scripts.

## Usage
Aggregate the dataset:

```bash
python3 yfcc100m/yfcc100m_download.py \
    --drives-dir /mnt
```

Caption the dataset:
```bash
python3 yfcc100m/yfcc100m_captioner.py \
    --drives-dir /mnt
```

Prepare the dataset:

```bash
python3 models/dataloader.py \
    --generated yfcc100m_captioned.jsonl \
    --drives-dir /mnt \
    --out ../datasets/eml_data
```

Train: A local DINOv3 ConvNeXt-Small model directory is required.

```bash
python3 train.py \
    --data ../datasets/eml_data \
    --model-dir /path/to/dinov3-convnext-small \
    --run runs/eml \
    --batch-size 95 \
    --epochs 100
```
