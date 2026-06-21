# TMF-Seg

Official implementation of our MICCAI 2026 paper:

Text-Guided Multi-Frequency Latent Diffusion for Medical Image Segmentation

TMF-Seg is a text-guided multi-frequency latent diffusion framework for medical image segmentation. It integrates image-text conditional fusion with text-guided multi-frequency enhancement for diffusion-based segmentation.

## News

- 2026-06: TMF-Seg has been provisionally accepted by MICCAI 2026.
- 2026-06: Code released for the MICCAI 2026 camera-ready submission.

## Overview

TMF-Seg contains three main components:

1. Image-Text Conditional Fusion: image and text conditions are integrated through cross-attention.
2. Text-Guided Multi-Frequency Enhancement: fused features are decomposed into high-, mid-, and low-frequency components and reweighted by textual context.
3. Latent Diffusion Segmentation: enhanced conditional features are fed into a latent diffusion U-Net for single-step segmentation.

## Environment

We recommend using Conda.

```bash
conda env create -f environment.yaml
conda activate tmfseg
```

If package conflicts occur, please install PyTorch according to your CUDA version and then install the remaining dependencies from `environment.yaml`.

## Datasets and Text Annotations

We evaluate TMF-Seg on three public datasets:

- Kvasir-SEG
- MosMedData+
- QaTa-COV19

For MosMedData+ and QaTa-COV19, we use the released multimodal datasets from LViT:

```text
https://github.com/HUANGLIZI/LViT
```

These datasets contain paired images, masks, and text annotations. We use the released text annotations directly, without additional manual text annotation.

For Kvasir-SEG, the image-mask pairs can be downloaded from the official dataset source. The corresponding text annotations are available upon reasonable request by email.

Recommended dataset structure, using Kvasir-SEG as an example:

```text
Kvasir-SEG/
├── train/
│   ├── frames/
│   ├── masks/
│   └── KSeg_train.xlsx
├── val/
│   ├── frames/
│   ├── masks/
│   └── KSeg_val.xlsx
└── test/
    ├── frames/
    ├── masks/
    └── KSeg_test.xlsx
```

Please update the image, mask, and text annotation paths in the corresponding configuration files under `configs/`.

## Training

Example training command:

```bash
nohup python -u main.py --base configs/SDSeg/kseg_text-ldm-kl-8.yaml -t --gpus 0, --name experiment_name > nohup/experiment_name.log 2>&1 &
```

Please replace `configs/SDSeg/kseg_text-ldm-kl-8.yaml`, `0`, and `experiment_name` with the target configuration file, GPU ID, and experiment name.

For different datasets, use the corresponding configuration files under `configs/SDSeg/`.

## Testing

Example testing command:

```bash
python -u scripts/slice2seg.py --dataset kseg
```

Please replace `kseg` with the target dataset name when evaluating other datasets.

## Checkpoints

Pretrained checkpoints are not included at the current stage. The model can be trained from scratch using the provided scripts and configuration files.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{gao2026tmfseg,
  title={Text-Guided Multi-Frequency Latent Diffusion for Medical Image Segmentation},
  author={Gao, Qiang and Wang, Yi and Zhang, Yong and Du, Lan and Li, Yong and Chen, Cunjian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2026}
}
```

If you use the released text annotations of MosMedData+ or QaTa-COV19, please also cite LViT:

```bibtex
@article{li2023lvit,
  title={Lvit: language meets vision transformer in medical image segmentation},
  author={Li, Zihan and Li, Yunxiang and Li, Qingde and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Zhang, You and Hong, Qingqi},
  journal={IEEE transactions on medical imaging},
  volume={43},
  number={1},
  pages={96--107},
  year={2023},
  publisher={IEEE}
}
```

## License

This project is released under the license provided in this repository.
