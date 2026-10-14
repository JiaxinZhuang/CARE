<p align="center">
  <img src="assets/header.png" alt="CARE" width="100%" />
</p>

# CARE

**CARE** is a clinically validated CT foundation model designed for comprehensive CT analysis across five major regions.
Trained through large-scale vision–language pretraining, it provides generalizable representations for diverse downstream applications, including volume-level recognition, region-level recognition, pixel-level recognition, and vision–language tasks.


## Installation

### Prerequisites

- A CUDA-compatible GPU is recommended for CARE inference.
- Download the pretrained CARE model checkpoint as described below.

### Install Dependencies

```bash
git clone https://github.com/JiaxinZhuang/CARE.git
cd CARE
pip install -r requirements.txt
```

### Download Model Checkpoint

The pretrained CARE checkpoint will be provided through the CARE model repository. Download the checkpoint and place it in the `ckpts/` directory at the root of this repository.

```bash
mkdir -p ckpts
# Download the CARE checkpoint and place it in ckpts/
```

The expected directory structure is:

```text
CARE/
├── ckpts/
│   └── care.pth
├── requirements.txt
└── README.md
```

The checkpoint can then be loaded from `ckpts/care.pth` for model inference and downstream evaluation.

## Scope of This Release

In addition to CARE model loading, CT preprocessing, feature extraction, and the quickstart notebook, this repository releases code for six downstream tasks:

| Task | Released directory | Dataset |
|---|---|---|
| Volume fine-tuning (FT) | `downstream/0_volume_recognition/abnormality_cls_ft/` | Not included |
| Volume linear probing (LN) | `downstream/0_volume_recognition/abnormality_cls_ln/` | CT-RATE |
| Region classification (ROI) | `downstream/1_region_recognition/roi_cls/` | PANORAMA |
| Segmentation (SEG) | `downstream/2_pixel_recognition/segmentation/` | KiPA22 |
| Image-text retrieval (RET) | `downstream/3_vision_language_tasks/retrieval/` | CTRG |
| Report generation (RG) | `downstream/3_vision_language_tasks/report_generation/` | FLARE 2025 Task 05 |

FT is released as code only; its data, splits, and task-specific weights are not included.
The four public downstream datasets are CT-RATE, PANORAMA, KiPA22, and CTRG; report generation uses FLARE 2025 Task 05.
Required datasets and checkpoints are not bundled.
Users must obtain them from the respective data or model providers or from a future CARE Hugging Face release.

Private institutional CT volumes, radiology reports, and patient-level annotations are not included.
Prospective validation, worklist prioritization, and crossover RCT analyses rely on restricted institutional data and are not included in the publicly runnable examples.


## Acknowledgments

- Data preprocessing was supported by [MONAI](https://github.com/Project-MONAI/MONAI), [MedGemma](https://huggingface.co/google/medgemma-4b-it), and [Qwen3](https://github.com/QwenLM/Qwen3).
- Vision–language pretraining was built on [OpenCLIP](https://github.com/mlfoundations/open_clip).


## License and Terms of Use

The models and associated code are released under the [CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/) and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, sublicensing, or other monetization of the CARE models and their derivatives is prohibited and requires prior written approval.

Downloading the models or feature packages may require prior registration on Hugging Face and agreement to the applicable terms of use. By downloading the models, you agree not to distribute, publish, or reproduce copies of the models. If another user within your organization wishes to use the models, they must register as an individual user and agree to comply with the terms of use.

Users may not attempt to re-identify any deidentified data used to develop the underlying models or feature packages. Commercial entities should contact the corresponding author or appropriate institutional licensing office.

See [LICENSE](LICENSE) for the repository license notice.


## Citation
If you use CARE in your research, please cite:
```bibtex
@misc{zhuang2026care,
  title={A clinically validated CT foundation model with large-scale vision--language pretraining},
  author={Zhuang, Jiaxin and Li, Tao and Wu, Linshan and Guo, Zhengrui and Qiu, Zelin and others},
  year={2026},
}
```


## Version History
- v1.1.0 (2026-10): Update with CARE checkpoint and downstream evaluation workflow.