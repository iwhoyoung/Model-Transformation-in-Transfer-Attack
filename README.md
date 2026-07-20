# MoTA

This repository contains a minimal implementation for the paper **So-Called Input Transformation-Based Attacks May be Model Transformation-Based Attacks**.

Authors: Yang Hu, Tao Yang, Yuheng He, Qingyun Sun, Xiuli Bi, Bin Xiao, and Jianxin Li.

The paper reinterprets conventional input transformation-based transfer attacks as model transformation-based attacks. During adversarial example generation, a transformation is not only an input augmentation. When it is composed with a surrogate model, it forms a transformed surrogate model. Optimizing adversarial examples over these transformed surrogate models can improve transferability across unseen target models.

This codebase implements **MoTA**, a Model Transformation-based Attack. The core implementation is:

```text
transferattack/input_transformation/mota.py
```

Use this code only for authorized robustness evaluation, reproducible research, or academic experiments.

## Method Overview

`MoTA` computes the gradient on the original surrogate model, then repeatedly samples composed transformations and accumulates gradients from the transformed surrogate models. The averaged gradient is used with momentum to update the adversarial perturbation.

The implementation uses two transformation pools:

- `transform_num <= 20`: lightweight in-domain pool with `resizedpad`, `random_crop`, and `shift`.
- `transform_num > 20`: full pool with `rotate`, `blockshuffle`, `resizedpad`, `random_crop`, `ssm`, `shift`, and `ide`.

This design supports the paper's efficiency-effectiveness trade-off study: a smaller number of transformations reduces running time, while a larger number of transformations expands the transformed surrogate model space and usually improves transferability.

## Project Structure

```text
mota/
|-- main.py
|-- requirements.txt
|-- LICENSE
|-- run_transform_sweep.sh
`-- transferattack/
    |-- __init__.py
    |-- attack.py
    |-- utils.py
    `-- input_transformation/
        |-- __init__.py
        `-- mota.py
```

Execution flow:

```text
main.py
-> transferattack.load_attack_class("mota")
-> transferattack/input_transformation/mota.py::MoTA
-> transferattack/attack.py::Attack
-> transferattack/utils.py
```

## Installation

Python 3.8 to 3.10 and a CUDA GPU are recommended. The current entry point explicitly moves models and images with `.cuda()`, so generation and evaluation require a GPU.

```bash
conda create -n mota python=3.10 -y
conda activate mota
pip install -r requirements.txt
```

`requirements.txt` uses PyTorch 1.12.1 with CUDA 11.6 by default and pins `numpy<2` for compatibility with this PyTorch generation. If your CUDA version is different, install the matching `torch` and `torchvision` first, then install `numpy<2 pandas pillow timm tqdm`.

## Data Format

The default input directory is `./data`:

```text
data/
|-- labels.csv
`-- images/
    |-- ILSVRC2012_val_00000001.JPEG
    `-- ...
```

For untargeted attacks, `labels.csv` should contain at least:

```csv
filename,label
ILSVRC2012_val_00000001.JPEG,65
```

For targeted attacks, add a `targeted_label` column:

```csv
filename,label,targeted_label
ILSVRC2012_val_00000001.JPEG,65,970
```

`label` and `targeted_label` should use 0-based ImageNet class IDs consistent with torchvision/timm classification outputs.

## Generate Adversarial Examples

Example command:

```bash
python main.py \
  --attack mota \
  --model resnet18 \
  --input_dir ./data \
  --output_dir ./results/resnet18_mota \
  --batchsize 8 \
  --epoch 10 \
  --transform_num 2000 \
  --eps 16 \
  --GPU_ID 0
```

Key arguments:

- `--attack mota`: the only attack registered in this minimal repository.
- `--model resnet18`: source surrogate model. Any supported torchvision or timm model name can be used.
- `--epoch 10`: number of optimization iterations, passed to `MoTA(num_iter=...)`.
- `--transform_num 2000`: number of random transformations sampled per iteration.
- `--eps 16`: perturbation budget. The code converts it to `16/255`.
- `--batchsize`: computation is heavy when `transform_num=2000`; start with 4 or 8 for memory testing.
- `--alpha`: kept for entry-point compatibility. The current attack uses `eps / epoch` internally.

Generated adversarial images are saved to `--output_dir` with the same filenames as the input images.

## Transform Number Sweep

The script [run_transform_sweep.sh](run_transform_sweep.sh) explores the running-time and attack-performance trade-off under different `transform_num` settings. It is intended for the experiment corresponding to **Figure 9: Pareto frontier of our proposed MoTA**. The script reduces the transformation number to measure how attack success rate changes with running time.

By default, it runs the following surrogate models:

```text
resnet18, densenet121, inception_v3, vit_base_patch16_224, vit_small_patch16_224
```

For each model, it tests:

```text
1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000
```

Each setting first generates adversarial examples and records training time, then immediately runs evaluation:

```bash
bash run_transform_sweep.sh
```

Default output root:

```text
adv_data/
```

Logs:

```text
adv_data/logs/
```

Runtime summary:

```text
adv_data/runtime_summary.csv
```

The `train_seconds` column records the raw training time for each `model + transform_num` setting. For Figure 9, normalize the running time by the MI-FGSM training time:

```text
normalized_time = train_seconds / train_seconds_of_MI_FGSM
```

You can override models, GPU, batch size, and other settings with environment variables:

```bash
MODELS_STR="resnet18 resnet50" GPU_ID=1 BATCHSIZE=4 EVAL_BATCHSIZE=16 bash run_transform_sweep.sh
```

If `MODELS_STR` includes `vit_small_patch16_224`, set `VIT_SMALL_CHECKPOINT` before running the sweep script.

## Evaluation

After adversarial examples are generated, run:

```bash
python main.py \
  --eval \
  --input_dir ./data \
  --output_dir ./results/resnet18_mota \
  --batchsize 32 \
  --GPU_ID 0
```

Evaluation mode reads labels from `--input_dir/labels.csv`, reads generated images from `--output_dir`, and prints attack success rates on the evaluation models.

## vit_small_patch16_224 Checkpoint

If you use `vit_small_patch16_224`, set the checkpoint path with:

```bash
export VIT_SMALL_CHECKPOINT=/path/to/pytorch_model.bin
```

Windows PowerShell:

```powershell
$env:VIT_SMALL_CHECKPOINT = "<path-to>\pytorch_model.bin"
```

This variable is not needed for torchvision models such as `resnet18` or `resnet50`.



## Acknowledgements

This repository is built on the TransferAttack framework and reuses its attack base class, data loading utilities, model loading utilities, and evaluation workflow. The transformation sampling implementation in `mota.py` is used to reproduce MoTA-related experiments. Several transformation ideas are related to prior transfer-based adversarial attack methods including DIM, SSM, SIA, BSR, L2T, and OPS.

## Citation

If you use this repository, please cite the corresponding paper. Replace venue pages and DOI with the final publication metadata after publication:

```bibtex
@inproceedings{mota_2026,
  title     = {So-Called Input Transformation-Based Attacks May be Model Transformation-Based Attacks},
  author    = {Hu, Yang and Yang, Tao and He, Yuheng and Sun, Qingyun and Bi, Xiuli and Xiao, Bin and Li, Jianxin},
  booktitle = {Proceedings of ACM Multimedia},
  year      = {2026},
  note      = {Manuscript}
}
```

If you use the TransferAttack framework components, please also cite TransferAttack:

```bibtex
@article{wang2026devling,
  title   = {Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation},
  author  = {Xiaosen Wang and Zhijin Ge and Bohan Liu and Zheng Fang and Fengfan Zhou and Ruixuan Zhang and Shaokang Wang and Yuyang Luo},
  journal = {arXiv preprint arXiv:2602.23117},
  year    = {2026}
}
```

## References

- TransferAttack codebase: https://github.com/Trustworthy-AI-Group/TransferAttack
- Xiaosen Wang, Zhijin Ge, Bohan Liu, Zheng Fang, Fengfan Zhou, Ruixuan Zhang, Shaokang Wang, and Yuyang Luo. 2026. *Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation*. arXiv:2602.23117.
- Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and Jianguo Li. 2018. *Boosting Adversarial Attacks with Momentum*. CVPR.
- Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu Wang, Zhou Ren, and Alan L. Yuille. 2019. *Improving Transferability of Adversarial Examples with Input Diversity*. CVPR.
- Junhua Zou, Zhisong Pan, Junyang Qiu, Xin Liu, Ting Rui, and Wei Li. 2020. *Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs, Diversity-Ensemble and Region Fitting*. ECCV.
- Yuyang Long, Qilong Zhang, Boheng Zeng, Lianli Gao, Xianglong Liu, Jian Zhang, and Jingkuan Song. 2022. *Frequency Domain Model Augmentation for Adversarial Attack*. ECCV.
- Xiaosen Wang, Zeliang Zhang, and Jianping Zhang. 2023. *Structure Invariant Transformation for Better Adversarial Transferability*. ICCV.
- Kunyu Wang, Xuanran He, Wenxuan Wang, and Xiaosen Wang. 2024. *Boosting Adversarial Transferability by Block Shuffle and Rotation*. CVPR.
- Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, and Chenliang Xu. 2024. *Learning to Transform Dynamically for Better Adversarial Transferability*. CVPR.
- Yu Guo, Weiquan Liu, Qingshan Xu, Shijun Zheng, Shujun Huang, Yu Zang, Siqi Shen, Chenglu Wen, and Cheng Wang. 2025. *Boosting Adversarial Transferability through Augmentation in Hypothesis Space*. CVPR.
