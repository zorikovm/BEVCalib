# BEVCalib: LiDAR-Camera Calibration via Geometry-Guided Bird's-Eye View Representation

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246) [![Website](https://img.shields.io/badge/Website-BEVCalib-blue?style=for-the-badge)](https://cisl.ucr.edu/BEVCalib) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) [![Python](https://img.shields.io/badge/python-3.11-yellow?style=for-the-badge)](https://www.python.org) [![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

<hr style="border: 2px solid gray;"></hr>

## Getting Started

### Prerequistes

The code is built with following libraries:

- Python = 3.11
- Pytorch = 2.6.0
- cuda-toolkit = 11.8
- [spconv-cu118](https://github.com/traveller59/spconv)
- OpenCV
- pandas
- open3d
- transformers
- [deformable_attention](https://github.com/lucidrains/deformable-attention)
- tensorboard
- wandb

We recommend using the following command to install cuda-toolkit=11.8:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

After installing the above dependencies, please run the following command to install [bev_pool](https://github.com/mit-han-lab/bevfusion) operation
```bash
cd ./kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

We also provide a [Dockerfile](Dockerfile/Dockerfile) for easy setup, please execute the following command to build the docker image and install cuda extensions:
```bash
docker build -f Dockerfile/Dockerfile -t bevcalib .
docker run --gpus all -it -v$(pwd):/workspace bevcalib
### In the docker, run the following command to install cuda extensions
cd ./kitti-bev-calib/img_branch/bev_pool && python setup.py build_ext --inplace
```

### Evaluation
We provide a pretrained model for evaluation. Please download the pretrained model from [Google Drive]() and place it in the `./ckpt` directory. Please run the following command to evaluate the model:
```bash
```

### Training
We provide instructions to reproduce our results on the KITTI-Ododemetry dataset. Please run: 
```bash
python kitti-bev-calib/main_kitti.py --log_dir ./logs/kitti \
        --dataset_root YOUR_PATH_TO_KITTI/kitti-odemetry \
        --save_ckpt_per_epoches 50 --num_epochs 500 --label 20_1.5 --angle_range_deg 20 --trans_range 1.5 \
        --deformable 0 --bev_encoder 1 --batch_size 16 --xyz_only 1 --scheduler 1 --lr 1e-4 --step_size 100
```
You can change `--angle_range_deg` and `--trans_range` to train under different noise settings. You can also try to use `--pretrain_ckpt` to load a pretrained model for fine-tuning on your own dataset.

### Acknowledgement
BEVCalib is appreciated by the following great open-source projects: [BEVFusion](https://github.com/mit-han-lab/bevfusion?tab=readme-ov-file), [LCCNet](https://github.com/IIPCVLAB/LCCNet), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [spconv](https://github.com/traveller59/spconv), and [Deformable Attention](https://github.com/lucidrains/deformable-attention).

### Citation
```bibtex
```