# Benchmark

## Overview

We provide instructions on reproducing the results reported in the paper, including:

-   Fully unsupervised object-centric learning on CLEVRTex, CelebA, MOVi-D/E/Solid/Tex
-   Video prediction and VQA on Physion with SlotFormer
-   Unsupervised segmentation on PASCAL VOC and MS COCO with DINO

**We provide config files for SlotDiffusion AND baselines.**

### Pre-trained Weights

We provide pre-trained weights of our models on [Google Drive](https://drive.google.com/file/d/1PSElX2ucqqLuCjjl2_skM-7-qjwb2hWh/view?usp=sharing) to facilitate future research.
Please download the pre-trained weights `pretrained.zip` and unzip them to [`pretrained/`](../pretrained/).

## Basic Usage

**We provide a unified script [train.py](../scripts/train.py) to train all models used in this project.**
You should always call it in the root directory of this repo (i.e. calling `python scripts/train.py xxx`).

**All of the model training can be done by specifying the task it belongs to (we have 3 tasks: `img_based`, `video_based`, `vp_vqa`), providing a config file (called `params` here), and adding other args.**
Please check the config file for the number of GPUs and other resources (e.g. `num_workers` CPUs) before launching a training.

For example, to train a Slot Attention model on CLEVRTex dataset, simply run:

```
python scripts/train.py --task img_based --params slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128.py --fp16 --cudnn
```

Other arguments include:

-   `--weight`: resume training from this weight
-   `--ddp`: use DDP multi-GPU training (needed when using `>=2` GPUs)
-   `--fp16`: enable half-precision training (highly recommended)
-   `--cudnn`: enable cudnn benchmark (highly recommended)
-   `--local_rank`/`--local-rank`: required by DDP, don't change it

During training, model checkpoints and visualizations will be saved under `./checkpoint/$PARAMS/models/`.

When producing final results (e.g. image/video visualizations), we usually save them under the same directory as the model weight used to generate them.

See the docs of each task below for more details.

### Scripts

We provide helper scripts if you're running experiments on a Slurm GPU cluster.

You can use [sbatch_run.sh](../scripts/sbatch_run.sh) to automatically generate a sbatch file and submit the job to slurm.
Simply running:

```
GPUS=$NUM_GPU CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=$QOS \
    ./scripts/sbatch_run.sh $PARTITION $JOB_NAME \
    scripts/train.py none (if DDP then change `none` to `ddp`) --py_args...
```

For example, to train a Slot Attention model on CLEVRTex dataset, we can set `--py_args...` as (see the config file for the number of GPU/CPU to use)

```
--task img_based \
    --params slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128.py \
    --fp16 --cudnn
```

Then this will be equivalent to running the following command in CLI:

```
python scripts/train.py --task img_based \
    --params slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128.py \
    --fp16 --cudnn
```

We also provide a script to **submit multiple runs of the same experiment with different random seeds** to slurm.
This is important because unsupervised object-centric learning is sometimes unstable due to weight initializations.
According to our experiments, Slot Attention and SAVi have the largest variance, while SLATE, STEVE and SlotDiffusion are often stable.

To use the duplicate-run script [dup_run_sbatch.sh](../scripts/dup_run_sbatch.sh), simply do:

```
GPUS=$NUM_GPU CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=$QOS REPEAT=$NUM_REPEAT \
    ./scripts/dup_run_sbatch.sh $PARTITION $JOB_NAME \
    scripts/train.py none $PARAMS --py_args...
```

The other parts are really the same as `sbatch_run.sh`.
The only difference is that we need to input the config file `$PARAMS` separately, so that the script will make several copies to it, and submit different jobs.

Again if we want to train Slot Attention on CLEVRTex dataset, with `1` GPU and `1x8=8` CPUs, duplicating `3` times, on `rtx6000` partition, and in the name of `sa_clevrtex_params-res128`, simply run:

```
GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal REPEAT=3 \
    ./scripts/dup_run_sbatch.sh rtx6000 sa_clevrtex_params-res128 \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128.py \
    --task img_based --fp16 --ddp --cudnn
```

## Image Data

For unsupervised object-centric learning on images, see [img_based.md](./img_based.md).

## Video Data

For unsupervised object-centric learning on videos, see [video_based.md](./video_based.md).

## Video Prediction and VQA

For video prediction and VQA tasks on Physion dataset, see [vp_vqa.md](./vp_vqa.md).
