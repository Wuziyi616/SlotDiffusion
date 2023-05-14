# Dataset Preparation

All datasets should be downloaded or soft-linked to `./data/`.
Or you can modify the `data_root` value in the config files.

## CLEVRTex

Please download CLEVRTex from their [project page](https://www.robots.ox.ac.uk/~vgg/data/clevrtex/) to `./data/CLEVRTex/`.
Specifically, you need to download 5 files: `ClevrTex (part 1, 4.7 GB)` to `ClevrTex (part 5, 4.7 GB)`.
They will be saved as `clevrtex_full_part1.tar.gz` to `clevrtex_full_part5.tar.gz`.

Unzip them with:

```shell
cat clevrtex_full_part*.tar.gz | tar -xzvf
```

You will get a directory named `./data/CLEVRTex/clevrtex_full/`.

## CelebA

We follow [torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html) to download and process it.
You can directly call `torchvision` to download it with:

```python
from torchvision.datasets import CelebA

dataset = CelebA('./data/CelebA', download=True)
```

Make sure you get a directory named `./data/CelebA/celeba/`.

## MOVi-D/E

Please use the provided download script [download_movi.py](../scripts/data_utils/download_movi.py).

Download MOVi-D with:

```shell
python download_movi.py --out_path ./data/MOVi --level d --image_size 128
```

Download MOVi-E with:

```shell
python download_movi.py --out_path ./data/MOVi --level e --image_size 128
```

This will save the datasets to `./data/MOVi/MOVi-D/` and `./data/MOVi/MOVi-E/`.

## MOVi-Solid/Tex

Download their `.tar.gz` files from [Google Drive](https://drive.google.com/drive/folders/1R-2M4V1MeFu5Ycig1ofynbxxmLKc3wgM) and unzip them.

Please rename them to `MOVi-Solid/` and `MOVi-Tex/`, and put them under `./data/MOVi/`.

## Physion

Please download Physion from their github [repo](https://github.com/cogtoolslab/physics-benchmarking-neurips2021#downloading-the-physion-dataset).
Specifically, we only need 2 files containing videos and label files.
The HDF5 files containing additional vision data like depth map, segmentation masks are not needed.

-   Download `PhysionTest-Core` (the 270 MB one) with the [link](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip), and unzip it to a folder named `PhysionTestMP4s`
-   Download `PhysionTrain-Dynamics` (the 770 MB one) with the [link](https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/PhysionTrainMP4s.tar.gz), and unzip it to a folder named `PhysionTrainMP4s`
-   Download the labels for the readout subset [here](https://github.com/cogtoolslab/physics-benchmarking-neurips2021/blob/master/data/readout_labels.csv), and put it under `PhysionTrainMP4s`

To speed up data loading, we want to extract frames from videos.
We extract all the videos under `PhysionTrainMP4s/` and `PhysionTestMP4s/*/mp4s-redyellow/`.
Please run the provided script `python scripts/data_utils/physion_video2frames.py`.
You can modify a few parameters in that file such as `data_root`, number of process to parallelize `NUM_WORKERS`.

## PASCAL VOC 2012

We use the `trainaug` subset which is widely adopted in previous unsupervised segmentation works.
Please download the processed dataset from [Google Drive](https://drive.google.com/file/d/1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH/view) (credit to this great [repo](https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation), where we also borrow the VOC dataloader code).

Unzip the downloaded `tgz` file. We do not need the folders with `saliency` in the name.
Please only take `images/`, `SegmentationClass/`, `SegmentationClassAug/`, `sets/` folders and place them under `./data/VOC/`.
Finally, we also need the instance segmentation masks for evaluation.
Please download this [file](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), unzip it, take the `VOCdevkit/VOC2012/SegmentationObject/` folder and place it under `./data/VOC/`.

## MS COCO 2017

Please download the data from their [website](https://cocodataset.org/#download).
Specifically, we need `2017 Train images [118K/18GB]`, `2017 Val images [5K/1GB]`, `2017 Train/Val annotations [241MB]`.

Unzip them and you will get 2 images folders `train2017/` and `val2017/`, and 2 annotation files `instances_train2017.json` and `instances_val2017.json`.
Please put the image folders under `./data/COCO/images/`, and the annotations json files under `./data/COCO/annotations/`.

## Summary

**The `data` directory should look like this:**

```
data/
├── CLEVRTex/
│   ├── clevrtex_full/
│   │   ├── 0/  # folder with images and other annotations (not used here)
│   │   ├── 1/
•   •   •
•   •   •
│   │   └── 49/
├── CelebA/
│   ├── celeba/
│   │   ├── img_align_celeba/  # lots of images
│   │   └── list_eval_partition.txt  # data split
├── MOVi/
│   ├── MOVi-D/
│   │   ├── train/
│   │   │   ├── 00000000/  # folder with video frames and per-frame masks
│   │   │   ├── 00000001/
•   •   •   •
•   •   •   •
│   │   ├── validation/
│   │   └── test/
│   ├── MOVi-E/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── MOVi-Solid/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── MOVi-Tex/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
├── Physion
│   ├── PhysionTestMP4s/
│   │   ├── Collide/  # 8 scenarios
│   │   ├── Contain/
•   •   •
•   •   •
│   │   ├── Support/
│   │   └── labels.csv  # test subset labels
│   ├── videos/
│   │   ├── Collide_readout_MP4s/  # 8 scenarios x 2 subsets (training, readout)
│   │   ├── Collide_training_MP4s/
•   •   •
•   •   •
│   │   ├── Support_readout_MP4s/
│   │   ├── Support_training_MP4s/
│   │   └── readout_labels.csv  # readout subset labels
├── VOC/
│   ├── images/  # lots of images
│   ├── SegmentationClass/  # lots of masks
│   ├── SegmentationClassAug/
│   ├── SegmentationObject/
│   └── sets/  # data split
├── COCO/
│   ├── images/
│   │   ├── train2017/  # lots of images
│   │   ├── val2017/
│   │── annotations/
│   │   ├── instances_train2017.json
└   └   └── instances_val2017.json
```
