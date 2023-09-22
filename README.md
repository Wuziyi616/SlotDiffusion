# SlotDiffusion

[**SlotDiffusion: Unsupervised Object-Centric Learning with Diffusion Models**](https://slotdiffusion.github.io/)<br/>
[Ziyi Wu](https://wuziyi616.github.io/),
Jingyu Hu*,
Wuyue Lu*,
[Igor Gilitschenski](https://tisl.cs.utoronto.ca/author/igor-gilitschenski/),
[Animesh Garg](https://animesh.garg.tech/)<br/>
_[NeurIPS'23](https://openreview.net/forum?id=ETk6cfS3vk) |
[GitHub](https://github.com/Wuziyi616/SlotDiffusion) |
[arXiv](https://arxiv.org/abs/2305.11281) |
[Project page](https://slotdiffusion.github.io/)_

**Unsupervised Video Object Segmentation**

<table style="width: 50%;">
  <tr>
    <td style="width: 33%; text-align: center;"> Video </td>
    <td style="width: 33%; text-align: center;"> GT&nbsp;&nbsp;&nbsp;&nbsp; </td>
    <td style="width: 33%; text-align: center;"> Ours&nbsp;&nbsp; </td>
  </tr>
  <tr>
    <td colspan="3">
      <img src="./assets/MOVi-D-seg-578.gif" alt="MOVi-D seg" width="100%">
    </td>
  </tr>
  <tr>
    <td colspan="3">
      <img src="./assets/MOVi-E-seg-31.gif" alt="MOVi-E seg" width="100%">
    </td>
  </tr>
</table>

**Slot-based Image Editing**

<img src="./assets/CLEVRTex-edit.png"  width="70%">

## Introduction

This is the official PyTorch implementation for paper: [SlotDiffusion: Unsupervised Object-Centric Learning with Diffusion Models]().
The code contains:

-   SOTA unsupervised object-centric models, Slot Attention, SAVi, SLATE, STEVE, and **SlotDiffusion**
-   Unsupervised object segmentation, image/video reconstruction, compositional generation on 6 datasets
-   Video prediction and VQA on Physion dataset
-   Scale up to real-world datasets: PASCAL VOC and COCO

## Update

-   2023.9.21: The paper is accepted by NeurIPS 2023 as a Spotlight presentation!
-   2023.5.24: Initial code release!

## Installation

Please refer to [install.md](docs/install.md) for step-by-step guidance on how to install the packages.

## Experiments

**This codebase is tailored to [Slurm](https://slurm.schedmd.com/documentation.html) GPU clusters with preemption mechanism.**
For the configs, we mainly use A40 with 40GB memory (though many experiments don't require so much memory).
Please modify the code accordingly if you are using other hardware settings:

-   Please go through `scripts/train.py` and change the fields marked by `TODO:`
-   Please read the config file for the model you want to train.
    We use DDP with multiple GPUs to accelerate training.
    You can use less GPUs to achieve a better memory-speed trade-off

### Dataset Preparation

Please refer to [data.md](docs/data.md) for dataset downloading and pre-processing.

### Reproduce Results

Please see [benchmark.md](docs/benchmark.md) for detailed instructions on how to reproduce our results in the paper.

## Possible Issues

See the troubleshooting section of [nerv](https://github.com/Wuziyi616/nerv#possible-issues) for potential issues.

Please open an issue if you encounter any errors running the code.

## Citation

Please cite our paper if you find it useful in your research:

```
@article{wu2023slotdiffusion,
  title={SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models},
  author={Wu, Ziyi and Hu, Jingyu and Lu, Wuyue and Gilitschenski, Igor and Garg, Animesh},
  journal={NeurIPS},
  year={2023}
}
```

## Acknowledgement

We thank the authors of [Slot Attention](https://github.com/google-research/google-research/tree/master/slot_attention), [slot_attention.pytorch](https://github.com/untitled-ai/slot_attention), [SAVi](https://github.com/google-research/slot-attention-video/), [SLATE](https://github.com/singhgautam/slate), [STEVE](https://github.com/singhgautam/steve), [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion), [DPM-Solver](https://github.com/LuChengTHU/dpm-solver), [DINOSAUR](https://github.com/amazon-science/object-centric-learning-framework), [MaskContrast](https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation) and [SlotFormer](https://github.com/Wuziyi616/SlotFormer) for opening source their wonderful works.

## License

SlotDiffusion is released under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions about the code, please contact Ziyi Wu dazitu616@gmail.com
