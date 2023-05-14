# Unsupervised Object-Centric Learning on Images

We evaluate different tasks on the following 4 image datasets:

-   `CLEVRTex`: object segmentation, image reconstruction, compositional generation
-   `CelebA`: image reconstruction, compositional generation
-   `VOC/COCO`: object segmentation

We will take `SlotDiffusion` on `CLEVRTex` for example.
The 2 baselines `Slot Attention` and `SLATE` follow similar steps.
To run on other datasets, simply replace the config file with the desired one.

## SlotDiffusion on CLEVRTex

SlotDiffusion training involves 2 steps: first train a VQ-VAE to discretize images into patch tokens, and then train a slot-conditioned Latent Diffusion Model (LDM) to reconstruct these tokens.

### Train VQ-VAE

Run the following command to train VQ-VAE:

```
python scripts/train.py --task img_based \
    --params slotdiffusion/img_based/configs/sa_ldm/vqvae_clevrtex_params-res128.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained VQ-VAE weight** as `pretrained/vqvae_clevrtex_params-res128.pth`.

### Train SlotDiffusion

Run the following command to train SlotDiffusion on VQ-VAE tokens:

```
python scripts/train.py --task img_based \
    --params slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128.py \
    --fp16 --cudnn
```

Alternatively, we provide **pre-trained SlotDiffusion weight** as `pretrained/sa_ldm_clevrtex_params-res128.pth`.

### Evaluate on Object Segmentation

**Go to `slotdiffusion/img_based/`**, and then run the following command to evaluate the object segmentation performance:

```
python test_seg.py --params configs/sa_ldm/sa_ldm_clevrtex_params-res128.py \
    --weight $WEIGHT \
    --bs 32  # optional, change to desired value
```

### Evaluate on Image Reconstruction

**Go to `slotdiffusion/img_based/`**, and then run the following command to evaluate the image reconstruction performance (we support DDP testing as reconstruction is slow, especially for SLATE):

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=29501 \
    test_recon.py --params configs/sa_ldm/sa_ldm_clevrtex_params-res128.py \
    --weight $WEIGHT \
    --bs 32  # optional, change to desired value
```

### Evaluation on Compositional Generation

**Go to `slotdiffusion/img_based/`**, and then run the following command to evaluate the image reconstruction performance (DDP to speed up testing as well):

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=29501 \
    test_comp_gen.py --params configs/sa_ldm/sa_ldm_clevrtex_params-res128.py \
    --weight $WEIGHT \
    --bs 32  # optional, change to desired value
```

**Note:**

-   The compositional generation implemented here is a simplied version, where we randomly compose slots within a batch to generate novel samples.
    According to our experiments, the FID result is close to the visual concept library method described in paper Section 3.3.
    Therefore, we implement it here to simplify the evaluation process
-   To compute the FID, you need to manually call the `pytorch-fid` package.
    Suppose you test the weight located at `xxx/model.pth`, we will save the GT images under `xxx/eval/gt_imgs/`, and the generated images under `xxx/eval/comp_imgs/`.
    Run `python -m pytorch_fid xxx/eval/gt_imgs xxx/eval/comp_imgs` to compute the FID
-   The reconstructed images after running `test_recon.py` will be saved under `xxx/eval/recon_imgs/`

## Baseline: Slot Attention

Slot Attention training does not require any pre-trained tokenizers.
You can train it with the provided [config files](../slotdiffusion/img_based/configs/sa/).

## Baseline: SLATE

Similar to SlotDiffusion, SLATE training consists of 2 steps: pre-train dVAE, and then train SLATE.
You can train it with the provided [config files](../slotdiffusion/img_based/configs/slate/).
