# Unsupervised Object-Centric Learning on Videos

We evaluate object segmentation, video reconstruction, compositional generation on 4 video datasets: `MOVi-D`, `MOVi-E`, `MOVi-Solid`, `MOVi-Tex`.

We will take `SlotDiffusion` on `MOVi-E` for example.
The 2 baselines `SAVi` and `STEVE` follow similar steps.
To run on other datasets, simply replace the config file with the desired one.

## SlotDiffusion on MOVi-E

SlotDiffusion training involves 2 steps: first train a VQ-VAE to discretize frames into patch tokens, and then train a slot-conditioned Latent Diffusion Model (LDM) to reconstruct these tokens.

### Train VQ-VAE

Run the following command to train VQ-VAE:

```
python scripts/train.py --task video_based \
    --params slotdiffusion/video_based/configs/savi_ldm/vqvae_movie_params-res128.py \
    --fp16 --ddp --cudnn
```

Alternatively, we provide **pre-trained VQ-VAE weight** as `pretrained/vqvae_movie_params-res128.pth`.

### Train SlotDiffusion

Run the following command to train SlotDiffusion on VQ-VAE tokens:

```
python scripts/train.py --task video_based \
    --params slotdiffusion/video_based/configs/savi_ldm/savi_ldm_movie_params-res128.py \
    --fp16 --cudnn
```

Alternatively, we provide **pre-trained SlotDiffusion weight** as `pretrained/savi_ldm_movie_params-res128.pth`.

### Evaluate on Object Segmentation

Run the following command to evaluate the object segmentation performance:

```
python slotdiffusion/video_based/test_seg.py \
    --params slotdiffusion/video_based/configs/savi_ldm/savi_ldm_movie_params-res128.py \
    --weight $WEIGHT \
    --bs 32  \ # optional, change to desired value
    --seq_len -1  # i.e. full video length, can be changed
```

### Evaluate on Video Reconstruction

Run the following command to evaluate the video reconstruction performance (we support DDP testing as reconstruction is slow, especially for STEVE):

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=29501 \
    slotdiffusion/video_based/test_recon.py \
    --params slotdiffusion/video_based/configs/savi_ldm/savi_ldm_movie_params-res128.py \
    --weight $WEIGHT \
    --bs 1
```

**Note:** You can add the `--save_video` flag to only save a few videos for visualization, instead of testing over the entire dataset.

### Evaluation on Compositional Generation

Run the following command to evaluate the image reconstruction performance (DDP to speed up testing as well):

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=29501 \
    slotdiffusion/video_based/test_comp_gen.py \
    --params slotdiffusion/video_based/configs/savi_ldm/savi_ldm_movie_params-res128.py \
    --weight $WEIGHT \
    --bs 1
```

**Note:**

-   The compositional generation implemented here is a simplied version, where we randomly compose slots within a batch to generate novel samples.
    According to our experiments, the FVD result is close to the visual concept library method described in paper Section 3.3.
    Therefore, we implement it here to simplify the evaluation process
-   To compute the FVD, we adopt the implementation from [StyleGAN-V](https://github.com/universome/stylegan-v).
    Suppose you test the weight located at `xxx/model.pth`, we will save the GT videos under `xxx/eval/gt_vids/`, and the generated images under `xxx/eval/comp_vids/`.
    Please download the StyleGAN-V repo and run the following command instead that repo to compute the FVD:
    ```
    python src/scripts/calc_metrics_for_dataset.py \
        --real_data_path .../xxx/eval/gt_vids \
        --fake_data_path .../xxx/eval/comp_vids \
        --mirror 1 --gpus 1 --resolution 128 \
        --metrics fvd2048_16f --verbose 1 --use_cache 0
    ```
-   The reconstructed videos after running `test_recon.py` will be saved under `xxx/eval/recon_vids/`
-   You can add the `--save_video` flag to only save a few videos for visualization, instead of testing over the entire dataset.
    The videos will be saved under `xxx/vis/`

## Baseline: SAVi

SAVi training does not require any pre-trained tokenizers.
You can train it with the provided [config files](../slotdiffusion/video_based/configs/savi/).

## Baseline: STEVE

Similar to SlotDiffusion, STEVE training consists of 2 steps: pre-train dVAE, and then train STEVE.
You can train it with the provided [config files](../slotdiffusion/video_based/configs/steve/).
