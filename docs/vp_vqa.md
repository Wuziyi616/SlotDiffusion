# Video Prediction and VQA on Physion

We experiment on video prediction and VQA tasks on Physion dataset in 3 steps:

-   Train a SlotDiffusion model to extract slots from Physion videos
-   Train a SlotFormer dynamics model to learn the scene dynamics, and evaluate video prediction results
-   Train a linear readout model over rollout slots to perform VQA

## Train SlotDiffusion and Extract Slots

As described in [video_based.md](./video_based.md), SlotDiffusion training consists of 2 steps: pre-train VQ-VAE, and then train a slot-conditioned LDM.

We skip detailed training steps and assume you are using our **pre-trained weight** as `pretrained/savi_ldm_physion_params-res128.pth`.

Then, we'll need to extract slots and save them.
Please run [extract_slots.py](../slotdiffusion/video_based/extract_slots.py):

```
python slotdiffusion/video_based/extract_slots.py \
    --params slotdiffusion/video_based/configs/savi_ldm/savi_ldm_physion_params-res128.py \
    --weight $WEIGHT \
    --subset $SUBSET \
    --save_path $SAVE_PATH  # e.g. './data/Physion/slots/$SUBSET_slots.pkl'
```

There are 3 subsets in Physion dataset: `training`, `readout`, `test`.
You need to extract slots from all of them (16G, 7.8G, 1.2G).

## Train SlotFormer

Train a SlotFormer model on extracted slots by running:

```
python scripts/train.py --task vp_vqa \
    --params slotdiffusion/vp_vqa/configs/ldmslotformer_physion_params-res128.py \
    --fp16 --cudnn --ddp
```

Alternatively, we provide **pre-trained SlotFormer weight** as `pretrained/savi_ldm_slotformer_physion_params-res128.pth`.

### Test Video Prediction

Run the following command to evaluate the video prediction performance:

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=29501 \
    slotdiffusion/vp_vqa/test_vp.py \
    --params slotdiffusion/vp_vqa/configs/ldmslotformer_physion_params-res128.py \
    --weight $WEIGHT \
    --bs 1
```

**Note:** You can add the `--save_video` flag to only save a few videos for visualization, instead of testing over the entire dataset.

## VQA

For the VQA task, we follow the official benchmark protocol as:

-   Train a dynamics model (SlotFormer) using `training` subset slots
-   Unroll slots on `readout` and `test` subset
-   Train a linear readout model on the unrolled `readout` subset slots + GT labels
-   Evaluate the linear readout model on the unrolled `test` subset slots + GT labels

### Unroll SlotFormer for VQA task

To unroll videos, please run [rollout_physion_slots.py](../slotdiffusion/vp_vqa/rollout_physion_slots.py):

```
python slotdiffusion/vp_vqa/rollout_physion_slots.py \
    --params slotdiffusion/vp_vqa/configs/ldmslotformer_physion_params-res128.py \
    --weight $WEIGHT \
    --subset $SUBSET \
    --save_path $SAVE_PATH  # e.g. './data/Physion/slots/rollout_$SUBSET_slots.pkl'
```

This will unroll slots for Physion videos, and save them into a `.pkl` file.
Please unroll for both `readout` and `test` subset.

### Train Linear Readout Model

Train a linear readout model on rollout slots in the `readout` subset by running:

```
python scripts/train.py --task vp_vqa \
    --params slotdiffusion/vp_vqa/configs/readout_physion_params.py \
    --fp16 --cudnn
```

This will train a readout model that takes in slots extracted from a video, and predict whether two object-of-interests contact during the video.

### Evaluate VQA Result

Finally, we can evaluate the trained readout model on rollout slots in the `test` subset, which is the number we report in the paper.
To do this, please run [test_physion_vqa.py](../slotdiffusion/vp_vqa/test_physion_vqa.py):

```
python slotdiffusion/vp_vqa/test_physion_vqa.py \
    --params slotdiffusion/vp_vqa/configs/readout_physion_params.py \
    --weight $WEIGHT
```

You can specify a single weight file to test, or a directory.
If the later is provided, we will test all the weights under that directory, and report the best accuracy of all the models tested.
You can also use the `--threshs ...` flag to specify different thresholds for binarizing the logits to 0/1 predictions.
Again, if multiple thresholds are provided, we will test all of them and report the best one.

**Note**: in our experiments, we noticed that the readout accuracy is very unstable.
So we usually train over three random seeds (using `dup_run_sbatch.sh`), and report the best performance among them.
