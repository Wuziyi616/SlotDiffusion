from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotAttention-Diffusion'

    # training settings
    # the training of this linear readout model is very fast
    gpus = 1
    max_epochs = 50
    eval_interval = 2
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 8  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay without Warmup
    optimizer = 'Adam'
    lr = 1e-3
    warmup_steps_pct = 0.  # no warmup

    # data settings
    dataset = 'physion_slots_label_readout'  # fit on readout set
    data_root = './data/Physion'
    slots_root = './data/Physion/slots/rollout-physion_readout_slots.pkl'
    tasks = ['all']
    n_sample_frames = 6  # useless
    frame_offset = 1  # take all video frames
    # we only take the first 75 frames of each video
    # due to error accumulation in the rollout, models trained on all frames
    # will overfit to some artifacts in later frames
    video_len = 75
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'PhysionReadout'
    resolution = (128, 128)

    # LDMslotformer on physion has 8 slots, each with 192 feature dimension.
    slot_size = 192
    num_slots = 8
    readout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        agg_func='max',
        feats_dim=slot_size,
    )

    vqa_loss_w = 1.
