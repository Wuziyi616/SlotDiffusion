from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotAttention-Diffusion'

    # training settings
    gpus = 1
    max_epochs = 50
    save_interval = 0.5
    eval_interval = 2
    save_epoch_end = True
    n_samples = 8  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = -1.  # no clipping
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    movi_level = 'Solid'
    dataset = 'steve_movi'
    data_root = './data/MOVi'
    n_sample_frames = 1
    frame_offset = 1  # no offset
    video_len = 24
    load_mask = False
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'dVAE'
    resolution = (128, 128)
    vocab_size = 4096  # codebook size

    # temperature for gumbel softmax
    # decay from 1.0 to 0.1 in the first 15% of total steps
    init_tau = 1.
    final_tau = 0.1
    tau_decay_pct = 0.15

    # loss settings
    recon_loss_w = 1.
