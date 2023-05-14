from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 1
    max_epochs = 100
    save_interval = 0.5
    eval_interval = 4
    save_epoch_end = True
    n_samples = 10  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = -1.  # no clipping
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    dataset = 'celeba'
    data_root = './data/CelebA'
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
