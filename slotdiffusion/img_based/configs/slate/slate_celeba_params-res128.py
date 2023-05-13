from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotAttention-Diffusion'

    # training settings
    gpus = 1  # A40
    max_epochs = 100
    save_interval = 0.5
    eval_interval = 2
    save_epoch_end = True
    n_samples = 10  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # 1e-4 for the main SLATE model
    dec_lr = 3e-4  # 3e-4 for the Transformer decoder
    weight_decay = 0.0
    clip_grad = 1.  # following the paper
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    dataset = 'celeba'
    data_root = './data/celeba_pytorch'
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'SLATE'
    resolution = (128, 128)

    # Slot Attention
    slot_size = 192
    slot_dict = dict(
        num_slots=4,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=3,
    )

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='',
    )

    # CNN Encoder
    enc_dict = dict(
        resnet='resnet18',
        use_layer4=False,  # True will downsample img by 8, False is 4
        # will use GN
        enc_out_channels=slot_size,
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=8,
        dec_num_heads=4,
        dec_d_model=slot_size,
    )

    # loss settings
    loss_dict = dict(
        use_img_recon_loss=False,  # additional img recon loss via dVAE decoder
    )

    token_recon_loss_w = 1.
    img_recon_loss_w = 1.
