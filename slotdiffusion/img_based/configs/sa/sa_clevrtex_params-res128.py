from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 1
    max_epochs = 200
    save_interval = 2
    eval_interval = 5
    save_epoch_end = False
    n_samples = 10  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 4e-4
    weight_decay = 0.0
    clip_grad = -1  # SA paper doesn't say any clipping?
    warmup_steps_pct = 0.025  # SA paper uses 10k steps over 500k total steps

    # data settings
    dataset = 'clevrtex'
    data_root = './data/CLEVRTex'
    load_mask = True
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'SA'
    resolution = (128, 128)

    # Slot Attention
    slot_size = 192
    slot_dict = dict(
        num_slots=11,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=3,
    )

    # CNN Encoder
    enc_dict = dict(
        resnet='resnet18',
        use_layer4=False,  # False will downsample img by 8, True is 4
        # will use GN
        enc_out_channels=slot_size,
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(slot_size, 128, 128, 128, 128),
        dec_resolution=(8, 8),
        dec_ks=5,
        dec_norm='',
    )

    # loss configs
    loss_dict = dict(use_img_recon_loss=True, )

    img_recon_loss_w = 1.  # slots image reconstruction loss weight
