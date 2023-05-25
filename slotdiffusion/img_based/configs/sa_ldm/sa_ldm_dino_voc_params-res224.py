from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 1
    max_epochs = 500
    save_interval = 0.5
    eval_interval = max_epochs // 50
    save_epoch_end = False
    n_samples = 10  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4
    dec_lr = 2 * lr  # DDPM uses 2e-4, LDM even lower 1e-4
    weight_decay = 0.0
    clip_grad = 0.05
    warmup_steps_pct = 0.05

    # data settings
    dataset = 'voc'
    data_root = './data/VOC'
    load_anno = True
    norm_mean = 0.5
    norm_std = 0.5
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size
    num_workers = 8
    grad_accum_steps = 1  # LDM saves 4x memory

    # model configs
    model = 'SADiffusion'
    resolution = (224, 224)
    img_ch = 3

    # Slot Attention
    slot_size = 192
    slot_dict = dict(
        # DINOSAUR uses 6 slots on VOC
        num_slots=6,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=3,
    )

    # DINO encoder
    enc_dict = dict(
        dino='dino-vits8',
        enc_out_channels=slot_size,
        patch_size=8,  # downsample img by 8 or 16
        # True uses small sized dino (channel = 384)
        # False uses large sized dino (channel = 768)
        small_size=True,
        resolution=resolution,
    )

    # LDM Decoder
    latent_ch = 3
    vae_dict = dict(
        vae_type='VQVAE',
        enc_dec_dict=dict(
            resolution=resolution[0],
            in_channels=img_ch,
            z_channels=latent_ch,
            ch=64,  # base_channel
            ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
            num_res_blocks=2,
            attn_resolutions=[],
            out_ch=img_ch,
            dropout=0.0,
        ),
        vq_dict=dict(
            n_embed=4096,  # vocab_size
            embed_dim=latent_ch,
            percept_loss_w=1.0,
        ),
        vqvae_ckp_path='./pretrained/vqvae_voc_params-res224.pth',
    )

    unet_dict = dict(
        in_channels=latent_ch,  # latent feature in
        model_channels=128,  # >=64 can eliminate the `color-bias` problem
        out_channels=latent_ch,  # latent feature noise out
        num_res_blocks=2,
        attention_resolutions=(8, 4, 2),  # actually the downsampling factor
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        dims=2,  # 2D data
        use_checkpoint=False,  # LDM saves 4x memory
        num_head_channels=32,
        resblock_updown=False,  # usually False
        conv_resample=True,  # up/downsample followed by Conv
        transformer_depth=1,
        context_dim=slot_size,  # condition on slots
        n_embed=None,  # VQ codebook support for LDM
    )
    dec_dict = dict(
        resolution=tuple(res // 4 for res in resolution),
        vae_dict=vae_dict,
        unet_dict=unet_dict,
        use_ema=False,  # DDPM and LDM do use EMA
        diffusion_dict=dict(
            pred_target='eps',  # 'eps' or 'x0', predict noise or direct x0
            z_scale_factor=1.,  # 1.05
            timesteps=1000,
            beta_schedule="linear",
            # the one used in LDM
            linear_start=0.0015,
            linear_end=0.0195,
            cosine_s=8e-3,  # doesn't matter for linear schedule
            log_every_t=200,  # log every t steps in denoising sampling
            logvar_init=0.,
        ),
        conditioning_key='crossattn',  # 'concat'
        cond_stage_key='slots',
    )

    # loss configs
    loss_dict = dict(use_denoise_loss=True, )

    denoise_loss_w = 1.  # DM denoising loss weight
