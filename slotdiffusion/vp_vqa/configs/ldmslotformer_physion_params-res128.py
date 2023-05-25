from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings, from slot former
    gpus = 2
    max_epochs = 25  # ~450k steps
    save_interval = 0.125  # save every 0.125 epoch
    eval_interval = 2  # evaluate every 2 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 8  # visualization after each epoch

    # optimizer settings, from DM
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4
    weight_decay = 0.0
    warmup_steps_pct = 0.05

    # data settings, Physion
    dataset = 'physion_slots_training'
    data_root = './data/Physion'
    slots_root = './data/Physion/slots/physion_training_slots.pkl'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 15 + 10  # train on video clips of 6 frames
    frame_offset = 3
    video_len = 150  # take the first 150 frames of each video
    train_batch_size = 128 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'LDMSlotFormer'
    resolution = (128, 128)
    img_ch = 3
    input_frames = 15  # burn-in frames

    # Slot Attention
    slot_size = 192
    num_slots = 8
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=2,
    )

    # rollout
    rollout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        history_len=input_frames,
        t_pe='sin',  # sine temporal P.E.
        slots_pe='',  # no slots P.E.
        # Transformer-related configs
        d_model=256,
        num_layers=12,
        num_heads=8,
        ffn_dim=256 * 4,
        norm_first=True,
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
        vqvae_ckp_path='./pretrained/vqvae_physion_params-res128.pth',
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
            z_scale_factor=1.,  # 1.235
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
        dec_ckp_path='./pretrained/savi_ldm_physion_params-res128.pth',
    )

    # loss configs
    loss_dict = dict(
        use_denoise_loss=False,
        use_img_recon_loss=False,
        rollout_len=n_sample_frames - rollout_dict['history_len'],
    )

    slot_recon_loss_w = 1.
    denoise_loss_w = 1.  # DM denoising loss weight
    img_recon_loss_w = 1.
