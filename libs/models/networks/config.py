config_spectralnet = dict(
    architecture=[256, 64, 64, 64, 32],
)

config_lstm = dict(
    n_layers=4,
    dropout_rate=0.1,
)

config_cnn = dict(
    conv_layers=[8, 8, 8, 32, 32],  # [64, 128, 256, 256, 256],
    latent_feat=32,
    kernel_size=3,
    pool_size=2,
    dropout_rate=0.1,
)

config_TSTransformer = dict(
    feat_dim=1,
    d_model=64,
    n_heads=4,
    num_layers=2,
    dim_feedforward=32,
    dropout=0.1,
    pos_encoding='fixed',
    activation='relu',
    norm='BatchNorm',
    freeze=False,
    patch_size=32, 
)

config_spectralformer = dict(
    image_size=1,
    near_band=16,
    dim=4,
    depth=3,
    heads=4,
    mlp_dim=8,
    dropout=0.1,
    emb_dropout=0.1,
    mode='CAF'
)

config_nhits = dict(
    feature_dim=32, 
    num_blocks=3, 
    mlp_units=256
)
