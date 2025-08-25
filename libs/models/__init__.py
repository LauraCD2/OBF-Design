__all__ = "BACKBONES"

from .networks import (
    SpectralNet,
    TSTransformerEncoder,
    Lstm,
    CNN,
    SpectralFormer,
    NHITS,
)

from .networks.config import (
    config_spectralnet,
    config_TSTransformer,
    config_lstm,
    config_cnn,
    config_spectralformer,
    config_nhits,
)

BACKBONES = dict(
    spectralnet=[SpectralNet, config_spectralnet],
    cnn=[CNN, config_cnn],
    lstm=[Lstm, config_lstm],
    transformer=[TSTransformerEncoder, config_TSTransformer],
    spectralformer=[SpectralFormer, config_spectralformer],
    nhits=[NHITS, config_nhits],
)
