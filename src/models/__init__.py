from .unet2d import TrackNetV2
from .resunet2d import ChangsTrackNet
from .monotrack import MonoTrack
from .hrnet import HRNet
from .deepball import DeepBall
from .ballseg import BallSeg
from .hrcnet import HRCNetForWASB
from .dw3dconv import DW3DConvForWASB

__factory = {
    'tracknetv2': TrackNetV2,
    'monotrack': MonoTrack,
    'restracknetv2': ChangsTrackNet,
    'hrnet': HRNet,
    'hrcnet': HRCNetForWASB,
    'dw3dconv': DW3DConvForWASB,
    'deepball': DeepBall,
    'ballseg': BallSeg
        }

def build_model(cfg):
    model_name = cfg['model']['name']
    if not model_name in __factory.keys():
        raise KeyError('invalid model: {}'.format(model_name ))
    if model_name=='tracknetv2' or model_name=='resunet2d' or model_name=='monotrack':
        frames_in  = cfg['model']['frames_in']
        frames_out = cfg['model']['frames_in']
        bilinear   = cfg['model']['bilinear']
        halve_channel = cfg['model']['halve_channel']
        model      = __factory[model_name]( frames_in*3, frames_out, bilinear=bilinear, halve_channel=halve_channel)
    elif model_name=='higher_hrnet' or model_name=='cls_hrnet' or model_name=='hrnet':
        model = __factory[model_name](cfg['model'])
    elif model_name=='hrcnet':
        mcfg = cfg["model"]
        in_channels = mcfg["in_channels"]
        out_channels = mcfg["out_channels"]
        high_channels = mcfg["high_channels"]
        low_channels = mcfg["low_channels"]

        num_stages = mcfg["num_stages"]
        high_block = mcfg["high_block"]
        low_block = mcfg["low_block"]
        num_high_blocks = mcfg["num_high_blocks"]
        num_low_blocks = mcfg["num_low_blocks"]
        upsample_mode = mcfg["upsample_mode"]
        down_cfg = mcfg.get("downsample", {})
        downsample_kwargs = {
            "num_blocks": down_cfg.get("num_blocks", 4),
            "expand_ratio": down_cfg.get("expand_ratio", 4.0),
            "use_se": down_cfg.get("use_se", True),
            "se_reduction": down_cfg.get("se_reduction", 4),
            "activation": down_cfg.get("activation", "hswish"),
            "bn_momentum": mcfg.get("bn_momentum", 0.1),
        }
        st_cfg = mcfg.get("spatial_transformer", {})
        transformer_kwargs = {
            "d_model": st_cfg.get("d_model", low_channels),
            "num_heads": st_cfg.get("num_heads", 8),
            "dim_ff": st_cfg.get("dim_ff", st_cfg.get("d_model", low_channels) * 4),
            "dropout": st_cfg.get("dropout", 0.1),
            "depth": st_cfg.get("depth", 2),
        }
        temporal_cfg = mcfg.get("temporal", {})
        temporal_type = temporal_cfg.get("type", "transformer")
        temporal_kwargs = temporal_cfg.get(temporal_type, {})
        model = HRCNetForWASB(
            in_channels=in_channels,
            out_channels=out_channels,
            high_channels=high_channels,
            low_channels=low_channels,
            num_stages=num_stages,
            high_block=high_block,
            low_block=low_block,
            num_high_blocks=num_high_blocks,
            num_low_blocks=num_low_blocks,
            upsample_mode=upsample_mode,
            downsample_kwargs=downsample_kwargs,
            transformer_kwargs=transformer_kwargs,
            temporal_type=temporal_type,
            temporal_kwargs=temporal_kwargs,
        )
    elif model_name=='dw3dconv':
        mcfg = cfg["model"]
        in_channels = mcfg["in_channels"]
        out_channels = mcfg["out_channels"]
        depths = tuple(mcfg.get("depths", [2, 2, 8, 2]))
        channels = tuple(mcfg.get("channels", [32, 64, 128, 256]))
        num_spatial_stages = mcfg.get("num_spatial_stages", None)
        bottleneck_depth = mcfg.get("bottleneck_depth", 2)
        model = DW3DConvForWASB(
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            channels=channels,
            num_spatial_stages=num_spatial_stages,
            bottleneck_depth=bottleneck_depth,
        )
    elif model_name=='restracknetv2':
        frames_in        = cfg['model']['frames_in']
        frames_out       = cfg['model']['frames_out']
        halve_channel    = cfg['model']['halve_channel']
        mode             = cfg['model']['mode']
        neck_channels    = cfg['model']['neck_channels']
        out_mid_channels = cfg['model']['out_mid_channels']
        blocks           = cfg['model']['blocks']
        channels         = cfg['model']['channels']
        model = __factory[model_name]( frames_in*3, 
                                       frames_out, 
                                       mode=mode, 
                                       halve_channel=halve_channel, 
                                       neck_channels=neck_channels, 
                                       out_mid_channels=out_mid_channels, 
                                       blocks=blocks, channels=channels)
    elif model_name=='deepball':
        frames_in              = cfg['model']['frames_in']
        frames_out             = cfg['model']['frames_out']
        class_out              = cfg['model']['class_out']
        block_channels         = cfg['model']['block_channels']
        block_maxpools         = cfg['model']['block_maxpools']
        first_conv_kernel_size = cfg['model']['first_conv_kernel_size']
        last_conv_kernel_size  = cfg['model']['last_conv_kernel_size']
        first_conv_stride      = cfg['model']['first_conv_stride']
        model = __factory[model_name]( frames_in*3, frames_out*class_out,
                                       block_channels=block_channels,
                                       block_maxpools=block_maxpools, 
                                       first_conv_kernel_size=first_conv_kernel_size, 
                                       first_conv_stride=first_conv_stride, 
                                       last_conv_kernel_size=last_conv_kernel_size)
    elif model_name=='ballseg':
        frames_in     = cfg['model']['frames_in']
        frames_out    = cfg['model']['frames_out']
        scale_factors = cfg['model']['scale_factors']
        backbone      = cfg['model']['backbone']
        model = __factory[model_name]( in_channels=frames_in*3, nclass=frames_out*1, backbone=backbone, scale_factors=scale_factors)

    return model

