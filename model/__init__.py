import logging
import torch
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    if opt['model'].get('compile', False):
        m.netG.denoise_fn = torch.compile(m.netG.denoise_fn, mode='reduce-overhead', fullgraph=True)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
