import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import numpy as np
from osgeo import gdal


def save_tif(img, geo, path):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(path, img.shape[1], img.shape[0], img.shape[2], gdal.GDT_Byte)

    dataset.SetGeoTransform(geo[0])
    dataset.SetProjection(geo[1])

    for i in range(img.shape[2]):
        dataset.GetRasterBand(i+1).WriteArray(img[:, :, i]) 

    dataset.FlushCache()
    dataset = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for idx,  val_data in enumerate(val_loader):
        logger.info('Processing {}/{}'.format(idx, len(val_loader)))
        diffusion.feed_data(val_data)
        diffusion.test()
        visuals = diffusion.get_current_visuals(need_LR=False)

        for bidx, sr_img in enumerate(visuals['SR']):
            sr_img = sr_img.numpy()
            sr_img = np.transpose(sr_img, (1, 2, 0))
            min_max = (-1, 1)
            sr_img = np.clip(sr_img, *min_max)
            sr_img = (sr_img - min_max[0]) / (min_max[1] - min_max[0])
            sr_img = (sr_img * 255.0).round().astype(np.uint8)

            save_tif(
                sr_img,
                val_set.get_geo_ref(int(val_data['Index'][bidx])),
                '{}/{}_{}_{}_sr.tif'.format(result_path, current_step, idx, bidx)
            )

    # merge result
    os.system('gdal_merge.py -o {}/img.tif {}/*.tif'.format(opt['path']['experiments_root'], result_path))
