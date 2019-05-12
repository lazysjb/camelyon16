import os

import numpy as np
from openslide import open_slide
import pandas as pd

from params import args
from utils.config import IMG_PARTITION_PARAMS, NON_GRAY_RATIO_THRESHOLD
from utils.slide_utils import get_inference_file_name, read_slide


HEATMAP_OUTPUT_ZOOM_LEVEL = 5


def _extract_row_col_id_from_file_name(file_name):
    components = file_name.split('_split_')[-1].split('.')[0].split('_')
    return (int(components[0]), int(components[1]))


def _extract_row_col_ids_from_file_name_list(file_name_list):
    temp = [_extract_row_col_id_from_file_name(f) for f in file_name_list]

    row_ids, col_ids = list(zip(*temp))
    row_ids = np.sort(np.unique(row_ids).astype(int))
    col_ids = np.sort(np.unique(col_ids).astype(int))

    return (row_ids, col_ids)


def _get_mask_range_for_pred(pred_row_id,
                             pred_col_id,
                             pred_width,
                             pred_height,
                             zoom_factor):
    x_start = int((pred_row_id * pred_height) / zoom_factor)
    x_end = int(((pred_row_id + 1) * pred_height) / zoom_factor)

    y_start = int((pred_col_id * pred_width) / zoom_factor)
    y_end = int(((pred_col_id + 1) * pred_width) / zoom_factor)

    return x_start, x_end, y_start, y_end


def generate_original_mask(img_id):
    """Generate original slide and mask image"""
    slide_img_filename = 'tumor_{}.tif'.format(img_id)
    mask_img_filename = 'tumor_{}_mask.tif'.format(img_id)

    slide = open_slide(os.path.join(args.raw_source_dir, slide_img_filename))
    mask = open_slide(os.path.join(args.raw_source_dir, mask_img_filename))
    heatmap_level_dim = mask.level_dimensions[HEATMAP_OUTPUT_ZOOM_LEVEL]

    slide_img = read_slide(slide,
                           x=0,
                           y=0,
                           level=HEATMAP_OUTPUT_ZOOM_LEVEL,
                           width=heatmap_level_dim[0],
                           height=heatmap_level_dim[1])

    mask_img = read_slide(mask,
                          x=0,
                          y=0,
                          level=HEATMAP_OUTPUT_ZOOM_LEVEL,
                          width=heatmap_level_dim[0],
                          height=heatmap_level_dim[1])
    mask_img = mask_img[:, :, 0]
    return slide_img, mask_img


def generate_prediction_mask(inference_model_name,
                             partition_option,
                             data_split_type,
                             img_id):
    """Generate image with prediction to be used in heatmap"""
    inference_file_name = get_inference_file_name(inference_model_name,
                                                  partition_option,
                                                  data_split_type)
    inference_dir = os.path.join(args.output_data_dir, 'inference')
    inference_df = pd.read_pickle(os.path.join(inference_dir,
                                               inference_file_name))

    # Force non-tissue region prediction to be 0
    filter_mask = (inference_df['is_roi'] == 1) & \
                  (inference_df['non_gray_ratio'] > NON_GRAY_RATIO_THRESHOLD)
    inference_df.loc[~filter_mask, 'y_pred_prob'] = 0
    inference_df = inference_df[inference_df['img_id'] == img_id].copy()

    slide_img_filename = 'tumor_{}.tif'.format(img_id)
    mask_img_filename = 'tumor_{}_mask.tif'.format(img_id)

    slide = open_slide(os.path.join(args.raw_source_dir, slide_img_filename))
    mask = open_slide(os.path.join(args.raw_source_dir, mask_img_filename))
    heatmap_level_dim = mask.level_dimensions[HEATMAP_OUTPUT_ZOOM_LEVEL]

    slide_img = read_slide(slide,
                           x=0,
                           y=0,
                           level=HEATMAP_OUTPUT_ZOOM_LEVEL,
                           width=heatmap_level_dim[0],
                           height=heatmap_level_dim[1])

    mask_img = read_slide(mask,
                          x=0,
                          y=0,
                          level=HEATMAP_OUTPUT_ZOOM_LEVEL,
                          width=heatmap_level_dim[0],
                          height=heatmap_level_dim[1])
    mask_img = mask_img[:, :, 0]

    pred_mask = np.zeros_like(mask_img).astype(float)

    partition_settings = IMG_PARTITION_PARAMS[partition_option]
    partition_zoom_level = partition_settings['zoom_level']
    partition_width = partition_settings['partition_width']
    partition_height = partition_settings['partition_height']
    downsample_factors = slide.level_downsamples
    zoom_factor = int(downsample_factors[HEATMAP_OUTPUT_ZOOM_LEVEL] /
                      downsample_factors[partition_zoom_level])

    for idx, row in inference_df.iterrows():
        pred_row_id, pred_col_id = _extract_row_col_id_from_file_name(row['file_name'])
        x_start, x_end, y_start, y_end = _get_mask_range_for_pred(pred_row_id,
                                                                  pred_col_id,
                                                                  partition_width,
                                                                  partition_height,
                                                                  zoom_factor)
        pred_mask[x_start:x_end, y_start:y_end] = row['y_pred_prob']

    return slide_img, pred_mask
