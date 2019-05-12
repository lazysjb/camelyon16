from glob import glob
import os
from PIL import Image

import numpy as np
from openslide import open_slide
import pandas as pd
from tqdm import tqdm

from params import args
from utils.config import (
    IMG_PARTITION_PARAMS, NON_GRAY_RATIO_THRESHOLD, PARTITION_META_INFO_FILENAME,
    ROI_MASK_FILEFORMAT, ROI_ZOOM_LEVEL
)
from utils.image_preprocess import calc_non_gray_ratio_for_image, read_slide_partitions
from utils.slide_utils import get_meta_info_with_train_test_split


def create_image_partition(partition_option):
    """Create image patches with a specified partition option"""
    slide_meta_df = get_meta_info_with_train_test_split()

    partition_settings = IMG_PARTITION_PARAMS[partition_option]
    partition_root_dir = os.path.join(args.img_data_dir,
                                      partition_settings['partition_dir'])
    print('Creating image partitions under directory: {}'.format(partition_root_dir))

    zoom_level = partition_settings['zoom_level']
    partition_width = partition_settings['partition_width']
    partition_height = partition_settings['partition_height']

    if not os.path.exists(partition_root_dir):
        os.mkdir(partition_root_dir)

    for split_type in ['train', 'val', 'test']:
        partition_sub_dir = os.path.join(partition_root_dir, split_type)
        if not os.path.exists(partition_sub_dir):
            os.mkdir(partition_sub_dir)

        for img_type in ['slide', 'mask']:
            partition_sub_sub_dir = os.path.join(partition_sub_dir, img_type)
            if not os.path.exists(partition_sub_sub_dir):
                os.mkdir(partition_sub_sub_dir)

    for idx, row in tqdm(slide_meta_df.iterrows()):
        split_type = row['type']
        partition_sub_dir = os.path.join(partition_root_dir, split_type)

        slide_img_filepath = os.path.join(args.raw_source_dir, row['slide_img_filename'])
        mask_img_filepath = os.path.join(args.raw_source_dir, row['mask_img_filename'])
        slide = open_slide(slide_img_filepath)
        mask = open_slide(mask_img_filepath)

        img_id = row['img_id']
        slide_img_partition_file_prefix = os.path.join(partition_sub_dir,
                                                       'slide',
                                                       'tumor_slide_{}_split'.format(img_id))
        mask_img_partition_file_prefix = os.path.join(partition_sub_dir,
                                                      'mask',
                                                      'tumor_mask_{}_split'.format(img_id))

        _ = read_slide_partitions(
            slide,
            zoom_level,
            partition_width=partition_width,
            partition_height=partition_height,
            save_mode=True,
            save_file_prefix=slide_img_partition_file_prefix
        )
        _ = read_slide_partitions(
            mask,
            zoom_level,
            partition_width=partition_width,
            partition_height=partition_height,
            save_mode=True,
            is_mask=True,
            save_file_prefix=mask_img_partition_file_prefix
        )

        slide.close()
        mask.close()


def _get_slide_img_file_paths_for_img_id(img_dir,
                                         img_id):
    img_file_prefix = 'tumor_slide_{}_split'.format(img_id)
    return glob(os.path.join(img_dir, img_file_prefix) + '*.png')


def _get_matching_mask_for_img_file(img_filepath):
    basename = os.path.basename(img_filepath)

    file_path_components = img_filepath.split('/')
    mask_basename = basename.replace('slide', 'mask').replace('.png', '.npy')
    file_path_components[-2] = 'mask'
    file_path_components[-1] = mask_basename
    return '/'.join(file_path_components)


def _extract_row_col_id_from_file_name(file_name):
    components = file_name.split('_split_')[-1].split('.')[0].split('_')
    return (int(components[0]), int(components[1]))


def create_partition_meta(partition_option):
    """For each partitioned patch, create meta info"""
    slide_meta_df = get_meta_info_with_train_test_split()

    partition_settings = IMG_PARTITION_PARAMS[partition_option]
    partition_root_dir = os.path.join(args.img_data_dir,
                                      partition_settings['partition_dir'])
    partition_meta_dir = os.path.join(partition_root_dir, 'meta')
    partition_zoom_level = partition_settings['zoom_level']
    partition_width = partition_settings['partition_width']
    partition_height = partition_settings['partition_height']

    print('Creating image partition meta info under directory: {}'.format(
        partition_meta_dir))

    if not os.path.exists(partition_meta_dir):
        os.mkdir(partition_meta_dir)

    result = []

    for idx, row in tqdm(slide_meta_df.iterrows()):
        split_type = row['type']
        partition_sub_dir = os.path.join(partition_root_dir, split_type)
        img_id = row['img_id']

        slide_img_filename = 'tumor_{}.tif'.format(img_id)
        slide = open_slide(os.path.join(args.raw_source_dir, slide_img_filename))
        roi_downsample_factor = slide.level_downsamples[ROI_ZOOM_LEVEL]
        partition_downsample_factor = slide.level_downsamples[partition_zoom_level]
        downsample_factor = int(roi_downsample_factor / partition_downsample_factor)
        roi_mask_dir = os.path.join(args.img_data_dir, 'roi', 'opening_mask')
        roi_mask_file_path = os.path.join(roi_mask_dir,
                                          ROI_MASK_FILEFORMAT.format(img_id))
        roi_mask = np.load(roi_mask_file_path)
        slide.close()

        img_file_paths = _get_slide_img_file_paths_for_img_id(
            os.path.join(partition_sub_dir, 'slide'),
            img_id
        )

        for img_file_path in img_file_paths:
            img_basename = os.path.basename(img_file_path)
            mask_file_path = _get_matching_mask_for_img_file(img_file_path)
            mask_pixel_count = np.load(mask_file_path).sum()
            label = int(mask_pixel_count > 0)

            img_arr = np.asarray(Image.open(img_file_path))
            non_gray_ratio = calc_non_gray_ratio_for_image(img_arr)

            row_idx, col_idx = _extract_row_col_id_from_file_name(img_file_path)
            x_start = row_idx * partition_height // downsample_factor
            x_end = x_start + partition_height // downsample_factor
            y_start = col_idx * partition_width // downsample_factor
            y_end = y_start + partition_width // downsample_factor

            opening_patch = roi_mask[x_start:x_end, y_start:y_end]
            is_roi = int(opening_patch.sum() > 1)

            result.append({
                'img_id': img_id,
                'file_name': img_basename,
                'label': label,
                'non_gray_ratio': non_gray_ratio,
                'is_non_gray': int(non_gray_ratio > NON_GRAY_RATIO_THRESHOLD),
                'is_roi': is_roi
            })

    result_df = pd.DataFrame(result)

    save_path = os.path.join(partition_meta_dir,
                             PARTITION_META_INFO_FILENAME)
    result_df.to_json(save_path)
    print('Saved output in {}'.format(save_path))

    return result_df
