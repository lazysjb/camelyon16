"""
Define train / val / test at slide level
Get high level info for slides
"""
import os

import numpy as np
from openslide import open_slide
import pandas as pd
from sklearn.model_selection import train_test_split

from params import args
from utils.config import (
    ALL_SLIDE_IDS, ALL_SLIDE_META_INFO_FILENAME,
    INFERENCE_FILE_MAPS, TRAIN_VAL_TEST_SPLIT_FILENAME
)
from utils.image_preprocess import read_slide


DEFAULT_ZOOM_LEVEL_FOR_META_INFO = 5
N_VAL_SLIDES = 4
N_TEST_SLIDES = 4

np.random.seed(828)


def get_meta_info_for_all_slides(save=True):
    """Get high level meta info for all slides"""
    meta_infos = []

    for img_id in ALL_SLIDE_IDS:
        slide_img_filename = 'tumor_{}.tif'.format(img_id)
        mask_img_filename = 'tumor_{}_mask.tif'.format(img_id)

        slide = open_slide(os.path.join(args.raw_source_dir, slide_img_filename))
        mask = open_slide(os.path.join(args.raw_source_dir, mask_img_filename))

        _validate_slide_and_mask(slide, mask)

        level_dimensions = slide.level_dimensions
        level_downsamples = slide.level_downsamples

        level_dimension_for_default_zoom = level_dimensions[DEFAULT_ZOOM_LEVEL_FOR_META_INFO]

        # size of img in pixels
        img_size = level_dimension_for_default_zoom[0] * level_dimension_for_default_zoom[1]

        mask_img = read_slide(mask,
                              x=0,
                              y=0,
                              level=DEFAULT_ZOOM_LEVEL_FOR_META_INFO,
                              width=level_dimension_for_default_zoom[0],
                              height=level_dimension_for_default_zoom[1])
        mask_img = mask_img[:, :, 0]

        # size of mask in pixels
        mask_size = mask_img.sum()

        meta_info = {
            'img_id': img_id,
            'slide_img_filename': slide_img_filename,
            'mask_img_filename': mask_img_filename,
            'level_dimensions': level_dimensions,
            'level_downsamples': level_downsamples,
            'ref_level_img_size': img_size,
            'ref_level_mask_size': mask_size,
            'ref_level_mask_proportion': (mask_size / img_size) * 100,
            'ref_level': DEFAULT_ZOOM_LEVEL_FOR_META_INFO,
        }

        meta_infos.append(meta_info)

    meta_df = pd.DataFrame(meta_infos)

    if save:
        save_path = os.path.join(args.meta_data_dir,
                                 ALL_SLIDE_META_INFO_FILENAME)
        meta_df.to_pickle(save_path)
        print('Saved output in {}'.format(save_path))

    return meta_df


def _validate_slide_and_mask(slide, mask):
    """Sanity checks"""
    # In some cases slide level dimensions is more
    # Sanity check 1
    for i, dims in enumerate(mask.level_dimensions):
        assert slide.level_dimensions[i] == dims

    # Sanity check 2
    for i, dims in enumerate(slide.level_dimensions):
        assert (slide.level_downsamples[i] * np.array(dims)
                == np.array(slide.level_dimensions[0])).all()

    for i, dims in enumerate(mask.level_dimensions):
        assert (mask.level_downsamples[i] * np.array(dims)
                == np.array(mask.level_dimensions[0])).all()

    # Sanity check 3
    assert (slide.level_count - mask.level_count) in [0, 1]
    return


def get_meta_info_with_train_test_split():
    """Join slide level meta info with train/val/test info"""
    meta_info_path = os.path.join(args.meta_data_dir,
                                  ALL_SLIDE_META_INFO_FILENAME)
    meta_info = pd.read_pickle(meta_info_path)
    split_data_path = os.path.join(args.meta_data_dir,
                                   TRAIN_VAL_TEST_SPLIT_FILENAME)
    split_df = pd.read_pickle(split_data_path)
    joined_df = meta_info.merge(split_df, on='img_id')
    return joined_df


def get_train_val_test_split(save=True):
    meta_info_path = os.path.join(args.meta_data_dir,
                                  ALL_SLIDE_META_INFO_FILENAME)
    meta_info = pd.read_pickle(meta_info_path)
    meta_info = meta_info.sort_values('ref_level_mask_size', ascending=False).copy()
    meta_info['mask_size_category'] = np.where(
        meta_info['ref_level_mask_size'] < meta_info['ref_level_mask_size'].median(),
        'small', 'large')

    train_df, test_df = train_test_split(meta_info,
                                         test_size=N_TEST_SLIDES,
                                         stratify=meta_info['mask_size_category'])
    train_df, val_df = train_test_split(train_df,
                                        test_size=N_VAL_SLIDES,
                                        stratify=train_df['mask_size_category'])

    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()

    train_df['type'] = 'train'
    val_df['type'] = 'val'
    test_df['type'] = 'test'

    split_df = pd.concat([train_df, val_df, test_df])[['img_id', 'type']]

    if save:
        save_path = os.path.join(args.meta_data_dir,
                                 TRAIN_VAL_TEST_SPLIT_FILENAME)
        split_df.to_pickle(save_path)
        print('Saved output in {}'.format(save_path))

    return split_df


def get_inference_file_name(model_name,
                            data_partition,
                            split_type):
    def _is_match(x):
        return (x['model'] == model_name) and (x['partition'] == data_partition) and\
               (x['split_type'] == split_type)

    result = list(filter(lambda x: _is_match(x), INFERENCE_FILE_MAPS))
    if len(result) != 1:
        raise ValueError('Incorrect input params!')

    return result[0]['file_name']
