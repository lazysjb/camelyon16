import math
import os
from PIL import Image

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np      # noqa: E402
from openslide import open_slide    # noqa: E402
from skimage.color import rgb2gray  # noqa: E402

from params import args     # noqa: E402
from utils.config import (
    ROI_CONTOUR_FILEFORMAT, ROI_MASK_FILEFORMAT, ROI_ZOOM_LEVEL)    # noqa: E402
from utils.slide_utils import get_meta_info_with_train_test_split, read_slide   # noqa: E402


def read_slide_partitions(slide,
                          level,
                          partition_width=256,
                          partition_height=256,
                          include_padding=False,
                          is_mask=False,
                          show_plot=False,
                          save_mode=False,
                          save_file_prefix=None):
    """ Read slide in partitioned images of defined width / height

    Args:
        slide: openslide object
        level: zoom level number
        partition_width: width of each partitioned image
        partition_height: height of each partitioned image
        include_padding: if True, include paddings in the last partition and show
            image. If False, cut off the last partition
        is_mask: if True, only show the first channel
        show_plot: if True, plot the partitions
        save_mode: if True, save partitioned files to jpeg
        save_file_prefix: if save_mode is True, partitions are saved with save_file_prefix

    Returns:
        List of partitioned numpy arrays (RGB)

    """
    if save_mode:
        if save_file_prefix is None:
            raise ValueError('save_file_prefix must be defined if save_mode is set to True')

        if is_mask:
            # save as numpy array
            save_path = save_file_prefix + '_{row_id}_{col_id}.npy'
        else:
            # save as png image
            save_path = save_file_prefix + '_{row_id}_{col_id}.png'
        print('Saving under the file name patterns: ', save_path)

    slide_height, slide_width = slide.level_dimensions[level]
    downsample_factor = int(slide.level_downsamples[level])

    if include_padding:
        n_cols = math.ceil(slide_height / partition_height)
        n_rows = math.ceil(slide_width / partition_width)
    else:
        n_cols = slide_height // partition_height
        n_rows = slide_width // partition_width

    partitions = []

    for i in range(n_rows):
        for j in range(n_cols):
            x = j * partition_width * downsample_factor
            y = i * partition_height * downsample_factor

            partition_image = read_slide(slide,
                                         x=x,
                                         y=y,
                                         level=level,
                                         width=partition_width,
                                         height=partition_height)

            # Sanity check - original image is always 3 channels
            assert len(partition_image.shape) == 3
            assert partition_image.shape[-1] == 3

            # If it is mask, select only the first channel
            if is_mask:
                partition_image = partition_image[:, :, 0]
                if save_mode:
                    # save as numpy array
                    np.save(save_path.format(row_id=i, col_id=j), partition_image)
            else:
                if save_mode:
                    # save as PNG file
                    im = Image.fromarray(partition_image)
                    im.save(save_path.format(row_id=i, col_id=j), format='PNG')

            if show_plot:
                partitions.append(partition_image)

    if show_plot:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
        for idx in range(len(partitions)):
            i = idx // n_cols
            j = idx % n_cols
            axes[i, j].imshow(partitions[idx])
        plt.tight_layout()
        plt.show()

    return partitions


def read_slide_partitions_with_overlap(slide,
                                       level,
                                       partition_width=256,
                                       partition_height=256,
                                       offset=(64, 64),
                                       overlap=128,
                                       is_mask=False,
                                       save_file_prefix=None):
    """Read slide partitions with specified offset and overlap"""
    if is_mask:
        # save as numpy array
        save_path = save_file_prefix + '_{row_id}_{col_id}_overlap_{overlap}_offset_{offset}.npy'
    else:
        # save as png image
        save_path = save_file_prefix + '_{row_id}_{col_id}_overlap_{overlap}_offset_{offset}.png'
    print('Saving under the file name patterns: ', save_path)

    slide_height, slide_width = slide.level_dimensions[level]
    downsample_factor = int(slide.level_downsamples[level])
    offset_x, offset_y = offset

    j = 0
    MAX_COUNT = 100000
    counter = 0

    x_start_init = offset_x * downsample_factor

    x_start = x_start_init
    x_end = x_start + partition_height * downsample_factor

    y_start_init = offset_y * downsample_factor

    while x_end < slide_height * downsample_factor:

        i = 0
        y_start = y_start_init
        y_end = y_start + partition_width * downsample_factor

        while y_end < slide_width * downsample_factor:
            partition_image = read_slide(slide,
                                         x=x_start,
                                         y=y_start,
                                         level=level,
                                         width=partition_width,
                                         height=partition_height)

            # Adding +1 to i, j for the file_names so that the zoom levels match up easier
            if is_mask:
                partition_image = partition_image[:, :, 0]
                np.save(save_path.format(row_id=i+1,
                                         col_id=j+1,
                                         overlap=overlap,
                                         offset=offset_x), partition_image)
            else:
                im = Image.fromarray(partition_image)
                im.save(save_path.format(row_id=i+1,
                                         col_id=j+1,
                                         overlap=overlap,
                                         offset=offset_x), format='PNG')

            # Sanity check - original image is always 3 channels
            assert len(partition_image.shape) == 3
            assert partition_image.shape[-1] == 3

            i += 1
            y_start = y_start_init + (partition_width - overlap) * i * downsample_factor
            y_end = y_start + partition_width * downsample_factor

            counter += 1
            if counter > MAX_COUNT:
                raise ValueError('Unexpectedly large number of iterations!')

        j += 1
        x_start = x_start_init + (partition_height - overlap) * j * downsample_factor
        x_end = x_start + partition_height * downsample_factor

    return


def calc_non_gray_ratio_for_image(image, intensity_threshold=0.8):
    """Calculate non gray ratio for a patch of image"""
    im_gray = rgb2gray(image / 255.)
    non_gray_mask = (im_gray <= intensity_threshold)
    return non_gray_mask.mean()


def _get_normal_image_contours(cont_img, rgb_image):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_rgb_image_array = np.array(rgb_image)

    line_color = (255, 0, 0)
    cv2.drawContours(contours_rgb_image_array, contours, -1, line_color, 3)
    return contours_rgb_image_array


def get_roi_mask_and_contour_for_img_array(slide_img):
    """Create ROI mask and contour for given slide image"""
    hsv = cv2.cvtColor(slide_img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([30, 30, 30])
    upper_red = np.array([200, 200, 200])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((50, 50), dtype=np.uint8)
    image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))

    open_kernel = np.ones((30, 30), dtype=np.uint8)
    image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close),
                                                  cv2.MORPH_OPEN, open_kernel))
    contour_rgb = _get_normal_image_contours(np.array(image_open), slide_img)
    return contour_rgb, np.array(image_open)


def create_roi_mask_and_contour_for_all_images():
    """Create Region of Interest Mask for all slides"""
    slide_meta_df = get_meta_info_with_train_test_split()
    print('Saving ROI contour and mask info under {}'.format(
        os.path.join(args.img_data_dir, 'roi')))

    for idx, row in slide_meta_df.iterrows():
        img_id = row['img_id']
        slide_img_filename = 'tumor_{}.tif'.format(img_id)

        slide = open_slide(os.path.join(args.raw_source_dir, slide_img_filename))
        roi_level_dim = slide.level_dimensions[ROI_ZOOM_LEVEL]
        slide_img = read_slide(slide,
                               x=0,
                               y=0,
                               level=ROI_ZOOM_LEVEL,
                               width=roi_level_dim[0],
                               height=roi_level_dim[1])

        contour_rgb, img_opening_mask = get_roi_mask_and_contour_for_img_array(slide_img)
        contour_pil = Image.fromarray(contour_rgb, 'RGB')

        contour_img_dir = os.path.join(args.img_data_dir, 'roi', 'contour')
        contour_img_file_path = os.path.join(contour_img_dir,
                                             ROI_CONTOUR_FILEFORMAT.format(img_id))
        roi_mask_dir = os.path.join(args.img_data_dir, 'roi', 'opening_mask')
        roi_mask_file_path = os.path.join(roi_mask_dir,
                                          ROI_MASK_FILEFORMAT.format(img_id))

        contour_pil.save(contour_img_file_path, format='JPEG')
        np.save(roi_mask_file_path, img_opening_mask)
    return
