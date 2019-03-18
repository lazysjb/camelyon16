import math

import matplotlib.pyplot as plt
import numpy as np


# Taken directly from
# https://github.com/random-forests/applied-dl/blob/master/project/starter-code.ipynb
# See https://openslide.org/api/python/#openslide.OpenSlide.read_region
# Note: x,y coords are with respect to level 0.
def read_slide(slide, x, y, level, width, height, as_float=False):
    """Read a region from openslide object

    Args:
        slide: openslide object
        x: left most pixel co-ord
        y: top most pixel co-ord
        level: zoom level number
        width:
        height:
        as_float:

    Returns:
        Numpy RGB array

    """
    im = slide.read_region((x, y), level, (width, height))
    im = im.convert('RGB')  # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im


def read_slide_partitions(slide,
                          level,
                          partition_width=256,
                          partition_height=256,
                          include_padding=False,
                          show_plot=False):
    """ Read slide in partitioned images of defined width / height

    Args:
        slide: openslide object
        level: zoom level number
        partition_width: width of each partitioned image
        partition_height: height of each partitioned image
        include_padding: if True, include paddings in the last partition and show
            image. If False, cut off the last partition
        show_plot: if True, plot the partitions

    Returns:
        List of partitioned numpy arrays (RGB)

    """
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


def get_slide_filename_from_image_id(image_id):
    """Get slide filename"""
    return 'tumor_{}.tif'.format(image_id)


def get_mask_filename_from_image_id(image_id):
    """Get mask filename"""
    return 'tumor_{}_mask.tif'.format(image_id)
