import math
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np      # noqa: E402


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


def read_slide_partition_file(file_path):
    """Read partitioned slide file"""
    if not file_path.endswith('.png'):
        raise ValueError('Slide partition file expected to be png')
    return np.asarray(Image.open(file_path))


def read_mask_partition_file(file_path):
    """Read partitioned mask file - note that this is single channel"""
    if not file_path.endswith('.npy'):
        raise ValueError('Slide mask file expected to be npy')
    return np.load(file_path)


def get_slide_filename_from_image_id(image_id):
    """Get slide filename"""
    return 'tumor_{}.tif'.format(image_id)


def get_mask_filename_from_image_id(image_id):
    """Get mask filename"""
    return 'tumor_{}_mask.tif'.format(image_id)
