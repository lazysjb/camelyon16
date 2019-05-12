import argparse
import os


def get_args():
    """Get user input args from cmd"""
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--raw_source_dir',
        default=os.path.expanduser(
            '~/Personal/Columbia/Applied_DL/Camelyon_Project/data/source_data'))
    arg('--meta_data_dir',
        default=os.path.expanduser(
            '~/Personal/Columbia/Applied_DL/Camelyon_Project/data/test_dir'))
    arg('--img_data_dir',
        default=os.path.expanduser(
            '~/Personal/Columbia/Applied_DL/Camelyon_Project/data/test_dir/input_data'))
    arg('--output_data_dir',
        default=os.path.expanduser(
            '~/Personal/Columbia/Applied_DL/Camelyon_Project/data/test_dir/output_data'))

    arg('--img_partition_option', default='zoom_1_256_256')

    input_args = parser.parse_known_args()[0]

    return input_args


args = get_args()
