import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--raw_source_dir',
        default=os.path.expanduser(
            '~/Personal/Columbia/Applied_DL/Camelyon_Project/data/source_data'))
    arg('--meta_data_dir',
        default=os.path.expanduser(
            '~/Personal/Columbia/Applied_DL/Camelyon_Project/data/test_dir'))

    input_args = parser.parse_args()

    return input_args


args = get_args()
