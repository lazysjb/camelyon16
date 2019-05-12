ALL_SLIDE_IDS = (
    '001', '002', '005', '012', '016',
    '019', '023', '031', '035', '057',
    '059', '064', '075', '078', '081',
    '084', '091', '094', '096', '101', '110'
)

ALL_SLIDE_META_INFO_FILENAME = 'all_slides_meta_info.pkl'
TRAIN_VAL_TEST_SPLIT_FILENAME = 'train_val_test_split.pkl'
PARTITION_META_INFO_FILENAME = 'partition_meta_info.json'

NON_GRAY_RATIO_THRESHOLD = 0.4

IMG_PARTITION_PARAMS = {
    'zoom_1_256_256': {
        'partition_dir': 'zoom_1_256_256_partition',
        'overlap': False,
        'zoom_level': 1,
        'partition_width': 256,
        'partition_height': 256,
    },

    'zoom_4_256_256': {
        'partition_dir': 'zoom_4_256_256_partition',
        'overlap': False,
        'zoom_level': 4,
        'partition_width': 256,
        'partition_height': 256,
    },
}

INFERENCE_FILE_MAPS = [
    {
        'model': 'vgg16_transfer',
        'partition': 'zoom_1_256_256',
        'split_type': 'test',
        'file_name': 'vgg_transfer_learn_zoom_1_256_256_test.pkl'
    },

    {
        'model': 'vgg16_transfer',
        'partition': 'zoom_1_256_256',
        'split_type': 'val',
        'file_name': 'vgg_transfer_learn_zoom_1_256_256_val.pkl'
    },

]
