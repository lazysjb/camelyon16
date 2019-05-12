from utils.image_preprocess import create_roi_mask_and_contour_for_all_images
from utils.slide_utils import (
    get_meta_info_for_all_slides, get_train_val_test_split)

# Step 1: Run high level meta info for all 21 slides
print('Running meta information for all slides...')
_ = get_meta_info_for_all_slides(save=True)

# Step 2: Define train / val / test set on slide level
print('\nSplitting slides to train / val / test ...')
_ = get_train_val_test_split(save=True)

# Step 3: Create ROI info
print('\nCreating ROI information ...')
create_roi_mask_and_contour_for_all_images()
