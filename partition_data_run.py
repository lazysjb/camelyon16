from utils.image_partition import create_image_partition, create_partition_meta
from params import args

partition_option = args.img_partition_option
print('Creating partitions with option: {}'.format(partition_option))

# Step 1: Partition Data
print('\nCreating image partitions...')
create_image_partition(partition_option)

# Step 2: Create meta info on the partitioned images
print('\nCreating meta info for partitioned images...')
create_partition_meta(partition_option)
