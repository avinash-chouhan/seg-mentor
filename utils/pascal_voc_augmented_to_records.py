import os, sys
from pascal_voc import get_augmented_pascal_image_annotation_filename_pairs,\
                       convert_pascal_berkeley_augmented_mat_annotations_to_png
from tf_records import write_image_annotation_pairs_to_tfrecord

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

pascal_root = '/data/VOCdevkit/VOC2012'
pascal_berkeley_root = '/data/pascal_augmented_berkely/benchmark_RELEASE'

convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root)

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
                get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root,
                                                                     pascal_berkeley_root=pascal_berkeley_root,
                                                                     mode=2)

# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here
write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='/data/pascal_augmented_berkely/validation.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='/data/pascal_augmented_berkely/training.tfrecords')