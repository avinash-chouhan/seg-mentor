import os, sys
from pascal_voc import get_augmented_pascal_image_annotation_filename_pairs,\
                       convert_pascal_berkeley_augmented_mat_annotations_to_png
from tf_records import write_image_annotation_pairs_to_tfrecord

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

'''
TO BE ABLE TO TRAIN & TEST, RUN THIS SCRIPT TO CREATE A NICE TRAIN/VAL TFRECORDS PAIR,
 (having changed the paths to where you have inputs and where you want the outputs)
BUT PREVIOUSLY, PLEASE: 

Download the "Berekely augmented Pascal a.k.a SBD:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
and untar under "pascal_berkeley_root"

Download the Pascal-VOC2012 from :
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
and untar under "pascal_root"

----- Dataset Notes ----
If you have an hour, do read pascal_voc.py whose author should be lauded. 
 (be it Daniil of http://warmspringwinds.github.io/ or one of his spiritual fathers..)

If you have a minute, do read the rant below, it should summarize the status&spirit of things.

...So, VOC PASCAL is kind of mess, the files train/val/test sets X3 challenges (07', 11', 12'), 
  with various intersections between the sets of different years/challenges..
then come additional annotations on original files... 

Eventually the widest segmentation-relevant stuff seems to come from Berkely project called SBD:
http://home.bharathh.info/pubs/codes/SBD/download.html
Ignore their own rant re non-intersecting train/test with 5623 images.
Here it's done better, using more for train and a bit less for test.

In any case, have no fear, train/val disjointness is ensured in line 546 of pascal_voc.py

'''

# PATHS:
pascal_root = '/data/VOCdevkit/VOC2012'
pascal_berkeley_root = '/data/pascal_augmented_berkely/benchmark_RELEASE'
where_to_put_records = '/data/pascal_augmented_berkely/'

convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root)


# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
                get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root,
                                                                     pascal_berkeley_root=pascal_berkeley_root,
                                                                     mode=2)

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename=where_to_put_records+'/validation.tfrecords')

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename=where_to_put_records+'/training.tfrecords')

