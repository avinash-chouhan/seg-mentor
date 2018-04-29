import os, sys
import argparse
from pascal_voc import get_augmented_pascal_image_annotation_filename_pairs,\
                       convert_pascal_berkeley_augmented_mat_annotations_to_png
from tf_records import write_image_annotation_pairs_to_tfrecord

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def main(voc_path, sbd_path, tfrec_path):
    convert_pascal_berkeley_augmented_mat_annotations_to_png(sbd_path)
    # Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
    overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
                    get_augmented_pascal_image_annotation_filename_pairs(pascal_root=voc_path,
                                                                         pascal_berkeley_root=sbd_path,
                                                                         mode=2)

    write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                             tfrecords_filename=tfrec_path+'/validation.tfrecords')

    write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                             tfrecords_filename=tfrec_path+'/training.tfrecords')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transform VOCdata to TFrecord",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', '-p', type=str,
                        default='/data/',
                        help='the root path to the dir where all the data is')
    args = parser.parse_args()
    voc_path = os.path.join(args.path, 'VOCdevkit/VOC2012')
    sbd_path = os.path.join(args.path, 'SBD/benchmark_RELEASE')
    tfrec_path = os.path.join(args.path, 'TFrec' )
    main(voc_path, sbd_path, tfrec_path)

