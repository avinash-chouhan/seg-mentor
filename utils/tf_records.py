# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_image_annotation_pairs_to_tfrecord(filename_pairs, tfrecords_filename, max_files=4e4):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    i = 0
    for img_path, annotation_path in filename_pairs:
        i += 1
        if i > max_files:
            break
        img = np.array(Image.open(img_path))
        annotation = np.array(Image.open(annotation_path))

        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())

    writer.close()


def read_image_annotation_pairs_from_tfrecord(tfrecords_filename):
    """Return image/annotation pairs from the tfrecords file.
    The function reads the tfrecords file and returns image
    and respective annotation matrices pairs.
    Parameters
    ----------
    tfrecords_filename : string
        filename of .tfrecords file to read from
    
    Returns
    -------
    image_annotation_pairs : array of tuples (img, annotation)
        The image and annotation that were read from the file
    """
    
    image_annotation_pairs = []

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])

        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])

        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        img = img_1d.reshape((height, width, -1))

        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

        # Annotations don't have depth (3rd dimension)
        # TODO: check if it works for other datasets
        annotation = annotation_1d.reshape((height, width))

        image_annotation_pairs.append((img, annotation))
    
    return image_annotation_pairs


def read_tfrecord_and_decode_into_image_annotation_pair_tensors(tfrecord_filenames_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue
        String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    
    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension
    annotation_shape = tf.stack([height, width, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    return image, annotation


annotation_dims=3
squeeze_annotation = False


def parse_record(serialized_example):
    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])

    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    if annotation_dims==2:
        annotation_shape = tf.stack([height, width])
    elif annotation_dims==3:
        annotation_shape = tf.stack([height, width, 1])
    else:
        raise Exception("WTF bad flag")

    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    if squeeze_annotation:  # possibly used for train. how it is different from annotation_dims=2 - kill me. but it works.
        annotation = tf.squeeze(annotation)
    return image, annotation


def tfrecordify_coco_stuff_things(imgsdir = '/data/coco',
                                  labelsdir='/data/coco/stuffthings_pixellabels',
                                  traindir='train2017', valdir='val2017'):
    '''
        Assumes stuff (pardon the pun) is downloaded & extracted
         from https://github.com/nightrome/cocostuff#downloads (3 top files)
    '''
    import os
    trainpairs = [(os.path.join(imgsdir, traindir, fname.replace('png', 'jpg')),
                   os.path.join(labelsdir, traindir, fname)) \
                    for fname in os.listdir(os.path.join(labelsdir, traindir))]

    valpairs = [(os.path.join(imgsdir, valdir, fname.replace('png', 'jpg')), \
                 os.path.join(labelsdir, valdir, fname)) \
                    for fname in os.listdir(os.path.join(labelsdir, valdir))]

    write_image_annotation_pairs_to_tfrecord(filename_pairs=trainpairs,
                                            tfrecords_filename=imgsdir+'/training.tfrecords')

    write_image_annotation_pairs_to_tfrecord(filename_pairs=valpairs,
                                             tfrecords_filename=imgsdir+'/validation.tfrecords')

def tfrecordify_camvid(rootdir = '/data/camvid'):
    '''
        Assumes stuff is e.g. cloned from https://github.com/alexgkendall/SegNet-Tutorial // CamVid
         - 11-class version encoded as the usual #class (not color-code)

        TODO 32-class version
    '''
    import os
    trainpairs = [(os.path.join(rootdir, 'train', fname),
                   os.path.join(rootdir, 'trainannot', fname)) \
                    for fname in os.listdir(os.path.join(rootdir, 'train'))]

    valpairs = [(os.path.join(rootdir, 'val', fname),
                   os.path.join(rootdir, 'valannot', fname)) \
                  for fname in os.listdir(os.path.join(rootdir, 'val'))]

    testpairs = [(os.path.join(rootdir, 'test', fname),
                 os.path.join(rootdir, 'testannot', fname)) \
                for fname in os.listdir(os.path.join(rootdir, 'test'))]

    write_image_annotation_pairs_to_tfrecord(filename_pairs=trainpairs,
                                            tfrecords_filename=rootdir+'/training.tfrecords')

    write_image_annotation_pairs_to_tfrecord(filename_pairs=testpairs,
                                             tfrecords_filename=rootdir+'/validation.tfrecords')

if __name__ == "__main__":
    tfrecordify_camvid()
