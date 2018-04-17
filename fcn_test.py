import tensorflow as tf
import os, sys, argparse
from matplotlib import pyplot as plt

import numpy as np
slim = tf.contrib.slim

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# (!!) needed for code inside fcn_arch, utils..
#sys.path.append("/home/finkel/hailo_repos/phase2-dl-research/slim_models/")
sys.path.append("../tf-models-hailofork/research/slim/")

import fcn_arch, utils

sys.path.append("/home/finkel/hailo_repos/phase2-dl-research")
import nmldk
from resnet.resnet_architect import ArchitectResNet18
ArchitectFCN = ArchitectResNet18


validation_records = '/data/pascal_augmented_berkely/validation.tfrecords'
training_records = '/data/pascal_augmented_berkely/training.tfrecords'
number_of_classes = 21


pascal_voc_lut = utils.pascal_voc.pascal_segmentation_lut()

# Be careful: after adaptation, network returns final labels and not logits
fcnfunc = utils.inference.adapt_network_for_any_size_input(fcn_arch.fcn, 32)

validation = True

# That's our slightly hacky way to (optionally) sample and visualize some segmentations..
# X_in_visualize_each_Xth_seg =1
X_as_in_visualize_each_Xth_seg = 3455

# ...assumes "mode 2" in utils.pascal_voc.get_augmented_pascal_image_annotation_filename_pairs()
#     was used on creation of the tfrecord file
VALIDATION_SET_SIZE = utils.pascal_voc.PASCAL12_VALIDATION_WO_BERKELEY_TRAINING  # 904


def test(image, annotation, predictions, checkpoint, iterator,
         num_images=VALIDATION_SET_SIZE, tensors_to_eval=[], callback=None):
    """
        iterate over the validation set, visualize each so and so and/or compute overall mIoU metric
    """
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

    # Take away the masked out values from evaluation
    weights = tf.to_float(tf.not_equal(annotation_batch_tensor, 255))

    # validations set annotations seem to have "22" as an additional "ambiguous/borederline pixels" class
    #  gotta patch this out
    if validation:
        annotation_batch_tensor = tf.cast(annotation_batch_tensor, tf.int32)
        annotation_batch_tensor = annotation_batch_tensor * \
           tf.cast(tf.less_equal(annotation_batch_tensor, number_of_classes-1), tf.int32) # zero out bad guys

    # Define the accuracy metric: Mean Intersection Over Union
    miou, update_op = slim.metrics.streaming_mean_iou(predictions=predictions,
                                                      labels=annotation_batch_tensor,
                                                      num_classes=number_of_classes,
                                                      weights=weights)

    initializer = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(initializer)
        sess.run(iterator.initializer)
        saver.restore(sess, checkpoint)

        for i in xrange(num_images):
            # Display the image and the segmentation result
            if (i+1) % X_as_in_visualize_each_Xth_seg == 0:
                image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, predictions, update_op])

                upsampled_predictions = pred_np.squeeze()
                plt.figure(figsize=(30, 10))
                plt.suptitle('image #{0}'.format(i))
                plt.subplot(131)
                plt.imshow(image_np)
                plt.subplot(132)
                utils.visualization.visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut, image_np)
                plt.subplot(133)
                utils.visualization.visualize_segmentation_adaptive(annotation_np.squeeze(), pascal_voc_lut, image_np)
                plt.show()
            else:
                _eval_res = sess.run([update_op]+tensors_to_eval)
                if callback:
                    callback(_eval_res[1:])
        res = sess.run(miou)
        if num_images > 50:
            print("Final mIoU: " + str(res))
            # print("Pascal VOC 2012 Restricted (RV-VOC12) Mean IU: " + str(res))


def get_data_feed():
    dataset = tf.contrib.data.TFRecordDataset([validation_records]).map(utils.tf_records.parse_record)  # .batch(1)
    if args.pixels:
        dataset = dataset.map(lambda img, ann:
                              utils.augmentation.nonrandom_rescale(img, ann, [args.pixels, args.pixels]))

    iterator = dataset.repeat().make_initializable_iterator()
    return iterator


def single_image_feed(image_path):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_shape = tf.to_float(tf.shape(image))[:2]
    image = tf.expand_dims(image, 0)

    scale = tf.reduce_min(args.pixels / input_shape)
    image = tf.image.resize_nearest_neighbor(image, tf.cast(tf.round(input_shape * scale), tf.int32))
    image = tf.image.resize_image_with_crop_or_pad(image, args.pixels, args.pixels)
    image = tf.reshape(image, [1, args.pixels, args.pixels, 3])
    annotation = tf.zeros([1, args.pixels, args.pixels, 1])

    dataset = tf.contrib.data.Dataset.from_tensor_slices((image, annotation))
    iterator = dataset.repeat().make_initializable_iterator()
    return iterator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a FCN(-based) segmentation net')
    parser.add_argument('--basenet', dest='basenet', type=str,
                        help='the base feature extractor',
                        default='mobilenet')
    parser.add_argument('--narrowdeconv', dest='narrowdeconv', type=bool,
                        help='if True use 1K->21 1x1 then 21->21 deconv rather than wide 1K->21 deconv',
                        default=False)
    parser.add_argument('--checkpoint32', dest='checkpoint32', type=str,
                        help='path to checkpoint of the FCN32',
                        default="tmp/resnet_v1_18_dynDiffLR_bs16/fcn32.ckpt")
    parser.add_argument('--fcn16', dest='fcn16', type=bool,
                        help='if True add the fcn16 skip connection',
                        default=False)
    parser.add_argument('--pixels', dest='pixels', type=int,
                        help='if not zero, normalize (interpolate&crop) image (and annotation)'
                             ' to pixels X pixels before inference',
                        default=0)
    parser.add_argument('--vizstep', dest='vizstep', type=int,
                        help='set to X < size(val.set) to draw visualization each X images',
                        default=5555)
    parser.add_argument('--hquant', dest='hquant', type=bool,
                        help='set to True to do a quantized run...',
                        default=False)
    parser.add_argument('--single_image', dest='single_image', type=str,
                        help='set to a path in order to run on a single image',
                        default=None)
    args = parser.parse_args()

    X_as_in_visualize_each_Xth_seg = args.vizstep

    if args.basenet not in ['vgg', 'resnet_v1_50', 'resnet_v1_18', 'inception_v1', 'mobilenet_v1']:
        raise Exception("Not yet supported feature extractor")

    # run_test(net_func_str=args.basenet, narrowdeconv=args.narrowdeconv, checkpoint=args.checkpoint32)

    tf.reset_default_graph()

    if args.single_image:
        iterator = single_image_feed(args.single_image)
        X_as_in_visualize_each_Xth_seg = 1
    else:
        iterator = get_data_feed()

    image, annotation = iterator.get_next()

    if not args.hquant:
        predictions, _, __ = fcnfunc(image_batch_tensor=tf.expand_dims(image, axis=0),
                                     number_of_classes=number_of_classes,
                                     is_training=False,
                                     net_func=args.basenet,
                                     narrowdeconv=args.narrowdeconv,
                                     fcn16=args.fcn16)
        test(image, annotation, predictions, args.checkpoint32, iterator)
    else: 
	print "Please contact Hailo to enter Early Access Program and gain access to Hailo-quantized version of this net"
