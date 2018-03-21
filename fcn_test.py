import tensorflow as tf
import os, sys, argparse
from matplotlib import pyplot as plt


slim = tf.contrib.slim

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# (!!) needed for code inside fcn_arch, utils..
sys.path.append("/home/finkel/hailo_repos/phase2-dl-research/slim_models/")

import fcn_arch, utils

validation_records = '/data/pascal_augmented_berkely/validation.tfrecords'
training_records = '/data/pascal_augmented_berkely/training.tfrecords'
number_of_classes = 21


pascal_voc_lut = utils.pascal_voc.pascal_segmentation_lut()

# Be careful: after adaptation, network returns final labels and not logits
fcnfunc = utils.inference.adapt_network_for_any_size_input(fcn_arch.fcn, 32)

validation = True
#visualization_step =1
visualization_step = 3455


def test(image, annotation, predictions, checkpoint, iterator, take_first=None):
    """

    :param image:
    :param annotation:
    :param predictions:
    :param checkpoint:
    :param take_first:  use for small-set
    :param iterator:
    :return:
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

        # There are 904 images in restricted validation dataset
        for i in xrange(take_first or 904):

            image_np, annotation_np, pred_np, tmp = sess.run([image, annotation, predictions, update_op])

            # Display the image and the segmentation result
            if (i+1) % visualization_step == 0:
                upsampled_predictions = pred_np.squeeze()
                plt.figure(figsize=(30, 10))
                plt.subplot(131)
                plt.imshow(image_np)
                plt.subplot(132)
                utils.visualization.visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut, image_np)
                plt.subplot(133)
                utils.visualization.visualize_segmentation_adaptive(annotation_np.squeeze(), pascal_voc_lut, image_np)
                plt.show()

        res = sess.run(miou)
        print("Pascal VOC 2012 Restricted (RV-VOC12) Mean IU: " + str(res))


def run_test(net_func_str, checkpoint, tfrecord_filename=validation_records,
             narrowdeconv=False, take_first=None, is_training=False):

    tf.reset_default_graph()
    dataset = tf.contrib.data.TFRecordDataset([tfrecord_filename]).map(utils.tf_records.parse_record)  # .batch(1)
    iterator = dataset.repeat().make_initializable_iterator()
    image, annotation = iterator.get_next()
    predictions, fcn_32s_variables_mapping, _ = fcnfunc(image_batch_tensor=tf.expand_dims(image, axis=0),
                                                        number_of_classes=number_of_classes,
                                                        is_training=is_training,
                                                        net_func=net_func_str,
                                                        narrowdeconv=narrowdeconv)

    test(image, annotation, predictions, checkpoint, iterator, take_first=take_first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
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
    args = parser.parse_args()

    visualization_step = args.vizstep

    if args.basenet not in ['vgg', 'resnet_v1_50', 'resnet_v1_18', 'inception_v1', 'mobilenet_v1']:
        raise Exception("Not yet supported feature extractor")

    # run_test(net_func_str=args.basenet, narrowdeconv=args.narrowdeconv, checkpoint=args.checkpoint32)

    tf.reset_default_graph()
    dataset = tf.contrib.data.TFRecordDataset([validation_records]).map(utils.tf_records.parse_record)  # .batch(1)
    if args.pixels:
        dataset = dataset.map(lambda img, ann:
                          utils.augmentation.nonrandom_rescale(img, ann, [args.pixels, args.pixels]))

    iterator = dataset.repeat().make_initializable_iterator()
    image, annotation = iterator.get_next()
    predictions, fcn_32s_variables_mapping, _ = fcnfunc(image_batch_tensor=tf.expand_dims(image, axis=0),
                                                        number_of_classes=number_of_classes,
                                                        is_training=False,
                                                        net_func=args.basenet,
                                                        narrowdeconv=args.narrowdeconv,
                                                        fcn16=args.fcn16)

    test(image, annotation, predictions, args.checkpoint32, iterator) #, take_first=take_first)
