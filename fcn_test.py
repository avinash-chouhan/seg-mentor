import tensorflow as tf
import os, sys, argparse
from matplotlib import pyplot as plt

import numpy as np
slim = tf.contrib.slim

# (!!) needed for code inside fcn_arch, utils..
sys.path.append("../tf-models-hailofork/research/slim/")

tf_ver = float('.'.join(tf.__version__.split('.')[:2]))
if tf_ver >= 1.4:
    data = tf.data
else:
    data = tf.contrib.data

import fcn_arch, utils

validation_records = '/data/pascal_augmented_berkely/validation.tfrecords'
training_records = '/data/pascal_augmented_berkely/training.tfrecords'
number_of_classes = 21

pascal_voc_lut = utils.pascal_voc.pascal_segmentation_lut()

validation = True

# That's our slightly hacky way to (optionally) sample and visualize some segmentations..
# X_in_visualize_each_Xth_seg =1
X_as_in_visualize_each_Xth_seg = 3455

# ...assumes "mode 2" in utils.pascal_voc.get_augmented_pascal_image_annotation_filename_pairs()
#     was used on creation of the tfrecord file
VALIDATION_SET_SIZE = utils.pascal_voc.PASCAL12_VALIDATION_WO_BERKELEY_TRAINING  # 904 RV-VOC12


def test(image, annotation, predictions, checkpoint, iterator,
         num_images=VALIDATION_SET_SIZE, more_tensors_to_eval=[], callback=None):
    """
        iterate over the validation set, visualize each so and so and/or compute overall mIoU metric

        # Note tensors_to_eval & callback - a future placeholder for additional operation(s)
            to be run in the context of each image, without changing this function code..
    """
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

    # Prepare mask to exclude some pixels from evaluation - 
    # e.g. "padding margins" (=255) or "ambiguous" (=num_classes (additional class))
    weights = tf.to_float(tf.less_equal(annotation_batch_tensor, number_of_classes-1))
    # ...below line is redundant (because masked values shouldn't be used anyways..)
    #    but mean_iou won't work without it somehow..
    annotation_batch_tensor = annotation_batch_tensor*tf.cast(weights, tf.int32)

    miou, update_op = tf.metrics.mean_iou(predictions=predictions,
                                          labels=annotation_batch_tensor,
                                          num_classes=number_of_classes,
                                          weights=weights)

    conf_op = tf.confusion_matrix(tf.reshape(predictions, [-1]),
                                  tf.reshape(annotation_batch_tensor, [-1]),
                                  num_classes=number_of_classes,
                                  weights=tf.reshape(weights, [-1]))
    conf_mtx = np.zeros([number_of_classes]*2)

    initializer = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(initializer)
        sess.run(iterator.initializer)
        saver.restore(sess, checkpoint)

        for i in xrange(num_images):
            # note a bit different run for visualization vs. evaluation purposes...
            if (i+1) % X_as_in_visualize_each_Xth_seg == 0:
                # TODO implement as a default callback..
                image_np, annotation_np, pred_np, _, conf_tmp = \
                    sess.run([image, annotation, predictions, update_op, conf_op])
                visualize(image_np, annotation_np, pred_np.squeeze())
            else:
                _eval_res = sess.run([conf_op, update_op]+more_tensors_to_eval)
                conf_tmp = _eval_res[0]
                if callback: # a placeholder to inject more functionality w.o. changing this func
                    callback(_eval_res[2:])
            conf_mtx += conf_tmp

            final_miou = sess.run(miou)

        diag = conf_mtx.diagonal()
        err1 = conf_mtx.sum(axis=1) - conf_mtx.diagonal()
        err2 = conf_mtx.sum(axis=0) - conf_mtx.diagonal()
        iou = diag / (0.0 + diag + err1 + err2)

        print("Final mIoU for {0} images is {1:.2f}%".format(num_images, final_miou*100))
        print("\n\n ---- Breakup by class: ----")
        print(pascal_voc_lut)
        for i, x in enumerate(iou):
            print(pascal_voc_lut[i]+': {0:.2f}%'.format(x*100))
        print np.mean(iou) # a nice sanity check is to verify that it's the same as final_miou


def visualize(image_np, annotation_np, upsampled_predictions):
    plt.figure(figsize=(30, 10))
    plt.suptitle('image #{0}'.format(i))
    plt.subplot(131)
    plt.imshow(image_np)
    plt.subplot(132)
    utils.visualization.visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut, image_np)
    plt.subplot(133)
    utils.visualization.visualize_segmentation_adaptive(annotation_np.squeeze(), pascal_voc_lut, image_np)
    plt.show()

def get_data_feed(pixels):
    dataset = data.TFRecordDataset([validation_records]).map(utils.tf_records.parse_record)  # .batch(1)
    if pixels != 0:
        dataset = dataset.map(lambda img, ann:
                              utils.augmentation.nonrandom_rescale(img, ann, [pixels, pixels]))

    iterator = dataset.repeat().make_initializable_iterator()
    return iterator


def single_image_feed(image_path, pixels=None):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_shape = tf.to_float(tf.shape(image))[:2]
    image = tf.expand_dims(image, 0)

    pixels = pixels or tf.reduce_max(input_shape)
    scale = tf.reduce_min(pixels / input_shape)
    image = tf.image.resize_nearest_neighbor(image, tf.cast(tf.round(input_shape * scale), tf.int32))
    image = tf.image.resize_image_with_crop_or_pad(image, pixels, pixels)
    image = tf.reshape(image, [1, pixels, pixels, 3])
    annotation = tf.zeros([1, pixels, pixels, 1])

    dataset = data.Dataset.from_tensor_slices((image, annotation))
    iterator = dataset.repeat().make_initializable_iterator()
    return iterator

def segment_movie(fcnfunc, checkpoint, video_file_in, pixels=None):
    from PIL import Image

    image_ph = tf.placeholder(tf.int32) #, shape=)

    input_shape = tf.to_float(tf.shape(image_ph))[:2]
    image_t = tf.expand_dims(image_ph, 0)
    pixels = pixels or tf.reduce_max(input_shape)
    scale = tf.reduce_min(pixels / input_shape)

    inshape32 = tf.cast(tf.round(input_shape * scale), tf.int32)
    image_t = tf.image.resize_nearest_neighbor(image_t, inshape32)
    image_t = tf.image.resize_image_with_crop_or_pad(image_t, pixels, pixels)
    image_t3d = image_t = tf.reshape(image_t, [1,pixels,pixels,3])

    predictions = fcnfunc(image_t)

    cropback = True
    if cropback:
        image_t3d = tf.image.resize_image_with_crop_or_pad(image_t, inshape32[0], inshape32[1])
        predictions = tf.image.resize_image_with_crop_or_pad(tf.expand_dims(predictions, -1), inshape32[0], inshape32[1])

    image_t3d = tf.squeeze(image_t3d)
    predictions = tf.squeeze(predictions)

    # weights = tf.cast(tf.less_equal(predictions, number_of_classes - 1), tf.int64)
    # predictions = predictions * weights # tf.cast(weights, tf.int32)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.Saver().restore(sess, checkpoint)
        ext = '.mp4'
        video_file_out = video_file_in.replace(ext, '_segmented'+ext)

        from moviepy.editor import VideoFileClip
        input_clip = VideoFileClip(video_file_in)

        mask_alpha = round(0.3*255)
        colors = np.random.random([21, 3])
        colors -= np.min(colors, axis=1)[:, np.newaxis]
        colors /= np.max(colors, axis=1)[:, np.newaxis]
        colors *= 255
        colors = np.concatenate(( np.round(colors), mask_alpha*np.ones((21,1))), axis=1)
        colors[0][3] = 0 # don't color background

        def process_frame(image_in):
            scaled_image, inferred_pixel_labels = sess.run([image_t3d, predictions], {image_ph: image_in})
            seg_mask = np.take(colors, inferred_pixel_labels, axis=0)
            # print seg_mask.shape, type(seg_mask)
            image_in_walpha = np.concatenate((scaled_image, (255-mask_alpha)*np.ones(scaled_image.shape[:2]+(1,))), axis=2)
            # print inferred_pixel_labels.shape, seg_mask.shape, image_in_walpha.shape
            # print np.min(rescaled_image), np.max(rescaled_image)
            composite = Image.alpha_composite(Image.fromarray(np.uint8(image_in_walpha)),
                                              Image.fromarray(np.uint8(seg_mask)))
            composite_backscaled = composite.resize(image_in.shape[1::-1], Image.LANCZOS)
            return np.array(composite_backscaled)[:,:,:3]

        annotated_clip = input_clip.fl_image(process_frame)
        annotated_clip.write_videofile(video_file_out, audio=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a FCN(-based) segmentation net')
    parser.add_argument('--traindir', type=str,
                        help='the folder in which the results of the training run reside..',
                        default='')
    # TODO add back an option to run without preprocessing... ("native")
    parser.add_argument('--pixels', dest='pixels', type=int,
                         help='if not zero, normalize (interpolate&crop) image (and annotation)'
                              ' to pixels X pixels before inference... otherwise use the training setting',
                         default=0)
    parser.add_argument('--vizstep', type=int,
                        help='set to X < size(val.set) to draw visualization each X images',
                        default=5555)
    parser.add_argument('--hquant', type=bool,
                        help='set to True to do a Hailo-quantized run...',
                        default=False)
    parser.add_argument('--single_image', type=str,
                        help='set to a path in order to run on a single image',
                        default=None)
    parser.add_argument('--movie', type=str,
                        help='set to a path in order to run on a movie',
                        default=None)
    parser.add_argument('--gpu', '-g', type=int,
                        help='which GPU to run on (note: opposite to nvidia-smi)',
                        default=0)
    parser.add_argument('--afteriter', type=int,
                        help='if nonzero, use an intermediate checkpoint after such and such training batches',
                        default=0)

    if len(sys.argv) == 1:
        print("No args, running with defaults...")
        parser.print_help()

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    checkpoint = args.traindir+'/fcn.ckpt' 
    if args.afteriter!=0 :
	checkpoint += ('_'+str(args.afteriter))
    testpixels = args.pixels

    # get the architecture configuration from the training run folder..
    import json
    trainargs = json.load(open(args.traindir+'/runargs'))
    args.__dict__.update(trainargs)

    pixels = testpixels or args.pixels # use what's given, with the training as default.

    tf.reset_default_graph()

    fcn_builder = fcn_arch.FcnArch(number_of_classes=number_of_classes, is_training=False, net=args.basenet,
                                   trainable_upsampling=args.trainable_upsampling, fcn16=args.fcn16)

    # note: after adaptation, network returns final labels and not logits
    if pixels == (pixels/32)*32.0 :
        fcnfunc_img2labels = lambda img: tf.argmax(fcn_builder.build_net(img), dimension=3)
    else:
        print '...non-mult of 32, doing the generic adaptation...'
        fcnfunc_img2labels = utils.inference.adapt_network_for_any_size_input(fcn_builder.build_net, 32)

    if args.movie:
        segment_movie(fcnfunc_img2labels, checkpoint, args.movie, pixels)
        exit()
    if args.single_image:
        iterator = single_image_feed(args.single_image, pixels)
        X_as_in_visualize_each_Xth_seg = 1
    else:
        iterator = get_data_feed(pixels)
        X_as_in_visualize_each_Xth_seg = args.vizstep

    image, annotation = iterator.get_next()

    if not args.hquant:
        predictions = fcnfunc_img2labels(tf.expand_dims(image, axis=0))
        test(image, annotation, predictions, checkpoint, iterator)
    else: 
	print("Please contact Hailo to enter Early Access Program and gain access to Hailo-quantized version of this net")
