import tensorflow as tf
import os, sys, argparse
from matplotlib import pyplot as plt
import ipdb

import numpy as np
slim = tf.contrib.slim

# assuming our fork of tf-models is cloned side-by-side with current repo
sys.path.append("../tf-models-hailofork/research/slim/")

tf_ver = float('.'.join(tf.__version__.split('.')[:2]))
if tf_ver >= 1.4:
    data = tf.data
else:
    data = tf.contrib.data

import arch, train, utils


def iter_test(annotation, predictions, checkpoint, iterator, args,
              more_tensors_to_eval=[], callback=None):
    """
        iterate over the validation set, and compute overall (m)IoU metric(s)

        # Note tensors_to_eval & callback - a placeholder for additional operation(s)
            to be run in the context of each image, without changing this function code..
            e.g. visualize images 20-25
    """
    annotation_b = tf.expand_dims(annotation, axis=0)
    # Mask out the irrelevant (a.k.a ambiguous a.k.a unlabeled etc.) pixels from evaluation -
    weights = tf.to_float(tf.less(annotation_b, args.num_classes))

    # note labels clipped to be inside range for legit confusion matrix -
    #   but that doesn't harm result thanks to masking by weights.
    labels_b_clipped = tf.clip_by_value(annotation_b, 0, args.num_classes - 1)
    miou, update_op = tf.metrics.mean_iou(predictions=predictions,
                                          labels=labels_b_clipped,
                                          num_classes=args.num_classes,
                                          weights=weights)
    conf_op = tf.confusion_matrix(tf.reshape(predictions, [-1]),
                                  tf.reshape(labels_b_clipped, [-1]),
                                  num_classes=args.num_classes,
                                  weights=tf.reshape(weights, [-1]))

    conf_mtx = np.zeros([args.num_classes]*2)
    initializer = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(initializer)
        sess.run(iterator.initializer)
        saver.restore(sess, checkpoint)

        for i in range(args.num_images):
            _eval_res = sess.run([conf_op, update_op]+more_tensors_to_eval)
            conf_tmp = _eval_res[0]
            if callback: # a placeholder to inject more functionality w.o. changing this func
                callback(i, _eval_res[2:])
            conf_mtx += conf_tmp

        final_miou = sess.run(miou)

    print("Final mIoU for {0} images is {1:.2f}%".format(args.num_images, final_miou * 100))

    print("\n\n ---- Breakup by class: ----")
    diag = conf_mtx.diagonal()
    err1 = conf_mtx.sum(axis=1) - conf_mtx.diagonal()
    err2 = conf_mtx.sum(axis=0) - conf_mtx.diagonal()
    iou = diag / (0.0 + diag + err1 + err2)
    # print(pascal_voc_lut)
    for i, x in enumerate(iou):
        print(args.clabel2cname[i]+': {0:.2f}%'.format(x*100))
    print(np.mean(iou)) # just a sanity check to verify that it's the same as final_miou
    print(conf_mtx.sum(axis=1), conf_mtx.sum(axis=0))


def visualize(image_np, upsampled_predictions, annotation_np=None, i=0,
              od_class2color=None, clabel2cname=None):

    fsize = (20, 8) if annotation_np is None else (40, 5)
    # ipdb.set_trace()
    plt.figure(figsize=fsize)
    subplots = 2 if annotation_np is None else 3
    plt.suptitle('image #{0}'.format(i), fontsize=16)
    plt.subplot(1, subplots, 1); plt.title("orig image", fontsize=16)
    plt.imshow(image_np)
    plt.subplot(1, subplots, 2); plt.title("segmentation", fontsize=16)
    utils.visualization.visualize_segmentation_adaptive(upsampled_predictions.squeeze(), image_np,
                                                        od_class2color=od_class2color, clabel2cname=clabel2cname)
    if annotation_np is not None:
        plt.subplot(1, subplots, 3); plt.title("ground truth", fontsize=16)
        utils.visualization.visualize_segmentation_adaptive(annotation_np.squeeze(), image_np,
                                                            od_class2color=od_class2color, clabel2cname=clabel2cname)
    plt.show()

def get_data_feed(val_rec_fname, pixels=None):
    '''
        returning 4-element feed: orig_shape, scale, image, annotation.

        TODO: unify parts with with prepare_graph()
    '''
    dataset = data.TFRecordDataset([val_rec_fname]).map(utils.tf_records.parse_record)  # .batch(1)
    # note - saving shape before rescale
    dataset = dataset.map(lambda img, ann: (tf.to_float(tf.shape(img)), img, ann))
    if pixels is not None:
        dataset = dataset.map(lambda orig_shape_f, img, ann:
                              (orig_shape_f, tf.reduce_min(pixels/orig_shape_f)) +
                              utils.augmentation.nonrandom_rescale(img, ann, [pixels, pixels]))
    else:
        dataset = dataset.map(lambda shape, img, ann:
                              (shape, 1, img, tf.cast(ann, tf.int32)))

    iterator = dataset.repeat().make_initializable_iterator()
    return iterator


def segment_image(fcnfunc, checkpoint, image_path, pixels=None, clabel2cname=None):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # TODO the generic line below would enable monochrome images but currently breaks other stuff...
    #  image = tf.image.decode_jpeg(image)
    image_t3d, predictions = prepare_graph(fcnfunc, image, pixels)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.Saver().restore(sess, checkpoint)
        scaled_image, inferred_pixel_labels = sess.run([image_t3d, predictions])#, {image_ph: image_in})
        visualize(scaled_image, inferred_pixel_labels, clabel2cname=clabel2cname)

def prepare_graph(fcnfunc, image_ph, pixels=None):
    '''
        Builds graph of preprocessing (scale large side to pixels, pad the other),
        running the segmentation and cropping the margins from the result.

        Some duplication to utils.augmentation.nonrandom_rescale, TODO reconsider design..
    '''
    image_t = tf.expand_dims(image_ph, 0)

    if pixels is None:
        predictions = tf.squeeze(fcnfunc(image_t))
        image_t3d = tf.cast(tf.squeeze(image_t), tf.int32)
        return image_t3d, predictions

    input_shape = tf.to_float(tf.shape(image_ph))[:2]
    scale = tf.reduce_min(pixels / input_shape)

    inshape32 = tf.cast(tf.round(input_shape * scale), tf.int32)
    #image_t = tf.image.resize_nearest_neighbor(image_t, inshape32)
    image_t = tf.image.resize_area(image_t, inshape32) # better in case of downsampling (avoids aliasing a.k.a "moire")

    image_t = tf.image.resize_image_with_crop_or_pad(image_t, pixels, pixels)
    image_t3d = image_t = tf.reshape(image_t, [1,pixels,pixels,3])

    predictions = fcnfunc(image_t)

    cropback = True
    if cropback:
        image_t3d = tf.image.resize_image_with_crop_or_pad(image_t, inshape32[0], inshape32[1])
        predictions = tf.image.resize_image_with_crop_or_pad(predictions, inshape32[0], inshape32[1])

    image_t3d = tf.cast(tf.squeeze(image_t3d), tf.int32)
    predictions = tf.squeeze(predictions)

    return image_t3d, predictions

def segment_movie(fcnfunc, checkpoint, video_file_in, pixels=None):
    from PIL import Image
    from moviepy.editor import VideoFileClip

    image_ph = tf.placeholder(tf.int32)  # , shape = smth w. pixels (doesn't work..:)
    image_t3d, predictions = prepare_graph(fcnfunc, image_ph, pixels)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.Saver().restore(sess, checkpoint)
        #ext = '.mp4'
        ext = video_file_in[video_file_in.find('.') :]
        video_file_out = video_file_in.replace(ext, '_segmented'+ext)
        video_file_out = video_file_out.replace('.avi', '.mp4')
        #print video_file_out

        input_clip = VideoFileClip(video_file_in)

        mask_alpha = round(0.3*255)
        colors = np.random.random([21, 3])
        colors -= np.min(colors, axis=1)[:, np.newaxis]
        colors /= np.max(colors, axis=1)[:, np.newaxis]
        colors *= 255
        colors = np.concatenate(( np.round(colors), mask_alpha*np.ones((21,1))), axis=1)
        background_class = 0 # TODO what about other cases?
        colors[background_class][3] = 0

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


def main(args):
    '''
      TODO can't mass-test on other dataset family then trained;
               - consider adding support (although unnatural use)
    :param args:
    :return:
    '''
    checkpoint = args.traindir+'/fcn.ckpt'
    if args.afteriter!=0 :
        checkpoint += ('_'+str(args.afteriter))

    # get the architecture configuration from the training run folder..
    import json
    trainargs = json.load(open(args.traindir+'/runargs'))

    # resolve the working resolution - passed as arg CLI / used in train / "native" of image
    if args.pixels > 0:
        pixels = args.pixels
    elif args.pixels < 0:
        pixels = trainargs['pixels']
    else:
        pixels = None

    # Build net, using architecture flags which were used in train (test same net as trained)
    args.__dict__.update(trainargs)
    if type(args.extended_arch)in [str, unicode] and args.extended_arch != '':
        import newer_arch
        netclass = eval('newer_arch.' + args.extended_arch)
    else:
        netclass = arch.FcnArch
    print("Using architecture " + netclass.__name__)

    args = train.resolve_dataset_family(args)
    args.num_images = args.num_images or args.total_val_images
    net_builder = netclass(number_of_classes=args.num_classes, is_training=False, net=args.basenet,
                           trainable_upsampling=args.trainable_upsampling, fcn16=args.fcn16, fcn8=args.fcn8)

    # ..From logits to class predictions
    if pixels > 0 and pixels == (pixels/32)*32.0 :
        def netfunc_img2labels(img):
            tmp = tf.argmax(net_builder.build_net(img), dimension=3)
            return tf.expand_dims(tmp, 3)
    else:
        print('..."native" mode or size not multiple of 32, doing the generic adaptation...')
        fcnfunc_img2labels = utils.inference.adapt_network_for_any_size_input(net_builder.build_net, 32)

    # OK, let's get data build graph and run stuff!
    tf.reset_default_graph()

    if args.moviepath:
        # (!) this fails for "native" res. not a biggy i think. TODO retry fix
        segment_movie(netfunc_img2labels, checkpoint, args.moviepath, pixels)
    elif args.imagepath:
        segment_image(netfunc_img2labels, checkpoint, args.imagepath, pixels, args.clabel2cname)

    # run over the validation set
    else:
        val_tfrec_fname = os.path.join(args.datapath, 'validation.tfrecords')
        iterator = get_data_feed(val_tfrec_fname, pixels)
        orig_shape_f, scale, image_t, annotation_t = iterator.get_next()
        if args.hquant:
            print("Coming soon - quantized version for real-time deployments...")
        prediction_t = netfunc_img2labels(tf.expand_dims(image_t, axis=0))

        shape2crop_f = orig_shape_f*scale if pixels else orig_shape_f
        shape2crop = tf.cast(tf.round(shape2crop_f), tf.int32)
        imageT4viz, predT4viz, labelT4viz = \
            [tf.squeeze(tf.image.resize_image_with_crop_or_pad(x, shape2crop[0], shape2crop[1]))
                for x in image_t, prediction_t, annotation_t]

        def viz_cb(i, (image_np, predictions_np, annotation_np)):
            if args.first2viz <= i < args.last2viz:
                visualize(image_np, predictions_np, annotation_np, i,
                          clabel2cname=args.clabel2cname, od_class2color=args.__dict__.get('od_class2color'))

        # run over the images, sending some to visualization via the callback mechanism...
        iter_test(annotation_t, prediction_t, checkpoint, iterator, args,
                  callback=viz_cb, more_tensors_to_eval=[imageT4viz, predT4viz, labelT4viz])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a FCN(-based) segmentation net')
    parser.add_argument('--traindir', type=str,
                        help='the folder in which the results of the training run reside..',
                        default='')
    # TODO add back an option to run "native" (without pre-scaling to pixels x pixels at all)
    parser.add_argument('--pixels', dest='pixels', type=int,
                         help='A.) if >0, normalize images (&annotation) to [pixels X pixels] '
                                 ' before shoving them down the net''s throat,'
                                 ' by up(down)sampling larger side and padding the other to get square shape'                              
                              'B.) if -1 use the *pixels* used at train time (normally gives best results)'
                              'C.) if 0 dont preprocess ("native")',
                         default=-1)
    parser.add_argument('--first2viz', type=int,
                        help='set to X < size(val.set) to draw visualization for images between X and Y',
                        default=5555)
    parser.add_argument('--last2viz', type=int,
                        help='set to Y > X to draw visualization for images between X and Y',
                        default=5555)
    parser.add_argument('--num_images', type=int,
                        help='num first images of validation set to test on; '
                             'leave default/0 to use all availabe validation set',
                        default=0)
    parser.add_argument('--hquant', type=bool,
                        help='set to True to do a Hailo-quantized run...',
                        default=False)
    parser.add_argument('--imagepath', type=str,
                        help='set to a path in order to run on a single image',
                        default=None)
    parser.add_argument('--moviepath', type=str,
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

    main(args)
