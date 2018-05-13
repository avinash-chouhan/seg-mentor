import tensorflow as tf
import os, sys, argparse
from matplotlib import pyplot as plt

import numpy as np
slim = tf.contrib.slim

# assuming our fork of tf-models is cloned side-by-side with current repo
sys.path.append("../tf-models-hailofork/research/slim/")

tf_ver = float('.'.join(tf.__version__.split('.')[:2]))
if tf_ver >= 1.4:
    data = tf.data
else:
    data = tf.contrib.data

import fcn_arch, fcn_train, utils

pascal_val_rec_fname = '/data/pascal_augmented_berkely/validation.tfrecords'
pascal_number_of_classes = 21
pascal_voc_lut = utils.pascal_voc.pascal_segmentation_lut()

# ...assumes "mode 2" in utils.pascal_voc.get_augmented_pascal_image_annotation_filename_pairs()
#     was used on creation of the tfrecord file
PASCAL_VAL_SET_SIZE = utils.pascal_voc.PASCAL12_VALIDATION_WO_BERKELEY_TRAINING  # 904 RV-VOC12

CAMVID_VAL_DATA_SIZE = 237 # 107

def resolve_dataset_family(args):
    args = fcn_train.resolve_dataset_family(args)
    if args.dataset_family == 'pascal':
        args.total_val_images = PASCAL_VAL_SET_SIZE
    elif args.dataset_family == 'camvid':
        args.total_val_images = CAMVID_VAL_DATA_SIZE
    else:
        assert 0, "dataset family " + args.dataset_family + " not supported"
    return args

def iter_test(annotation, predictions, checkpoint, iterator, num_classes,
              num_images=PASCAL_VAL_SET_SIZE, classes_lut=pascal_voc_lut,
              more_tensors_to_eval=[], callback=None):
    """
        iterate over the validation set, and compute overall (m)IoU metric(s)

        # Note tensors_to_eval & callback - a placeholder for additional operation(s)
            to be run in the context of each image, without changing this function code..
            e.g. visualize images 20-25
    """
    annotation_batch_tensor = tf.expand_dims(annotation, axis=0)

    # Prepare mask to exclude some pixels from evaluation - 
    # e.g. "padding margins" (=255) or "ambiguous" (=num_classes (additional class))
    weights = tf.to_float(tf.less_equal(annotation_batch_tensor, num_classes-1))
    #  mean_iou won't work without the line below -
    #    expects valid class-label values even in masked out regions, apparently..
    annotation_batch_tensor = annotation_batch_tensor*tf.cast(weights, tf.int32)

    miou, update_op = tf.metrics.mean_iou(predictions=predictions,
                                          labels=annotation_batch_tensor,
                                          num_classes=num_classes,
                                          weights=weights)

    conf_op = tf.confusion_matrix(tf.reshape(predictions, [-1]),
                                  tf.reshape(annotation_batch_tensor, [-1]),
                                  num_classes=num_classes,
                                  weights=tf.reshape(weights, [-1]))
    conf_mtx = np.zeros([num_classes]*2)

    initializer = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(initializer)
        sess.run(iterator.initializer)
        saver.restore(sess, checkpoint)

        for i in range(num_images):
            _eval_res = sess.run([conf_op, update_op]+more_tensors_to_eval)
            conf_tmp = _eval_res[0]
            if callback: # a placeholder to inject more functionality w.o. changing this func
                callback(i, _eval_res[2:])
            conf_mtx += conf_tmp

        final_miou = sess.run(miou)

    print("Final mIoU for {0} images is {1:.2f}%".format(num_images, final_miou * 100))

    print("\n\n ---- Breakup by class: ----")
    diag = conf_mtx.diagonal()
    err1 = conf_mtx.sum(axis=1) - conf_mtx.diagonal()
    err2 = conf_mtx.sum(axis=0) - conf_mtx.diagonal()
    iou = diag / (0.0 + diag + err1 + err2)
    # print(pascal_voc_lut)
    for i, x in enumerate(iou):
        print(classes_lut[i]+': {0:.2f}%'.format(x*100))
    print(np.mean(iou)) # just a sanity check to verify that it's the same as final_miou
    print(conf_mtx.sum(axis=1), conf_mtx.sum(axis=0))

# cm = lsc.from_list('goo', [[y/256.0 for y in x] for x in camvid11_od_classes_colors.values()], 12)
# ax1=plt.imshow(im1, cmap=cm, vmin=-0.5, vmax=11.5); axc=plt.colorbar(ax1, ticks=range(13));
# axc.ax.set_yticklabels(camvid11_od_classes_colors.keys())
# ticks=np.arange(np.min(data), np.max(data)+1)


def visualize(image_np, upsampled_predictions, annotation_np=None, i=0, classes_lut=None):
    plt.figure(figsize=(30, 10))
    subplots = 2 if annotation_np is None else 3
    plt.suptitle('image #{0}'.format(i))
    plt.subplot(1, subplots, 1); plt.title("orig image")
    plt.imshow(image_np)
    plt.subplot(1, subplots, 2); plt.title("segmentation")
    utils.visualization.visualize_segmentation_adaptive(upsampled_predictions.squeeze(), classes_lut, image_np)
    if annotation_np is not None:
        plt.subplot(1, subplots, 3); plt.title("ground truth")
        utils.visualization.visualize_segmentation_adaptive(annotation_np.squeeze(), classes_lut, image_np)
    plt.show()

def get_data_feed(val_rec_fname=pascal_val_rec_fname, pixels=None):
    dataset = data.TFRecordDataset([val_rec_fname]).map(utils.tf_records.parse_record)  # .batch(1)
    if pixels is not None:
        dataset = dataset.map(lambda img, ann:
                              utils.augmentation.nonrandom_rescale(img, ann, [pixels, pixels]))
    else:
        dataset = dataset.map(lambda img, ann:
                              (img, tf.cast(ann, tf.int32)))
    iterator = dataset.repeat().make_initializable_iterator()
    return iterator


def segment_image(fcnfunc, checkpoint, image_path, pixels=None):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.decode_jpeg(image) # TODO this would enable monochrome images but currently breaks other stuff...
    image_t3d, predictions = prepare_ph_path(fcnfunc, image, pixels)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.Saver().restore(sess, checkpoint)
        scaled_image, inferred_pixel_labels = sess.run([image_t3d, predictions])#, {image_ph: image_in})
        visualize(scaled_image, inferred_pixel_labels)

def prepare_ph_path(fcnfunc, image_ph, pixels=None):
    image_t = tf.expand_dims(image_ph, 0)

    if pixels is None:
        predictions = tf.squeeze(fcnfunc(image_t))
        image_t3d = tf.cast(tf.squeeze(image_t), tf.int32)
        return image_t3d, predictions

    input_shape = tf.to_float(tf.shape(image_ph))[:2]

    #pixels = pixels or tf.reduce_max(input_shape) # won't really work w. pixels==None
    scale = tf.reduce_min(pixels / input_shape)

    inshape32 = tf.cast(tf.round(input_shape * scale), tf.int32)
    #image_t = tf.image.resize_nearest_neighbor(image_t, inshape32)
    image_t = tf.image.resize_area(image_t, inshape32) # better in case of downsampling (avoids aliasing a.k.a "moire")
    #print 'pixels', pixels
    image_t = tf.image.resize_image_with_crop_or_pad(image_t, pixels, pixels)
    image_t3d = image_t = tf.reshape(image_t, [1,pixels,pixels,3])

    predictions = fcnfunc(image_t)

    cropback = True
    if cropback:
        image_t3d = tf.image.resize_image_with_crop_or_pad(image_t, inshape32[0], inshape32[1])
        predictions = tf.image.resize_image_with_crop_or_pad(predictions, #tf.expand_dims(predictions, -1),
                                                             inshape32[0], inshape32[1])

    image_t3d = tf.cast(tf.squeeze(image_t3d), tf.int32)
    predictions = tf.squeeze(predictions)

    return image_t3d, predictions

def segment_movie(fcnfunc, checkpoint, video_file_in, pixels=None):
    from PIL import Image
    from moviepy.editor import VideoFileClip

    image_ph = tf.placeholder(tf.int32)  # , shape = smth w. pixels (doesn't work..:)
    image_t3d, predictions = prepare_ph_path(fcnfunc, image_ph, pixels)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        tf.train.Saver().restore(sess, checkpoint)
        ext = '.mp4'
        video_file_out = video_file_in.replace(ext, '_segmented'+ext)

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
        netclass = fcn_arch.FcnArch
    print("Using architecture " + netclass.__name__)

    args = resolve_dataset_family(args)
    fcn_builder = netclass(number_of_classes=args.num_classes, is_training=False, net=args.basenet,
                           trainable_upsampling=args.trainable_upsampling, fcn16=args.fcn16, fcn8=args.fcn8)
    # fcn_builder = fcn_arch.FcnArch(number_of_classes=number_of_classes, is_training=False, net=args.basenet,
    #                                trainable_upsampling=args.trainable_upsampling, fcn16=args.fcn16)

    # ..From logits to class predictions
    if pixels > 0 and pixels == (pixels/32)*32.0 :
        def fcnfunc_img2labels(img):
            tmp = tf.argmax(fcn_builder.build_net(img), dimension=3)
            return tf.expand_dims(tmp, 3)
    else:
        print('..."native" mode or size not multiple of 32, doing the generic adaptation...')
        fcnfunc_img2labels = utils.inference.adapt_network_for_any_size_input(fcn_builder.build_net, 32)

    # OK, let's get data build graph and run stuff!
    tf.reset_default_graph()

    if args.moviepath:
        # (!) this fails for "native" res. not so interesting..
        segment_movie(fcnfunc_img2labels, checkpoint, args.moviepath, pixels)
    elif args.imagepath:
        segment_image(fcnfunc_img2labels, checkpoint, args.imagepath, pixels)
    else: # run over the validation set
        val_tfrec_fname = os.path.join(args.datapath, 'validation.tfrecords')
        iterator = get_data_feed(val_tfrec_fname, pixels)
        image_t, annotation_t = iterator.get_next()
        if args.hquant:
            print("Coming soon - quantized version for real-time deployments...")
        prediction_t = fcnfunc_img2labels(tf.expand_dims(image_t, axis=0))
        def viz_cb(i, (image_np, upsampled_predictions, annotation_np)):
            if args.first2viz <= i < args.last2viz:
                visualize(image_np, annotation_np, upsampled_predictions, i, args.label2name)
        iter_test(annotation_t, prediction_t, checkpoint, iterator, num_classes=args.num_classes,
                  num_images=args.total_val_images, classes_lut=args.label2name,
                  callback=viz_cb, more_tensors_to_eval=[image_t, annotation_t, prediction_t])


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
