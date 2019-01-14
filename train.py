import tensorflow as tf, numpy as np
import os, sys, time, argparse

# assuming our fork of tf-models is cloned side-by-side with current repo
sys.path.append("../tf-models-hailofork/research/slim/")

import arch, utils

slim = tf.contrib.slim
#make small fixes to work with older version of TF
tf_ver = float('.'.join(tf.__version__.split('.')[:2]))
if tf_ver >= 1.4:
    data = tf.data
else:
    data = tf.contrib.data

# image_train_size = [512, 512]

PASCAL_VAL_SET_SIZE = utils.pascal_voc.PASCAL12_VALIDATION_WO_BERKELEY_TRAINING  # 904 RV-VOC12

CAMVID_VAL_DATA_SIZE = 237 # 107


def resolve_dataset_family(args):
    '''
        Given a dataset family, set the properties of the dataset (if not set explicitly)
        to the corresponding defaults of the family..

    :param args: arguments to fcn_train - both if we run from taining session.
                 or from test session given a folder resultant of a training session
                 (in this case the dataset family

    :return: aumgneted args

    # TODO - read num (of train/val) images from the tfrecord itself,
         thus removing reliance on consistency with tfrecord-ification done previously w. ext. script
    '''

    # Note: maintaining backwards compatibility for trains performed w.o. this args
    #        default to pascal... note you can also edit the json of training folder to circumvent defaults @test
    if not hasattr(args,'dataset_family'):
        args.dataset_family = 'pascal_seg'

    if args.dataset_family == 'pascal_seg':
        args.clabel2cname = utils.pascal_voc.pascal_segmentation_lut()
        #args.od_classes_colors = pascal_od_class2color # TODO set consistent colors
        if not hasattr(args,'num_classes') or args.num_classes == 0:
            args.num_classes = utils.pascal_voc.PASCAL_NUM_CLASSES
        # Note - assumes "mode 2" in utils.pascal_voc.get_augmented_pascal_image_annotation_filename_pairs()
        #     was used on creation of the tfrecord file
        if not hasattr(args,'total_train_images') or args.total_train_images == 0 :
            # ...assumes "mode 2" was used on creation of the tfrecord file (see pascal_voc.py)
            args.total_train_images = utils.pascal_voc.BERKELY_U_PASCAL12_TRAINING # 11127
        if not hasattr(args, 'total_val_images') or args.total_val_images == 0:
            args.total_val_images =utils.pascal_voc. PASCAL12_VALIDATION_WO_BERKELEY_TRAINING # 904 RV-VOC12

    elif args.dataset_family == 'camvid':
        args.clabel2cname = utils.camvid.camvid_lut # camvid_label2name
        args.od_class2color = utils.camvid.camvid11_od_class2color
        if args.num_classes == 0:
            args.num_classes = utils.camvid.NUM_CLASSES
        if args.total_train_images == 0:
            args.total_train_images = utils.camvid.TRAIN_DATASET_SIZE
        if not hasattr(args, 'total_val_images') or args.total_val_images == 0:
            args.total_val_images = utils.camvid.VAL_DATASET_SIZE
    else:
        assert 0, "dataset family " + args.dataset_family + " not supported"

    # NOTE: num_classes and 255 are used interchangeably for the masked-out a.k.a ambiguous a.k.a unlabeled
    args.clabel2cname[args.num_classes] = args.clabel2cname[255] = 'unlabeled'

    if not hasattr(args,'datapath') or args.datapath == '':
        args.datapath = '/local/data/' + args.dataset_family
    #print args.num_classes, args.total_train_images; exit()
    return args


# For mode of "go overfit over first images"
#  (a.k.a "training convergence sanity test")
#  - set a number here...
debug_loop_over_few_first_images = None  # 30


class Trainer:

    def __init__(self, args, checkpoint_path):
        args = resolve_dataset_family(args)

        self.args = args
        self.checkpoint_path = checkpoint_path
        self.train_tfrec = os.path.join(self.args.datapath, 'training.tfrecords')
        self.test_tfrec = os.path.join(self.args.datapath, 'validation.tfrecords')

        if args.extended_arch!='':
            import newer_arch
            netclass = eval('newer_arch.'+args.extended_arch)
        else:
            netclass = arch.FcnArch
        print("Using architecture "+netclass.__name__)

        self.fcn_builder = netclass(number_of_classes=self.args.num_classes, net=args.basenet, #is_training=True,
                                    trainable_upsampling=args.trainable_upsampling, fcn16=args.fcn16, fcn8=args.fcn8)

    def run_training(self, trainfolder, num_epochs=10, learning_rate=1e-6, decaylr=True,
                     new_vars_to_learn_faster=None, pretrained_vars=None):

        chkpnt2save_path = trainfolder + '/fcn.ckpt'

        filtered_labels_1hot_flat, filtered_logits_flat = \
            utils.training.get_goodpixel_logits_and_1hot_labels(self.annotation_batch,
                                                                self.upsampled_logits_batch,  self.args.num_classes)

        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(labels=filtered_labels_1hot_flat,
                                                                  logits=filtered_logits_flat
                                                                  )
        if self.args.dataset_family=='camvid':
            pxl_classes_flat = tf.argmax(filtered_labels_1hot_flat, axis=1)
            weights = tf.to_float(tf.gather_nd(utils.camvid.camvid11_class_weights,
                                               tf.expand_dims(pxl_classes_flat, -1)))
            cross_entropies = tf.multiply(cross_entropies, weights)


        # Normalize the cross entropy -- the number of elements
        # is different during each step due to mask out regions
        cross_entropy_loss = tf.reduce_mean(cross_entropies)

        predictions = tf.argmax(self.upsampled_logits_batch, axis=3)
        # _probabilities = tf.nn.softmax(self.upsampled_logits_batch)

        global_step = tf.Variable(0, trainable=False)
        total_iterations = self.args.total_train_images * num_epochs / self.args.batch_size
        if decaylr:
            # Simple: reduce x10 LR twice during train..
            dr = 0.1
            ds = total_iterations / 3.0
            # Sophisticated - slowly reduce.. (didn't help but you're invited to retry..)
            # ds = 500
            # dr = 0.01 ** (ds*1.0 / total_iterations)  # decay x100 across the whole range, e.g. 
            # base-lr down to x0.01 base-lr for the weak variant.

            strong_learning_rate = tf.train.exponential_decay(learning_rate * 10, global_step=global_step,
                                                              decay_steps=ds, decay_rate=dr, staircase=True)
            weak_learning_rate = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                                            decay_steps=ds, decay_rate=dr, staircase=True)
        else:
            weak_learning_rate = learning_rate
            strong_learning_rate = 10 * learning_rate

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope("adam_vars"):
                if new_vars_to_learn_faster:
                    train_op1 = tf.train.AdamOptimizer(learning_rate=strong_learning_rate,
                                                       epsilon=1e-3, beta1=self.args.momentum). \
                        minimize(cross_entropy_loss, var_list=new_vars_to_learn_faster, global_step=global_step)
                    train_op2 = tf.train.AdamOptimizer(learning_rate=weak_learning_rate,
                                                       epsilon=1e-3, beta1=self.args.momentum). \
                        minimize(cross_entropy_loss, var_list=pretrained_vars)
                    train_step = tf.group(train_op1, train_op2)
                    # TODO is global_step iterated on each minimize()?
                else:
                    train_step = tf.train.AdamOptimizer(learning_rate=weak_learning_rate,
                                                        epsilon=1e-3, beta1=self.args.momentum). \
                        minimize(cross_entropy_loss, global_step=global_step)

        # TODO - do we need to do smth special to include regularization losses?

        # assuming the "ambiguous/ unlabeled" is either num_classes or 255 (or whatever in between:))
        mask = tf.to_int32(tf.less(self.annotation_batch, self.args.num_classes))
        clipped_labels =tf.clip_by_value(self.annotation_batch, 0, self.args.num_classes-1)
        miou_score_op, miou_update_op = tf.metrics.mean_iou(predictions=predictions,
                                                            labels=clipped_labels,
                                                            num_classes=self.args.num_classes,
                                                            name='my_miou', weights=mask)
        running_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_miou")
        running_metric_initializer = tf.variables_initializer(var_list=running_metric_vars)

        cross_entropy_loss_summary_op = tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
        miou_summary_op = tf.summary.scalar('train mIOU', miou_score_op)
        test_miou_summary_op = tf.summary.scalar('test mIOU', miou_score_op)

        learning_rate_summary_op = tf.summary.scalar('weak learning rate', weak_learning_rate)

        '''
        Note: we can't use the merged summary like this:
           merged_summary_op = tf.summary.merge_all() 
        because we want to separate the actual (forward/backward) run
        from the calculation of the metric because of the way TensorFlow works
        see http://ronny.rest/blog/post_2017_09_11_tf_metrics/        
        '''

        # The op for initializing the variables.
        global_vars_init_op = tf.global_variables_initializer()
        local_vars_init_op = tf.local_variables_initializer()
        combined_loc_glob_init_op = tf.group(local_vars_init_op, global_vars_init_op)

        # We need this to save only model variables and omit
        # optimization-related and other variables.
        model_variables = slim.get_model_variables()
        saver = tf.train.Saver(model_variables)

        with tf.Session() as sess:

            summary_string_writer = tf.summary.FileWriter(trainfolder + '/trainboard', graph=sess.graph)

            training_handle = sess.run(self.train_iter.string_handle())
            test_handle = sess.run(self.test_iter.string_handle())

            sess.run(combined_loc_glob_init_op)
            self.init_fn(sess)
            sess.run(self.train_iter.initializer)
            sess.run(self.test_iter.initializer)
            t0 = time.time()
            for iter_count in range(total_iterations):
                # print '--full loop ', time.time() - t0
                t0 = time.time()
                sess.run(running_metric_initializer)
                # print 'post metric init: ', time.time()-t0
                _, __, cross_entropy, cross_entropy_summary, learnrate_summary = \
                    sess.run([train_step, miou_update_op, cross_entropy_loss,
                              cross_entropy_loss_summary_op, learning_rate_summary_op],
                             feed_dict={self.masterhandle: training_handle,
                                        self.fcn_builder.is_training_ph:True})
                # print 'post lost&trainstep sess run ', time.time() - t0

                # this was helpful to get the mIOU metric work:
                #   http://ronny.rest/blog/post_2017_09_11_tf_metrics/
                miou_score, miou_summary = sess.run([miou_score_op, miou_summary_op])
                #print 'post miou sess run ', time.time() - t0
                summary_string_writer.add_summary(cross_entropy_summary, iter_count)
                summary_string_writer.add_summary(learnrate_summary, iter_count)
                # summary_string_writer.add_summary(miou_summary, trainstep).
                """ NOTE: on second thought, this "single batch train mIoU" computed on line above is meanignless,
                  and the low-pass filter in tensorboard (this was the first thought...) doesn't help.
                  The correct aggregation is over the confusion matrix, with subsequent mIoU calculation."""

                print("Batch loss: {0:.2f}, Batch mIOU (%): {1:.2f}".format(cross_entropy, miou_score * 100))
                periodic_test_eval_steps = 100
                if (iter_count + 1) % periodic_test_eval_steps == 0:
                    """ Here we do a periodic evaluation on a valdiation (SUB)set,
                      on each step running the miou update op (updating the confusion matrix under the hood),
                      and computing the loss (which, in contrast to mIoU, is OK to just average over the batches)"""
                    save_path = saver.save(sess, chkpnt2save_path)
                    print("step ", iter_count, time.ctime(), "updated model in: %s" % save_path)
                    sess.run(running_metric_initializer)
                    num_test_images = 240  # 15 batches, ~1/4 of valid. set (shuffled..)
                    num_test_batches = num_test_images / self.args.batch_size
                    test_cross_entropy_arr = np.zeros(num_test_batches)
                    for _tb in range(num_test_batches):
                        _, test_cross_entropy_arr[_tb] = \
                            sess.run([miou_update_op, cross_entropy_loss],
                                     feed_dict={self.masterhandle: test_handle, 
                                                self.fcn_builder.is_training_ph:False})
                    test_miou_score, test_miou_summary = sess.run([miou_score_op, test_miou_summary_op])
                    test_loss = np.mean(test_cross_entropy_arr)
                    test_loss_summary = tf.Summary()
                    test_loss_summary.value.add(tag='test cross entropy loss', simple_value=test_loss)
                    for _prevstep in range(periodic_test_eval_steps)[::-1]:
                        # write same value for all prev. steps, for compatibility with filters, etc.
                        summary_string_writer.add_summary(test_loss_summary, iter_count - _prevstep)
                        summary_string_writer.add_summary(test_miou_summary, iter_count - _prevstep)
                    print("----test mIOU: ", test_miou_score)
                    print("----test loss: ", test_loss)
                    print("----finished test eval...", time.ctime())
                # make a copy once in a while for case we start to overfit and model becomes worse...
                if iter_count % 3000 == 0:
                    saver.save(sess, chkpnt2save_path + '_{0}'.format(iter_count))

                # Nice example of usage Dataset iterators advantage.. (with queues it would be way more messy)
                if debug_loop_over_few_first_images and (iter_count + 1) % debug_loop_over_few_first_images == 0:
                    print '---'
                    sess.run(self.train_iter.initializer)

            save_path = saver.save(sess, chkpnt2save_path)
            print("Model saved in file: %s" % save_path)

            summary_string_writer.close()

    def setup(self):

        train_dataset = data.TFRecordDataset([self.train_tfrec])
        train_dataset = train_dataset.map(utils.tfrecordify.parse_record)

        # do data augmentation (unless we're in the debug mode of "go overfit over first images")
        shuffle_and_augment = not debug_loop_over_few_first_images
        if shuffle_and_augment:
            train_dataset = train_dataset.map(lambda image_, annotation_:
                                              utils.augmentation.random_horiz_flip(image_, annotation_))
            train_dataset = train_dataset.map(lambda image_, annotation_:
                                              utils.augmentation.random_rescale(image_, annotation_, image_train_size))
            # TODO - reconsider the distort-color (originally disabled in Daniil's too) and maybe other augmentations...
            # train_dataset = train_dataset.map(lambda image_, annotation_:
            #                       (utils.augmentation.distort_randomly_image_color(image_), annotation_))
            train_dataset = train_dataset.map(lambda image_, annotation_: (image_, tf.squeeze(annotation_)))
            train_dataset = train_dataset.shuffle(buffer_size=3000)
        else:
            train_dataset = train_dataset.map(lambda image_, annotation_:
                                              utils.augmentation.nonrandom_rescale(image_, annotation_,
                                                                                   image_train_size))
        train_dataset = train_dataset.batch(self.args.batch_size)

        test_dataset = data.TFRecordDataset([self.test_tfrec]).map(utils.tfrecordify.parse_record)
        test_dataset = test_dataset.map(lambda img, ann:
                                        utils.augmentation.nonrandom_rescale(img, ann, image_train_size))
        test_dataset = test_dataset.map(lambda image_, annotation_: (image_, tf.squeeze(annotation_)))
        test_dataset = test_dataset.shuffle(buffer_size=300).batch(self.args.batch_size)

        self.train_iter = train_dataset.repeat().make_initializable_iterator()
        self.test_iter = test_dataset.repeat().make_initializable_iterator()

        self.masterhandle = tf.placeholder(tf.string, shape=[])
        switching_iterator = data.Iterator.from_string_handle(
            self.masterhandle, train_dataset.output_types, train_dataset.output_shapes)

        image_batch, self.annotation_batch = switching_iterator.get_next()

        self.upsampled_logits_batch = self.fcn_builder.build_net(image_batch_tensor=image_batch)
        self.feat_extractor_variables_mapping, self.new_vars = self.fcn_builder.get_pretrained_and_new_vars()

        self.init_fn = slim.assign_from_checkpoint_fn(model_path=self.checkpoint_path,
                                                      var_list=self.feat_extractor_variables_mapping)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a FCN(-based) segmentation net')
    parser.add_argument('--basenet', type=str,
                        help='the base feature extractor',
                        default='vgg_16')
    parser.add_argument('--extended_arch', type=str,
                        help='if nontrivial, use extended architecture according to name',
                        default='')
    parser.add_argument('--trainable_upsampling', type=bool,
                        help='if True use trainable_upsampling in the basic FCN architecture',
                        default=False)
    parser.add_argument('--fcn16', type=bool,
                        help='if True add the fcn16 skip connection',
                        default=False)
    parser.add_argument('--fcn8', type=bool,
                        help='if True add the fcn8 skip connection',
                        default=False)
    parser.add_argument('--fcn4', type=bool,
                        help='if True add the fcn4 skip connection',
                        default=False)
    parser.add_argument('--batch_size', type=int,
                        help='batch size',
                        default=16)
    parser.add_argument('--epochs', type=int,
                        help='num of epochs to train for',
                        default=60)
    parser.add_argument('--learnrate', type=float,
                        help='base learning rate',
                        default=3e-4)
    parser.add_argument('--momentum', type=float,
                        help='momentum - the beta1 of the Adam optimizer',
                        default=0.9)
    parser.add_argument('--difflr', type=bool,
                        help='if True use x10 learning rate for new layers w.r.t pretrained',
                        default=False)  # if decaying rate - decay with same schedule retaining ratio
    parser.add_argument('--decaylr', type=bool,
                        help='if True decay learn rate from x10 to x0.1 base LR',
                        default=False)
    # parser.add_argument('--pixels', type=int,
    #                     help=' normalize images (&labels) to [pixels X pixels] before shoving them down the net''s throat,'
    #                          ' by up(down)sampling larger side and padding the other to get square shape',
    #                     default=512)
    parser.add_argument('--net_inp_shape', dest='net_inp_shape', type=str,
                        help='A.) if >0, normalize images (&annotation) to height, width = net_inp_shape'
                             ' before shoving them down the net''s throat,'
                             ' by up(down)sampling larger side and padding the other to get square shape',
                        default='512,512')
    parser.add_argument('--datapath', type=str,
                        help='path where tfrecords are located; if not set will use /local/data/<dataset-family>',
                        default='')
    parser.add_argument('--num_classes', type=int,
                        help='number of classes (dataset dependent); if not set will use dataset-damily default',
                        default=0)
    parser.add_argument('--total_train_images', type=int,
                        help='size of the training set; if not set will use dataset-damily default',
                        default=0)
    parser.add_argument('--dataset_family', type=str,
                        help='pascal_seg/ camvid / coco / ...',
                        default='pascal_seg')
    parser.add_argument('--modelspath', type=str,
                        help='path where imagenet-pretrained FE checkpoints are located',
                        default='/local/data/models/')
    parser.add_argument('--gpu', '-g', type=int,
                        help='which GPU to run on (note: possibly opposite of nvidia-smi..)',
                        default=0)

    if len(sys.argv) == 1:
        print("No args, running with defaults...")
        parser.print_help()

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    vgg_checkpoint_path = os.path.join(args.modelspath, 'vgg_16.ckpt')
    mobilenet_checkpoint_path = os.path.join(args.modelspath, 'mobilenet_v1_1.0_224.ckpt')
    mobilenet_v2_checkpoint_path = os.path.join(args.modelspath, 'mobilenet_v2/mobilenet_v2_1.0_224.ckpt')
    inception_checkpoint_path = os.path.join(args.modelspath, 'inception_v1.ckpt')
    resnet50_checkpoint_path = os.path.join(args.modelspath, 'resnet_v1_50.ckpt')
    resnet18_checkpoint_path = os.path.join(args.modelspath, 'resnet_v1_18/model.ckpt')

    checkpoint_path = {'vgg_16': vgg_checkpoint_path,
                       'resnet_v1_50': resnet50_checkpoint_path,
                       'resnet_v1_18': resnet18_checkpoint_path,
                       'inception_v1': inception_checkpoint_path,
                       'mobilenet_v1': mobilenet_checkpoint_path,
                       'mobilenet_v2': mobilenet_v2_checkpoint_path,
                       }.get(args.basenet)
    if not checkpoint_path:
        raise Exception("Not yet supported feature extractor")

    # image_train_size = [args.pixels, args.pixels]
    image_train_size = [int(x) for x in args.net_inp_shape.split(',')]

    trainer = Trainer(args, checkpoint_path)
    trainer.setup()

    # Create a folder for this training and save configuration..
    import datetime, json
    today = datetime.date.today().strftime('%b%d')
    prefix = '{today}_{net}__'.format(net=args.basenet, today=today)
    thisrunnum = 1 + max([0] + [int(f[len(prefix):]) for f in os.listdir('./tmp') if prefix in f])
    trainfolder = './tmp/' + prefix + str(thisrunnum)
    os.makedirs(trainfolder)
    json.dump(args.__dict__, open(os.path.join(trainfolder, 'runargs'), 'w'), indent=2)

    # redirecting PRINT statements into log in same folder...
    sys.stdout = sys.stderr = open(os.path.join(trainfolder, 'runlog'), 'w', 1)  # line buffered

    if args.difflr:
        trainer.run_training(trainfolder=trainfolder, num_epochs=args.epochs,
                             learning_rate=args.learnrate,
                             decaylr=args.decaylr,
                             new_vars_to_learn_faster=trainer.new_vars,
                             pretrained_vars=trainer.feat_extractor_variables_mapping.values())
    else:
        trainer.run_training(trainfolder=trainfolder, num_epochs=args.epochs,
                             learning_rate=args.learnrate,
                             decaylr=args.decaylr,
                             )

