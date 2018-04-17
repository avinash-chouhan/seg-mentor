import tensorflow as tf, numpy as np
import os, sys, time, argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# (!!) needed for code inside fcn_arch, utils..
#sys.path.append("/home/finkel/hailo_repos/phase2-dl-research/slim_models/")
sys.path.append("../tf-models-hailofork/research/slim/")

import fcn_arch, utils

slim = tf.contrib.slim
checkpoints_dir = '/data/models'
log_folder = "tmp/logs"
vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')
mobilenet_checkpoint_path = os.path.join(checkpoints_dir, 'mobilenet_v1_224/mobilenet_v1_1.0_224.ckpt')
inception_checkpoint_path = os.path.join(checkpoints_dir, 'inception_v1.ckpt')
resnet50_checkpoint_path = os.path.join(checkpoints_dir, 'resnet_v1_50.ckpt')
resnet18_checkpoint_path = os.path.join(checkpoints_dir, 'resnet_v1_18/model.ckpt')
train_tfrec_fname = '/data/pascal_augmented_berkely/training.tfrecords'
test_tfrec_fname = '/data/pascal_augmented_berkely/validation.tfrecords'


image_train_size = [512, 512]

number_of_classes = 21

pascal_voc_lut = utils.pascal_voc.pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()

# ...follows from usage of "mode 2" in utils.pascal_voc.get_augmented_pascal_image_annotation_filename_pairs()
TRAIN_DATASET_SIZE = 11127

# ...assumes "mode 2" in utils.pascal_voc.get_augmented_pascal_image_annotation_filename_pairs()
#     was used on creation of the tfrecord file
TRAIN_DATASET_SIZE = utils.pascal_voc.BERKELY_U_PASCAL12_TRAINING  # 904

# For mode of "go overfit over first images"
#  (a.k.a "training convergence sanity test")
#  - set a number here...
debug_loop_over_few_first_images = None  # 30


class Trainer:

    def __init__(self, args, checkpoint_path):
        self.args = args
        self.checkpoint_path = checkpoint_path

        self.fcn_builder = fcn_arch.FcnArch(number_of_classes=number_of_classes, is_training=True, net=args.basenet,
                                            trainable_upsampling=args.trainable_upsampling, fcn16=args.fcn16)

    def run_training(self, trainfolder, num_epochs=10, learning_rate=1e-6, decaylr=True,
                     new_vars_to_learn_faster=None, pretrained_vars=None):

        summary_string_writer = tf.summary.FileWriter(trainfolder + '/trainboard')
        chkpnt2save_path = trainfolder + '/fcn.ckpt'

        valid_labels_batch_tensor, valid_logits_batch_tensor = \
            utils.training.get_valid_logits_and_labels(annotation_batch_tensor=self.annotation_batch,
                                                       logits_batch_tensor=self.upsampled_logits_batch,
                                                       class_labels=class_labels)

        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                                  labels=valid_labels_batch_tensor)

        # Normalize the cross entropy -- the number of elements
        # is different during each step due to mask out regions
        cross_entropy_loss = tf.reduce_mean(cross_entropies)

        predictions = tf.argmax(self.upsampled_logits_batch, axis=3)
        # _probabilities = tf.nn.softmax(self.upsampled_logits_batch)

        global_step = tf.Variable(0, trainable=False)
        total_iterations = TRAIN_DATASET_SIZE * num_epochs / self.args.batch_size
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
                    # note momentum change 0.99->0.9 on Mar22
                    train_op1 = tf.train.AdamOptimizer(learning_rate=strong_learning_rate,
                                                       epsilon=1e-3, beta1=0.9). \
                        minimize(cross_entropy_loss, var_list=new_vars_to_learn_faster, global_step=global_step)
                    train_op2 = tf.train.AdamOptimizer(learning_rate=weak_learning_rate,
                                                       epsilon=1e-3, beta1=0.9). \
                        minimize(cross_entropy_loss, var_list=pretrained_vars, global_step=global_step)

                    train_step = tf.group(train_op1, train_op2)
                else:
                    train_step = tf.train.AdamOptimizer(learning_rate=weak_learning_rate,
                                                        epsilon=1e-3, beta1=0.99). \
                        minimize(cross_entropy_loss, global_step=global_step)

        mask = tf.to_int32(tf.not_equal(self.annotation_batch, 255))
        miou_score_op, miou_update_op = tf.metrics.mean_iou(predictions=predictions,
                                                            labels=tf.multiply(self.annotation_batch, mask),
                                                            num_classes=number_of_classes,
                                                            name='my_miou')  # , weights=mask)
        running_metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_miou")
        running_metric_initializer = tf.variables_initializer(var_list=running_metric_vars)

        cross_entropy_loss_summary_op = tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
        miou_summary_op = tf.summary.scalar('train mIOU', miou_score_op)
        test_miou_summary_op = tf.summary.scalar('test mIOU', miou_score_op)

        learning_rate_summary_op = tf.summary.scalar('weak learning rate', weak_learning_rate)

        # Note: we can't use the merged summary because we want to separate the actual (forward/backward) run
        #   from the calculation of the metric because of the way TensorFlow works
        #   see http://ronny.rest/blog/post_2017_09_11_tf_metrics/
        # merged_summary_op = tf.summary.merge_all()

        # Create the log folder if doesn't exist yet
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # The op for initializing the variables.
        global_vars_init_op = tf.global_variables_initializer()
        local_vars_init_op = tf.local_variables_initializer()
        combined_loc_glob_init_op = tf.group(local_vars_init_op, global_vars_init_op)

        # We need this to save only model variables and qomit
        # optimization-related and other variables.
        model_variables = slim.get_model_variables()
        saver = tf.train.Saver(model_variables)

        with tf.Session() as sess:

            training_handle = sess.run(self.train_iter.string_handle())
            test_handle = sess.run(self.test_iter.string_handle())

            sess.run(combined_loc_glob_init_op)
            self.init_fn(sess)
            sess.run(self.train_iter.initializer)
            sess.run(self.test_iter.initializer)
            t0 = time.time()
            for trainstep in xrange(total_iterations):
                # print '--full loop ', time.time() - t0
                t0 = time.time()
                sess.run(running_metric_initializer)
                # print 'post metric init: ', time.time()-t0
                _, __, cross_entropy, cross_entropy_summary, learnrate_summary = \
                    sess.run([train_step, miou_update_op, cross_entropy_loss,
                              cross_entropy_loss_summary_op, learning_rate_summary_op],
                             feed_dict={self.masterhandle: training_handle})
                # print 'post lost&trainstep sess run ', time.time() - t0

                # this was helpful to get the mIOU metric work:
                #   http://ronny.rest/blog/post_2017_09_11_tf_metrics/
                miou_score, miou_summary = sess.run([miou_score_op, miou_summary_op])
                print 'post miou sess run ', time.time() - t0
                summary_string_writer.add_summary(miou_summary, trainstep)
                summary_string_writer.add_summary(cross_entropy_summary, trainstep)
                summary_string_writer.add_summary(learnrate_summary, trainstep)

                print("Batch loss: {0:.2f}, Batch mIOU (%): {1:.2f}".format(cross_entropy, miou_score * 100))
                periodic_test_eval_steps = 100
                if (trainstep + 1) % periodic_test_eval_steps == 0:
                    save_path = saver.save(sess, chkpnt2save_path)
                    print("step ", trainstep, time.ctime(), "updated model in: %s" % save_path)
                    sess.run(running_metric_initializer)
                    num_test_images = 240  # 15 batches, ~1/4 of valid. set (shuffled..)
                    num_test_batches = num_test_images / self.args.batch_size
                    test_cross_entropy_arr = np.zeros(num_test_batches)
                    for _tb in range(num_test_batches):
                        _, test_cross_entropy_arr[_tb] = \
                            sess.run([miou_update_op, cross_entropy_loss],
                                     feed_dict={self.masterhandle: test_handle})
                    test_miou_score, test_miou_summary = sess.run([miou_score_op, test_miou_summary_op])
                    test_loss = np.mean(test_cross_entropy_arr)
                    test_loss_summary = tf.Summary()
                    test_loss_summary.value.add(tag='test cross entropy loss', simple_value=test_loss)
                    for _prevstep in range(periodic_test_eval_steps)[::-1]:
                        # write same value for all prev. steps, for compatibility with filters, etc.
                        summary_string_writer.add_summary(test_loss_summary, trainstep - _prevstep)
                        summary_string_writer.add_summary(test_miou_summary, trainstep - _prevstep)
                    print("----test mIOU: ", test_miou_score)
                    print("----test loss: ", test_loss)
                    print("----finished test eval...", time.ctime())
                # make a copy once in a while for case we start to overfit and model becomes worse...
                if trainstep % 3000 == 0:
                    saver.save(sess, chkpnt2save_path + '_{0}'.format(trainstep))

                # Nice example of usage Dataset iterators advantage.. (with queues it would be way more messy)
                if debug_loop_over_few_first_images and (trainstep + 1) % debug_loop_over_few_first_images == 0:
                    print '---'
                    sess.run(self.train_iter.initializer)

            save_path = saver.save(sess, chkpnt2save_path)
            print("Model saved in file: %s" % save_path)

        summary_string_writer.close()

    def setup(self):

        train_dataset = tf.contrib.data.TFRecordDataset([train_tfrec_fname])
        train_dataset = train_dataset.map(utils.tf_records.parse_record)

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

        test_dataset = tf.contrib.data.TFRecordDataset([test_tfrec_fname]).map(utils.tf_records.parse_record)
        test_dataset = test_dataset.map(lambda img, ann:
                                        utils.augmentation.nonrandom_rescale(img, ann, image_train_size))
        test_dataset = test_dataset.map(lambda image_, annotation_: (image_, tf.squeeze(annotation_)))
        test_dataset = test_dataset.shuffle(buffer_size=300).batch(self.args.batch_size)

        self.train_iter = train_dataset.repeat().make_initializable_iterator()
        self.test_iter = test_dataset.repeat().make_initializable_iterator()

        self.masterhandle = tf.placeholder(tf.string, shape=[])
        switching_iterator = tf.contrib.data.Iterator.from_string_handle(
            self.masterhandle, train_dataset.output_types, train_dataset.output_shapes)

        image_batch, self.annotation_batch = switching_iterator.get_next()

        self.upsampled_logits_batch = self.fcn_builder.build_net(image_batch_tensor=image_batch)
        self.feat_extractor_variables_mapping, self.new_vars = self.fcn_builder.get_pretrained_and_new_vars()

        self.init_fn = slim.assign_from_checkpoint_fn(model_path=self.checkpoint_path,
                                                      var_list=self.feat_extractor_variables_mapping)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a FCN(-based) segmentation net')
    parser.add_argument('--basenet', dest='basenet', type=str,
                        help='the base feature extractor',
                        default='vgg_16')
    parser.add_argument('--extended_arch', dest='extended_arch', type=bool,
                        help='if True use extended architecture',
                        default=False)
    parser.add_argument('--trainable_upsampling', dest='trainable_upsampling', type=bool,
                        help='if True use trainable_upsampling in the basic FCN architecture',
                        default=False)
    parser.add_argument('--fcn16', dest='fcn16', type=bool,
                        help='if True add the fcn16 skip connection',
                        default=False)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='batch size',
                        default=16)
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='num of epochs to train for',
                        default=30)
    parser.add_argument('--learnrate', dest='learnrate', type=float,
                        help='base learning rate',
                        default=1e-4)
    parser.add_argument('--difflr', dest='difflr', type=bool,
                        help='if True use x10 learning rate for new layers w.r.t pretrained',
                        default=False)  # if decaying rate - decay with same schedule retaining ratio
    parser.add_argument('--decaylr', dest='decaylr', type=bool,
                        help='if True decay learn rate from x10 to x0.1 base LR',
                        default=True)
    parser.add_argument('--pixels', dest='pixels', type=int,
                        help=' preprocess (interpolate large side & pad) each image (and annotation)'
                             ' to (pixels)X(pixels) size for the train',
                        default=512)

    if len(sys.argv) == 1:
        print("No args, running with defaults...")
        parser.print_help()

    args = parser.parse_args()

    checkpoint_path = {'vgg_16': vgg_checkpoint_path,
                       'resnet_v1_50': resnet50_checkpoint_path,
                       'resnet_v1_18': resnet18_checkpoint_path,
                       'inception_v1': inception_checkpoint_path,
                       'mobilenet_v1': mobilenet_checkpoint_path,
                       }.get(args.basenet)
    if not checkpoint_path:
        raise Exception("Not yet supported feature extractor")

    image_train_size = [args.pixels, args.pixels]

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
