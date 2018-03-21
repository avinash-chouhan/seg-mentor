import tensorflow as tf, numpy as np
import os, sys, time, argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# (!!) needed for code inside fcn_arch, utils..
sys.path.append("/home/finkel/hailo_repos/phase2-dl-research/slim_models/")

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

# image_train_size = [384, 384] # basic
# image_train_size = [600, 600] # doesn't work because not multiple of 32?
# image_train_size = [608, 608]
image_train_size = [512, 512]


number_of_classes = 21

pascal_voc_lut = utils.pascal_voc.pascal_segmentation_lut()
class_labels = pascal_voc_lut.keys()

DATASET_SIZE = 11127

# For debug mode of "go overfit over first images" - set a number
debug_loop_over_few_first_images = None
# debug_loop_over_few_first_images = 30


class Trainer:

    def __init__(self, net_func, checkpoint_path, narrowdeconv=False, batch_size=16):
        self.net_func = net_func
        self.narrowdeconv = narrowdeconv
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size

    def run_training(self, trainfolder, num_epochs=10, learning_rate=1e-6, decaylr=True,
                     new_vars_to_learn_faster=None, pretrained_vars=None):

        # timestamp = time.ctime().replace(' ', '-').replace(':', '_')
        # summary_string_writer = tf.summary.FileWriter(log_folder+'/'+timestamp+'/trainsummary')
        summary_string_writer = tf.summary.FileWriter(trainfolder + '/trainboard')
        # val_writer = tf.summary.FileWriter(log_folder + '/' + timestamp + '/validation')
        chkpnt2save_path = trainfolder + '/fcn32.ckpt'

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
        _probabilities = tf.nn.softmax(self.upsampled_logits_batch)

        global_step = tf.Variable(0, trainable=False)
        total_iterations = DATASET_SIZE * num_epochs / self.batch_size
        if decaylr:
            ds = 500

            dr = 0.1 ** (ds*1.0 / total_iterations)
            # decay x100 across the whole range, e.g. x10 base-lr down to x0.1 base-lr for the weak variant.
            # TODO - parameterize?.... also note 0.1 vs. 0.01 workaround TODO figure out why..)
            # (one hypothesis - maybe our steps below include two gobal steps somehow...
            strong_learning_rate = tf.train.exponential_decay(learning_rate * 100, global_step=global_step,
                                                              decay_steps=ds, decay_rate=dr, staircase=True)
            weak_learning_rate = tf.train.exponential_decay(learning_rate * 10, global_step=global_step,
                                                            decay_steps=ds, decay_rate=dr, staircase=True)
        else:
            weak_learning_rate = learning_rate
            strong_learning_rate = 10 * learning_rate

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope("adam_vars"):
                if new_vars_to_learn_faster:
                    train_op1 = tf.train.AdamOptimizer(learning_rate=strong_learning_rate,
                                                       epsilon=1e-3, beta1=0.99). \
                        minimize(cross_entropy_loss, var_list=new_vars_to_learn_faster, global_step=global_step)
                    train_op2 = tf.train.AdamOptimizer(learning_rate=weak_learning_rate,
                                                       epsilon=1e-3, beta1=0.99). \
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
        # from the calculation of the metric because of the way TensorFlow works
        # see http://ronny.rest/blog/post_2017_09_11_tf_metrics/
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

                print("Batch loss: {0:.2f}, Batch mIOU (%): {1:.2f}".format(cross_entropy, miou_score*100))
                periodic_test_eval_steps = 100
                if (trainstep+1) % periodic_test_eval_steps == 0:
                    save_path = saver.save(sess, chkpnt2save_path)
                    print("step ", trainstep, time.ctime(), "updated model in: %s" % save_path)
                    sess.run(running_metric_initializer)
                    num_test_images = 240  # 15 batches, ~1/4 of valid. set (shuffled..)
                    num_test_batches = num_test_images / self.batch_size
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
                        summary_string_writer.add_summary(test_loss_summary, trainstep-_prevstep)
                        summary_string_writer.add_summary(test_miou_summary, trainstep-_prevstep)
                    print("----test mIOU: ", test_miou_score)
                    print("----test loss: ", test_loss)
                    print("----finished test eval...", time.ctime())
                # make a copy once in a while for case we start to overfit and model becomes worse...
                if trainstep % 3000 == 0:
                    saver.save(sess, chkpnt2save_path+'_{0}'.format(trainstep))

                if debug_loop_over_few_first_images and (trainstep + 1) % debug_loop_over_few_first_images == 0:
                    print '-'
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
            # TODO - consider the distort-color (disabled in Daniil's too,) and maybe other augmentations...
            # train_dataset = train_dataset.map(lambda image_, annotation_:
            #                       (utils.augmentation.distort_randomly_image_color(image_), annotation_))
            train_dataset = train_dataset.map(lambda image_, annotation_: (image_, tf.squeeze(annotation_)))
            train_dataset = train_dataset.shuffle(buffer_size=3000)
        else:
            train_dataset = train_dataset.map(lambda image_, annotation_:
                                              utils.augmentation.nonrandom_rescale(image_, annotation_,
                                                                                   image_train_size))
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = tf.contrib.data.TFRecordDataset([test_tfrec_fname]).map(utils.tf_records.parse_record)
        test_dataset = test_dataset.map(lambda img, ann:
                                        utils.augmentation.nonrandom_rescale(img, ann, image_train_size))
        test_dataset = test_dataset.map(lambda image_, annotation_: (image_, tf.squeeze(annotation_)))
        test_dataset = test_dataset.shuffle(buffer_size=300).batch(self.batch_size)

        self.train_iter = train_dataset.repeat().make_initializable_iterator()
        self.test_iter = test_dataset.repeat().make_initializable_iterator()

        self.masterhandle = tf.placeholder(tf.string, shape=[])
        switching_iterator = tf.contrib.data.Iterator.from_string_handle(
            self.masterhandle, train_dataset.output_types, train_dataset.output_shapes)

        # image_batch, self.annotation_batch = self.train_iter.get_next()
        image_batch, self.annotation_batch = switching_iterator.get_next()

        self.upsampled_logits_batch, self.feat_extractor_variables_mapping, self.new_vars = \
            fcn_arch.fcn(image_batch_tensor=image_batch, number_of_classes=number_of_classes,
                         is_training=True, net_func=self.net_func,
                         narrowdeconv=self.narrowdeconv, fcn16=self.fcn16)

        self.init_fn = slim.assign_from_checkpoint_fn(model_path=self.checkpoint_path,
                                                      var_list=self.feat_extractor_variables_mapping)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--basenet', dest='basenet', type=str,
                        help='the base feature extractor',
                        default='mobilenet')
    parser.add_argument('--narrowdeconv', dest='narrowdeconv', type=bool,
                        help='if True use 1K->21 1x1 then 21->21 deconv rather than wide 1K->21 deconv',
                        default=False)
    parser.add_argument('--fcn16', dest='fcn16', type=bool,
                        help='if True add the fcn16 skip connection',
                        default=False)
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        help='batch size',
                        default=16)
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='num of epochs to train for',
                        default=40)
    parser.add_argument('--learnrate', dest='learnrate', type=float,
                        help='base learning rate',
                        default=1e-6)
    parser.add_argument('--difflr', dest='difflr', type=bool,
                        help='if True use x10 learning rate for new layers w.r.t pretrained',
                        default=False)  # ?? does it become true at any --difflr option value, even without one?
    parser.add_argument('--decaylr', dest='decaylr', type=bool,
                        help='if True decay learn rate from x10 to x0.1 base LR',
                        default=False)
    parser.add_argument('--pixels', dest='pixels', type=int,
                        help=' preprocess (interpolate&crop) each image (and annotation)'
                             ' to (pixels)X(pixels) size for the train',
                        default=384)

    if len(sys.argv) == 1:
        print "No args, running with defaults..."
        parser.print_help()

    args = parser.parse_args()

    try:
        fe_dict = {'net_func': args.basenet,
                   'checkpoint_path': {'vgg': vgg_checkpoint_path,
                                       'resnet_v1_50': resnet50_checkpoint_path,
                                       'resnet_v1_18': resnet18_checkpoint_path,
                                       'inception_v1': inception_checkpoint_path,
                                       'mobilenet_v1': mobilenet_checkpoint_path,
                                     }[args.basenet]
                   }
    except:
        raise Exception("Not yet supported feature extractor")

    fe_dict['batch_size'] = args.batchsize
    fe_dict['narrowdeconv'] = args.narrowdeconv

    trainer = Trainer(**fe_dict)
    trainer.fcn16 = args.fcn16
    image_train_size = [args.pixels, args.pixels]
    trainer.setup()

    import datetime
    today = datetime.date.today().strftime('%b%d')
    prefix = '{today}_{net}__'.format(net=fe_dict['net_func'], today=today)
    thisrunnum = 1 + max([0] + [int(f[len(prefix):]) for f in os.listdir('./tmp') if prefix in f])
    trainfolder = './tmp/' + prefix + str(thisrunnum)

    os.makedirs(trainfolder)

    # open(os.path.join(trainfolder, 'runargs'), 'w').write(str(args.__dict__))
    import json
    json.dump(args.__dict__, open(os.path.join(trainfolder, 'runargs'), 'w'), indent=2)

    # redirecting PRINT statements into log in same folder...
    sys.stdout = sys.stderr = open(os.path.join(trainfolder, 'runlog'), 'w', 1)  # line buffered

    if args.difflr:
        trainer.run_training(trainfolder=trainfolder, num_epochs=args.epochs,
                             learning_rate=args.learnrate,
                             decaylr=args.decaylr,
                             new_vars_to_learn_faster=trainer.new_vars,
                             pretrained_vars=trainer.feat_extractor_variables_mapping.values())  # , learning_rate=lr)
    else:
        trainer.run_training(trainfolder=trainfolder, num_epochs=args.epochs,
                             learning_rate=args.learnrate,
                             decaylr=args.decaylr,
                             )
