import sys
import tensorflow as tf
from utils.upsampling import bilinear_upsample_weights

# assuming our fork of tensorflow models/research/slim is cloned side-by-side with current repo
sys.path.append("../tf-models-hailofork/research/slim/")
from nets import vgg, mobilenet_v1, inception_v1, resnet_v1, resnet_utils
from preprocessing import vgg_preprocessing, pytorch_resnet_preprocessing
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN

slim = tf.contrib.slim


def vgg_preprocess(image):
    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image)
    # Subtract the mean pixel value from each pixel
    return image_float - [_R_MEAN, _G_MEAN, _B_MEAN]


def inception_preprocess(image):
    # image_float = tf.image.convert_image_dtype(image, dtype=tf.float32) # RESCALES!
    image_float = tf.to_float(image) / 255.0
    image_float = tf.subtract(image_float, 0.5)
    return tf.multiply(image_float, 2.0)

def decoder_arg_scope_func(weight_decay=1e-4, use_batch_norm=True):
    with slim.arg_scope([slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm if use_batch_norm else None)\
         as arg_sc:
                return arg_sc

class BaseFcnArch:
    """
    Manages the generic FCN-derived meta-meta-architecture.

    Default subclass is FcnArch, see below.
    Enabling plug-in replacements and augmentations of decoder by subclassing,
      also enabling base feature extractors mix&match, accepting many TF-SLIM nets
      (and hopefully easily extendable to all of them)

    ** note that if the image size is not of the factor 32, the prediction of different size
    will be delivered. To adapt the network for an any size input use
    adapt_network_for_any_size_input(build_net, 32)

    :Constructor Arguments:

    :net_func:  = the feature extractor network defined as a TF-SLIM inference function
    :number_of_classes: int, e.g. 21 for PASCAL VOC
    :is_training: boolean, to be propagated into the net_func()
        Affects dropout and batchnorm layers of the feature extractor
     ---------- now some flags switching on/off features of architecture -------
    :fcn16: - boolean, if True add skip connection. Note that we don't use two-stage training,
              if necessary we can emulated that via differential learning schedule...
    :trainable_upsampling: - in FCN, whether to use learnable deconv or simple bilinear-interpolation
    """

    def __init__(self, number_of_classes=21, force_is_training_val=None, net='vgg_16',
                 fcn16=True, fcn8=False, fcn4=False, trainable_upsampling=False):

        self.number_of_classes = number_of_classes
        self.is_training_val = force_is_training_val
        self.fcn16 = fcn16
        self.fcn8 = fcn8
        self.fcn4 = fcn4
        self.trainable_upsampling = trainable_upsampling
        self.decoder_arg_scope_func = decoder_arg_scope_func
        self.fe = {'vgg_16': {'net_func': vgg.vgg_16,
                           'arg_scope_func': vgg.vgg_arg_scope,
                           'preproc_func': vgg_preprocess,
                           # Note first s16 - in contrast to last s16 in other nets..
                           'tname_s16_skipconn': 'first_s16',
                           'tname_s8_skipconn': 'first_s8'
                           #'logits_opname': 'fc8'
                            },
                   'inception_v1': {'net_func': inception_v1.inception_v1,
                                    'arg_scope_func': inception_v1.inception_v1_arg_scope,
                                    'preproc_func': inception_preprocess,
                                    # ..last s16 - pre last pool and 2 inception blocks that follow, at /32 level:
                                    'tname_s16_skipconn': 'Mixed_4f'
                            #        'logits_opname': 'Conv2d_0c_1x1'
                                    },
                   'resnet_v1_50': {'net_func': resnet_v1.resnet_v1_50,
                                    'arg_scope_func': resnet_utils.resnet_arg_scope,
                                    'preproc_func': vgg_preprocess,
                                    # ..last s16 - out of 5th (out of 6) unit of block3,
                                    #   just before stride to /32 in last (6th) unit of block3:
                                    'tname_s16_skipconn': '{0}/resnet_v1_50/block3/unit_5/bottleneck_v1'
                                    #'logits_opname': 'resnet_v1_50/logits'
                                    },
                   'resnet_v1_18': {'net_func': resnet_v1.resnet_v1_18,
                                    'arg_scope_func': resnet_utils.resnet_arg_scope,
                                    'preproc_func': pytorch_resnet_preprocessing.normalize_gen,
                                    # ..last s16 - out of block 3,
                                    #   just before stride to /32 in first unit of block4
                                    'tname_s16_skipconn': '{0}/resnet_v1_18/block3',
                                    'tname_s8_skipconn': '{0}/resnet_v1_18/block2',
                                    'tname_s4_skipconn': '{0}/resnet_v1_18/block1'
                                    #'logits_opname': 'resnet_v1_18/logits'
                                    },
                   'mobilenet_v1': {'net_func': mobilenet_v1.mobilenet_v1,
                                    'arg_scope_func': mobilenet_v1.mobilenet_v1_arg_scope,
                                    'preproc_func': inception_preprocess,
                                    # ..end of layer 11 - before the stride-2 layer 12 and stride-1 layer 13 (and last)
                                    'tname_s16_skipconn': 'Conv2d_11_pointwise'
                                    # TODO FCN8
                                    #'logits_opname': 'Conv2d_1c_1x1'
                                    },
            }.get(net)

        if self.fe is None:
            raise Exception('net {0} not supported!'.format(net))

    # ------ blocks to be implemented in subclasses to control the decoder ----
    def decode_32s(self, fe_out_pre_pool):
        raise NotImplemented()

    def upsample_x2_32s(self, decode_32s_out):
        raise NotImplemented()

    def combine_16s(self, post_upsample_16s, skip_conn_16s):
        raise NotImplemented()
    # ----
    def upsample_x2_16s(self, combine_16s_out):
        raise NotImplemented()

    def combine_8s(self, post_upsample_8s, skip_conn_8s):
        raise NotImplemented()

    # ----
    def upsample_x2_8s(self, combine_8s_out):
        raise NotImplemented()

    def combine_4s(self, post_upsample_4s, skip_conn_4s):
        raise NotImplemented()

    # ----
    def final_x4_upsample(self, combine_4s_out):
        raise NotImplemented()

    # ------------------------------------------------------------------------

    def _upsample_fixed_bilinear(self, input_tensor, upsample_factor=2, num_channels=None):
        '''Use bilinear interpolation to upsamlpe an image by a given upsample_factor'''
        input_shape = input_tensor.shape
        upsampled_tensor = tf.image.resize_bilinear(
                images=input_tensor,
                size=[input_shape[1]*upsample_factor, input_shape[2]*upsample_factor],
                align_corners=True
                )
        return upsampled_tensor
        # input_shape = tf.shape(input_tensor)
        # output_shape = tf.stack([
        #     input_shape[0],
        #     input_shape[1] * upsample_factor,
        #     input_shape[2] * upsample_factor,
        #     input_shape[3]
        # ])
        # num_channels = num_channels or self.number_of_classes
        # upsample_filter_np = bilinear_upsample_weights(upsample_factor, num_channels)
        # upsample_filter_tensor = tf.constant(upsample_filter_np)
        # upsampled = tf.nn.conv2d_transpose(input_tensor, upsample_filter_tensor, output_shape=output_shape,
        #                                    strides=[1, upsample_factor, upsample_factor, 1])
        # return upsampled

    def _upsample_learnable(self, input_tensor, upsample_factor=2, num_channels=None, scope='deconv'):
        num_channels = num_channels or self.number_of_classes
        return slim.conv2d_transpose(input_tensor, num_channels, scope=scope,\
                                     kernel_size=[upsample_factor*2, upsample_factor*2],
                                     stride=[upsample_factor, upsample_factor])

    def build_net(self, image_batch_tensor): #, force_is_training=None):
        """
            :arg
                image_batch_tensor : [batch_size, height, width, depth] tensor

            :returns:
                upsampled_logits : [batch_size, height, width, number_of_classes] Tensor
                    Tensor with logits representing predictions for each class.
                    Be careful, the output can be of different size compared to input,
                    use adapt_network_for_any_size_input to adapt network for any input size.
                    Otherwise, the input images sizes should be multiple of 32
        """

        # Pre-processing:
        mean_centered_image_batch = self.fe['preproc_func'](image_batch_tensor)

        # Use arg scope of feature extractor to create both the encoder and new layers of decoder..
        #  (e.g. batchnorm, weight decay, etc.)
        with slim.arg_scope(self.fe['arg_scope_func']()):

            with tf.variable_scope("base_fe_scope") as self.base_fe_scope:
                self.is_training_ph = tf.placeholder(tf.bool, name='is_training')
                is_training = self.is_training_val if self.is_training_val is not None else self.is_training_ph
                fe_out32s_pre_pool, end_points = self.fe['net_func'](mean_centered_image_batch,
                                                                     fc_conv_padding='SAME', # relevant for VGG only
                                                                     is_training=is_training,
                                                                     base_only=True)
            # Add more arg-scoping for decoder
            with slim.arg_scope(self.decoder_arg_scope_func()):
                post_decode_32s = self.decode_32s(fe_out32s_pre_pool)
                if not self.fcn16:
                    return self._upsample_fixed_bilinear(post_decode_32s, upsample_factor=32)
                # ------------------
                #  16
                post_upsample_16s = self.upsample_x2_32s(post_decode_32s)

                s16skip_ep_name = self.fe['tname_s16_skipconn'].format(self.base_fe_scope.name)
                skip_conn_16s = end_points.get(s16skip_ep_name)
                if skip_conn_16s is None:
                    self.debug_endpoints(end_points, s16skip_ep_name)

                post_combine_16s = self.combine_16s(post_upsample_16s, skip_conn_16s)
                if not self.fcn8:
                    return self._upsample_fixed_bilinear(post_combine_16s, upsample_factor=16)
                # ------------------
                # 8
                post_upsample_8s = self.upsample_x2_16s(post_combine_16s)

                s8skip_ep_name = self.fe['tname_s8_skipconn'].format(self.base_fe_scope.name)
                skip_conn_8s = end_points.get(s8skip_ep_name)
                if skip_conn_8s is None:
                    self.debug_endpoints(end_points, s8skip_ep_name)

                post_combine_8s = self.combine_8s(post_upsample_8s, skip_conn_8s)
                if not self.fcn4:
                    return self._upsample_fixed_bilinear(post_combine_8s, upsample_factor=8)
                # ------------------
                # 4
                post_upsample_4s = self.upsample_x2_8s(post_combine_8s)

                s4skip_ep_name = self.fe['tname_s4_skipconn'].format(self.base_fe_scope.name)
                skip_conn_4s = end_points.get(s4skip_ep_name)
                if skip_conn_4s is None:
                    self.debug_endpoints(end_points, s8skip_ep_name)

                post_combine_4s = self.combine_4s(post_upsample_4s, skip_conn_4s)
                return self.final_x4_upsample(post_combine_4s)

    def debug_endpoints(self, end_points, name):
        for k in end_points.keys():
            print k
        raise Exception('ERROR: Couldn''t find end point ' + name + ' in above endpoints ')

    def get_pretrained_and_new_vars(self):
        """
            Map the original net (e.g. vgg-16) variable names to the variables in our model. This is done
            to make it possible to use assign_from_checkpoint_fn() while providing this mapping.
            (!) assumes build_net() has been run

            :returns:
                pretrained_net_variables_mapping : dict {string: variable}
                    Dict which maps the FCN model's variables of those layers inherited from FE,
                     to feature-extractor (e.g. VGG-16) checkpoint variables
                    names. We need this to initialize the weights of FCN model ,
                    with the pre-trained weights of the feature extractor in a slim_models checkpoint

                new_vars       :         dict of new variables (layers beyond FE)
        """
        feat_extractor_variables = slim.get_variables(self.base_fe_scope)
        net_variables_mapping = {variable.name[len(self.base_fe_scope.name) + 1:-2]: variable
                                 for variable in feat_extractor_variables}

        new_vars = [var for var in slim.get_variables()
                        if var not in feat_extractor_variables]

        return net_variables_mapping, new_vars


class FcnArch(BaseFcnArch):
    '''
        Implementation of FCN:
        http://arxiv.org/pdf/1605.06211.pdf
        ('Fully Convolutional Networks for Semantic Segmentation' by Long,Shelhamer et al., 2016)
    '''
    def __init__(self, *args, **kwargs):
        BaseFcnArch.__init__(self, *args, **kwargs)

    def decode_32s(self, fe_out_pre_pool):
        return slim.conv2d(fe_out_pre_pool, self.number_of_classes, [1, 1], scope='1x1_fcn32_logits',
                           activation_fn=None, normalizer_fn=None)

    def upsample_x2_32s(self, decode_32s_out):
        if self.trainable_upsampling:
            return self._upsample_learnable(decode_32s_out)
        else:
            return self._upsample_fixed_bilinear(decode_32s_out)

    def combine_16s(self, post_upsample_16s, skip_conn_16s):
        skip_conn_16s_logits = slim.conv2d(skip_conn_16s, self.number_of_classes, [1, 1], scope='1x1_fcn16_logits',
                                           activation_fn=None, normalizer_fn=None,
                                           weights_initializer=tf.zeros_initializer)
        combined = skip_conn_16s_logits + post_upsample_16s
        return combined

    def upsample_x2_16s(self, combine_16s_out):
        if self.trainable_upsampling:
            return self._upsample_learnable(combine_16s_out)
        else:
            return self._upsample_fixed_bilinear(combine_16s_out)

    def combine_8s(self, post_upsample_8s, skip_conn_8s):
        skip_conn_8s_logits = slim.conv2d(skip_conn_8s, self.number_of_classes, [1, 1], scope='1x1_fcn8_logits',
                                           activation_fn=None, normalizer_fn=None,
                                           weights_initializer=tf.zeros_initializer)
        combined = skip_conn_8s_logits + post_upsample_8s
        return combined

