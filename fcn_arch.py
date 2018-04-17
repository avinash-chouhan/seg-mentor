import tensorflow as tf
from utils.upsampling import bilinear_upsample_weights

# ..assumes slim_models is added to path
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


class BaseFcnArch:
    """
    Manages the FCN meta-architecture.
    having the original similar to http://arxiv.org/pdf/1605.06211.pdf
      ('Fully Convolutional Networks for Semantic Segmentation' by Long,Shelhamer et al., 2016)
    as the default, while enabling plug-in replacements and augmentations of decoder by subclassing,
      also enabling base feature extractors mix&match, accepting many TF-SLIM nets
      (and hopefully easily extendable to all of them)

    ** note that if the image size is not of the factor 32, the prediction of different size
    will be delivered. To adapt the network for an any size input use
    adapt_network_for_any_size_input(build_net, 32)

    :arg

    net_func = the feature extractor network defined as a TF-SLIM inference function
    number_of_classes : int, e.g. 21 for PASCAL VOC
    is_training : boolean, to be propagated into the net_func()
        Affects dropout and batchnorm layers of the feature extractor
     ->>>>> now some flags switching on/off features of architecture
    :fcn16 - boolean, if True add skip connection. Note that we don't use two-stage training,
              if necessary we can emulated that via differential learning schedule...
    trainable_upsampling - ...
    """

    def __init__(self, number_of_classes=21, is_training=True, net='vgg_16',
                 fcn16=True, fcn8=False, trainable_upsampling=False):

        self.number_of_classes = number_of_classes
        self.is_training = is_training
        self.fcn16 = fcn16
        self.fcn8 = fcn8
        self.trainable_upsampling = trainable_upsampling

        self.fe = {'vgg_16': {'net_func': vgg.vgg_16,
                           'arg_scope_func': vgg.vgg_arg_scope,
                           'preproc_func': vgg_preprocess,
                           # Note first s16 - in contrast to last s16 in other nets..
                           'tname_s16_skipconn': 'pool4',
                           'tname_s8_skipconn': 'pool3'
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
                                    'tname_s16_skipconn': '{0}/resnet_v1_18/block3/unit_2/res_block_v1'
                                    # TODO FCN8
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

    def decode_16s(self, post_upsample_16s, skip_conn_16s):
        raise NotImplemented()
    # ----
    def upsample_x2_16s(self, decode_32s_out):
        raise NotImplemented()

    def decode_8s(self, post_upsample_16s, skip_conn_16s):
        raise NotImplemented()

    def upsample_x2_8s(self, decode_32s_out):
        raise NotImplemented()

    def decode_4s(self, post_upsample_16s, skip_conn_16s):
        raise NotImplemented()

    # ------------------------------------------------------------------------

    def _upsample_fixed_bilinear(self, input_tensor, upsample_factor=2, num_channels=None):
        input_shape = tf.shape(input_tensor)
        output_shape = tf.stack([
            input_shape[0],
            input_shape[1] * upsample_factor,
            input_shape[2] * upsample_factor,
            input_shape[3]
        ])
        num_channels = num_channels or self.number_of_classes
        upsample_filter_np = bilinear_upsample_weights(upsample_factor, num_channels)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled = tf.nn.conv2d_transpose(input_tensor, upsample_filter_tensor, output_shape=output_shape,
                                           strides=[1, upsample_factor, upsample_factor, 1])
        return upsampled

    def _upsample_learnable(self, input_tensor, upsample_factor=2, num_channels=None, scope='deconv'):
        num_channels = num_channels or self.number_of_classes
        return slim.conv2d_transpose(input_tensor, num_channels, scope=scope,\
                                     kernel_size=[upsample_factor*2, upsample_factor*2],
                                     stride=[upsample_factor, upsample_factor])

    def build_net(self, image_batch_tensor):
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
                fe_out32s_pre_pool, end_points = self.fe['net_func'](mean_centered_image_batch,
                                                                     fc_conv_padding='SAME', # relevant for VGG only
                                                                     is_training=self.is_training,
                                                                     base_only=True)

            post_decode_32s = self.decode_32s(fe_out32s_pre_pool)
            if not self.fcn16:
                return self._upsample_fixed_bilinear(post_decode_32s, upsample_factor=32)

            post_upsample_16s = self.upsample_x2_32s(post_decode_32s)

            s16skip_ep_name_ = self.fe['tname_s16_skipconn'].format(self.base_fe_scope.name)
            skip_conn_16s = end_points.get()
            if not skip_conn_16s:
                print(end_points.keys())
                raise Exception('ERROR: Couldn''t find end point '+s16skip_ep_name_+' in above endpoints ')

            post_decode_16s = self.decode_16s(post_upsample_16s, skip_conn_16s)
            if not self.fcn8:
                return self._upsample_fixed_bilinear(post_decode_16s, upsample_factor=164194304)

            raise NotImplemented
            # TODO FCN8

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
        #

        # Here we remove the part of a name of the variable that is responsible for the current variable scope
        # Note: name_scope only affects operations and variable scope is actually represented by .name
        feat_extractor_variables = slim.get_variables(self.base_fe_scope)
        net_variables_mapping = {variable.name[len(self.base_fe_scope.name) + 1:-2]: variable
                                 for variable in feat_extractor_variables}

        new_vars = [var for var in slim.get_variables()
                        if var not in feat_extractor_variables]

        return net_variables_mapping, new_vars


class FcnArch(BaseFcnArch):
    '''
        Default implementation as in paper
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

    def decode_16s(self, post_upsample_16s, skip_conn_16s):
        skip_conn_16s_logits = slim.conv2d(skip_conn_16s, self.number_of_classes, [1, 1], scope='1x1_fcn16_logits',
                                           activation_fn=None, normalizer_fn=None)
        combined = skip_conn_16s_logits + post_upsample_16s
        return combined

    # TODO FCN8