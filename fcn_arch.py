import tensorflow as tf
from utils.upsampling import bilinear_upsample_weights

# ..assumes slim_models is added to path
from nets import vgg, mobilenet_v1, inception_v1, resnet_v1, resnet_utils
from preprocessing import vgg_preprocessing, pytorch_resnet_preprocessing
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN

slim = tf.contrib.slim


def fcn(image_batch_tensor,
        number_of_classes,
        is_training=True,
        net_func=vgg.vgg_16,
        narrowdeconv=False,
        fcn16=False):
    """
    Returns the FCN-32s/16s model definition.
    The function returns the model definition of a network that was described
    in 'Fully Convolutional Networks for Semantic Segmentation' by Long et al.

    The network uses the injected feature extractor on input image (batch tensor),
      to get a a x32 smaller resolution feature map, in a fully convolutional manner,
      before final average pool (if not vgg) and the classificaiton layers removed...
    Then it upsamples x2 with fixed (bilinear) or learnable deconvolution,
      optionally with 1x1 classification layer ("narrow deconv") or alternatively,
      a direct 512->21 ("wide") deconv
    Then it optionally adds a skip connection  (in FCN16 manner) of 1x1 convolution,
      operated on higher resolution feature map earlier in the net (using the endpoints collection).
    Finally the 21-deep logits map is x16 upsampled to original resolution**

    ** note that if the image size is not of the factor 32, the prediction of different size
    will be delivered. To adapt the network for an any size input use 
    adapt_network_for_any_size_input(fcn, 32)
    
    Parameters
    ----------
    net_func = the feature extractor network defined as a TF-SLIM inference function
    image_batch_tensor : [batch_size, height, width, depth] tensor
    number_of_classes : int, e.g. 21 for PASCAL VOC
    is_training : boolean, to be propagated into the net_func()
        Affects dropout and batchnorm layers of the feature extractor
    fcn16 - boolean, if True add skip connection. We don't use two-stage training,
              if necessary we can emulated that via differential learning schedule...
    Returns
    -------
    upsampled_logits : [batch_size, height, width, number_of_classes] Tensor
        Tensor with logits representing predictions for each class.
        Be careful, the output can be of different size compared to input,
        use adapt_network_for_any_size_input to adapt network for any input size.
        Otherwise, the input images sizes should be of multiple 32.

    net_variables_mapping : dict {string: variable}
        Dict which maps the FCN model's variables to feature-extractor (e.g. VGG-16) checkpoint variables
        names. We need this to initilize the weights of FCN model ,
        with the pre-trained weights of the feature extractor in a slim_models checkpoint
    """

    with tf.variable_scope("fcn_32s") as fcn_32s_scope:

        upsample_factor = 32

        # Pre-processing:
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

        trainable_upsampling = True
        pre_fc_ep = 'pre_average_pool'

        if net_func == 'vgg':
            net_func = vgg.vgg_16
            arg_scope_func = vgg.vgg_arg_scope
            mean_centered_image_batch = vgg_preprocess(image_batch_tensor)
            trainable_upsampling = False
            # pre_fc_ep = 'fcn_32s/vgg_16/pool5' # before FC
            pre_fc_ep = 'fcn_32s/vgg_16/fc7'  # after global FC (convolutionalized to 7x7)and small 2nd FC
            # and just before final classification before FC
            last_class_fc = 'fc8'

        elif net_func == 'inception_v1':
            net_func = inception_v1.inception_v1
            arg_scope_func = inception_v1.inception_v1_arg_scope
            mean_centered_image_batch = inception_preprocess(image_batch_tensor)
            last_class_fc = 'Conv2d_0c_1x1'
            # pre last pool and inception block
            pre_dnsmp_ep = 'Mixed_4f'

        elif net_func == 'resnet_v1_50':
            net_func = resnet_v1.resnet_v1_50
            arg_scope_func = resnet_utils.resnet_arg_scope
            mean_centered_image_batch = vgg_preprocess(image_batch_tensor)
            # pre-last unit if block3, before last striding in last unit of block3
            pre_dnsmp_ep = 'fcn_32s/resnet_v1_50/block3/unit_5/bottleneck_v1'
            last_class_fc = 'resnet_v1_50/logits'

        elif net_func == 'resnet_v1_18':
            net_func = resnet_v1.resnet_v1_18
            arg_scope_func = resnet_utils.resnet_arg_scope
            mean_centered_image_batch = pytorch_resnet_preprocessing.normalize_gen(image_batch_tensor)
            # end of block 3, before last striding in first unit of block4
            pre_dnsmp_ep = 'fcn_32s/resnet_v1_18/block3/unit_2/res_block_v1'
            last_class_fc = 'resnet_v1_18/logits'

        elif net_func == 'mobilenet_v1':
            net_func = mobilenet_v1.mobilenet_v1
            arg_scope_func = mobilenet_v1.mobilenet_v1_arg_scope
            mean_centered_image_batch = inception_preprocess(image_batch_tensor)
            # end of layer 11 - before the stride-2 layer 12 and stride-1 layer 13 (and last)
            pre_dnsmp_ep = 'Conv2d_11_pointwise'
            last_class_fc = 'Conv2d_1c_1x1'
        else:
            raise Exception('net func {0} not supported!'.format(net_func))

        # Use arg scope of feature extractor to create both the encoder and new layers of decoder..
        #  (e.g. batchnorm, weight decay, etc.)
        with slim.arg_scope(arg_scope_func()):

            # Create all operations (inc. last layer which we possibly won't use..)
            logits, end_points = net_func(mean_centered_image_batch,
                                          num_classes=number_of_classes,
                                          is_training=is_training,
                                          spatial_squeeze=False,
                                          global_pool=False,
                                          fc_conv_padding='SAME')

            # Calculate the ouput size of the upsampled tensor
            downsampled_logits_shape = tf.shape(logits)
            upsampled_logits_shape = tf.stack([
                downsampled_logits_shape[0],
                downsampled_logits_shape[1] * upsample_factor,
                downsampled_logits_shape[2] * upsample_factor,
                downsampled_logits_shape[3]
            ])

            # Perform the partial trainable upsampling and (optionally) add skip branch:

            # print 'end_points.keys() : \n', end_points.keys()
            if narrowdeconv:
                deconv_in_fcn32 = slim.conv2d(end_points[pre_fc_ep], number_of_classes, [1, 1], scope='1x1_fcn32',
                                              activation_fn=None, normalizer_fn=None)
            else:  # wide deconv EXPERIMENTAL: also use wide post-pool context
                global_and_local = tf.concat((end_points['pre_average_pool'], end_points['post_average_pool']), -1)
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    deconv_in_fcn32 = slim.conv2d(global_and_local, 256, [1, 1], scope='1x1_fcn32_A')
                    deconv_in_fcn32 = slim.conv2d(deconv_in_fcn32, 128, [1, 1], scope='1x1_fcn32_B',
                                                  activation_fn=None, normalizer_fn=None)

            logits = slim.conv2d_transpose(deconv_in_fcn32, number_of_classes,
                                           kernel_size=[4, 4], stride=[2, 2], scope='deconv32')

            # ..upsampled by 2 already (x16 left..)
            upsample_factor /= 2

            if fcn16:
                fcn16_branch_logits = slim.conv2d(end_points[pre_dnsmp_ep],
                                                  number_of_classes,
                                                  [1, 1],
                                                  activation_fn=None,
                                                  normalizer_fn=None,
                                                  weights_initializer=tf.zeros_initializer,
                                                  scope='1x1_fcn16')
                logits = tf.add(logits, fcn16_branch_logits)

        # -------- finished the argscope (batchnorms etc.)

        # final bilinear-interp. upsampling
        upsample_filter_np = bilinear_upsample_weights(upsample_factor, number_of_classes)
        upsample_filter_tensor = tf.constant(upsample_filter_np)
        upsampled_logits = tf.nn.conv2d_transpose(logits,
                                                  upsample_filter_tensor,
                                                  output_shape=upsampled_logits_shape,
                                                  strides=[1, upsample_factor, upsample_factor, 1])

        # Map the original net (e.g. vgg-16) variable names to the variables in our model. This is done
        # to make it possible to use assign_from_checkpoint_fn() while providing this mapping.

        # Here we remove the part of a name of the variable that is responsible for the current variable scope
        # Note: name_scope only affects operations and variable scope is actually represented by .name
        feat_extractor_variables = slim.get_variables(fcn_32s_scope)
        net_variables_mapping = {variable.name[len(fcn_32s_scope.name) + 1:-2]: variable
                                 for variable in feat_extractor_variables}

        pretrained_net_variables_mapping = {k: v for k, v in net_variables_mapping.items() \
                                            if 'deconv' not in k and last_class_fc not in k and \
                                            '1x1_fcn' not in k and 'fcn16' not in k}

        new_vars = [var for var in feat_extractor_variables
                    if var.name not in pretrained_net_variables_mapping.keys()]

    return upsampled_logits, pretrained_net_variables_mapping, new_vars
