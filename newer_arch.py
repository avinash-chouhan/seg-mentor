from fcn_arch import *


class LinkNet(BaseFcnArch):
    '''
        LinkNet over our framework
    '''
    def __init__(self, *args, **kwargs):
        self.fcn16 = self.fcn8 = self.fcn4 = True
        # ! Note the forcing of skip-conn config should come before base arch construction...
        BaseFcnArch.__init__(self, *args, **kwargs)
        # self.end_ch = 256  # resnet-18 . others - TODO

    def _decode_block(self, input_tensor, n_ch):
        '''assuming input_tensor has 2*n_ch channels as in paper'''
        deconv_in = slim.conv2d(input_tensor, n_ch/2, [1, 1])
        deconv_out = slim.conv2d_transpose(deconv_in,  n_ch/2, \
                                           kernel_size=[6, 6], stride=[2, 2])
        return slim.conv2d(deconv_out, n_ch, [1, 1], activation_fn=None)

    def decode_32s(self, fe_out_pre_pool):
        ''' topmost horizontal link as per LinkNet diagram'''
        return fe_out_pre_pool

    def upsample_x2_32s(self, decode_32s_out):
        ''' Decoder-Block-4 as per LinkNet diagram '''
        return self._decode_block(decode_32s_out, 256)

# -----------------
    def combine_16s(self, post_upsample_16s, skip_conn_16s):
        ''' the topmost PLUS as per LinkNet diagram '''
        #return post_upsample_16s + skip_conn_16s

        # TEMP!!
        return tf.nn.relu(post_upsample_16s + slim.conv2d(skip_conn_16s, 256, [1, 1],
                                               weights_initializer=tf.zeros_initializer, activation_fn=None))

    def upsample_x2_16s(self, combine_16s_out):
        ''' the Decoder-Block-3 as per LinkNet diagram '''
        return self._decode_block(combine_16s_out, 128)

# -----------------
    def combine_8s(self, post_upsample_8s, skip_conn_8s):
        ''' the middle PLUS  as per LinkNet diagram '''
        # return post_upsample_8s + skip_conn_8s

        # TEMP!!
        return tf.nn.relu(post_upsample_8s + slim.conv2d(skip_conn_8s, 128, [1, 1],
                                              weights_initializer=tf.zeros_initializer, activation_fn=None))

    def upsample_x2_8s(self, combine_8s_out):
        ''' the Decoder-Block-3 as per LinkNet diagram '''
        return self._decode_block(combine_8s_out, 64)

# -----------------
    def combine_4s(self, post_upsample_4s, skip_conn_4s):
        ''' the bottom PLUS  as per LinkNet diagram '''
        # return post_upsample_4s + skip_conn_4s

        # TEMP!!
        return tf.nn.relu(post_upsample_4s + slim.conv2d(skip_conn_4s, 64, [1, 1],
                                              weights_initializer=tf.zeros_initializer, activation_fn=None))

    def final_x4_upsample(self, combine_4s_out):
        ''' Decoder-Block-1 + Final-Block as per LinkNet diagram '''

        # # TEMP!!
        # tmp = slim.conv2d(combine_4s_out, 64, [3, 3])
        # classified = slim.conv2d(tmp,  self.number_of_classes, [1, 1],
        #                         activation_fn=None, normalizer_fn=None)
        # return self._upsample_fixed_bilinear(classified, 4)

        # pre_final = self._decode_block(combine_4s_out, 64) - can't do that, would be one *2 upsampling too many
        # paper is not clear on that - EncoderBlock1 seems to be non-downsampling while it IS NOT in original ResNet18 paper..
        #  if it is not, then DecoderBlock1 can't be upsampling
        # ....Anyways, let's do two simple convs instead..
        tmp = slim.conv2d(combine_4s_out, 64, [3, 3])
        pre_final = slim.conv2d(tmp, 64, [3, 3])

        final_interm1 = slim.conv2d_transpose(pre_final,  32,\
                                          kernel_size=[4, 4], stride=[2, 2])
        final_interm2 = slim.conv2d(final_interm1, 32, [3, 3])

        #return slim.conv2d_transpose(final_interm2,  self.number_of_classes,\
        #                             kernel_size=[4, 4], stride=[2, 2],
        #                             activation_fn=None, normalizer_fn=None)
        classified = slim.conv2d(final_interm2,  self.number_of_classes, [1, 1],
                                activation_fn=None, normalizer_fn=None)
        return self._upsample_fixed_bilinear(classified, 2)