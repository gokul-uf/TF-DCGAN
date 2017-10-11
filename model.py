import tensorflow as tf
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class DCGAN(object):
    """
        Tensorflow implementation of DCGAN, with four CNN layers.
        We assume the input images are of size 64x64. 

        TODO
            1. finish the discriminator implementation
            2. finish the compute_loss implementation
                this requires calling the _create_discriminator twice, once with the generator output and another with the input_images
            3. Figure out the gradient flow while the training process occurs, in maybe main.py ?
            4. Create a pipeline class to feed data into the model
    """

    def __init__(self):
        self.image_size = 64
        self.noise_size = 100
        self.lrelu_alpha = 0.2
        self._create_placeholders()
        self.generator_output = self._create_generator()
        self.real_predictions = self._create_discriminator(inputs = self.input_images)
        self.fake_predictions = self._create_discriminator(inputs = sel.generator_output)
        self._compute_loss()

    def _create_placeholders(self):
        self.input_images = tf.placeholder(
            shape=[None, self.image_size, self.image_size],
            type=tf.float32,
            name="input_images")
        self.input_noise = tf.placeholder(
            shape=[None, self.noise_size], type=tf.float32, name="input_noise")

    def _create_generator(self):
        xav_init = tf.contrib.layers.xavier_initializer
        bnorm = tf.layers.batch_normalization
        with tf.variable_scope("generator"):
            fc_1 = tf.layers.dense(
                inputs=self.input_noise, units=4 * 4 * 512, name="fc_1")
            reshaped_noise = tf.reshape(
                self.input_noise,
                shape=[tf.shape(fc_1)[0], 4, 4, 512],
                name="reshaped_noise")

            def _create_deconv_bnorm_block(self,
                                           inputs,
                                           name,
                                           filters,
                                           activation=tf.nn.relu):
                with tf.variable_scope(name):
                    deconv = tf.layers.conv2d_transpose(
                        inputs=inputs,
                        filters=filters,
                        kernel_size=[5, 5],
                        strides=2,
                        kernel_initializer=xav_init(),
                        name="deconv")
                    deconv = activation(deconv)
                    bnorm_op = bnorm(deconv, name="bnorm")
                    return bnorm_op

            bnorm_1 = _create_deconv_bnorm_block(
                inputs=reshaped_noise, filters=256, name="block_1")
            print("bnorm_1 shape: {}".format(bnorm_1.shape))

            bnorm_2 = _create_deconv_bnorm_block(
                inputs=bnorm_1, filters=128, name="block_2")
            print("bnorm_2 shape: {}".format(bnorm_2.shape))

            bnorm_3 = _create_deconv_bnorm_block(
                inputs=bnorm_1, filters=64, name="block_3")
            print("bnorm_3 shape: {}".format(bnorm_3.shape))

            bnorm_4 = _create_deconv_bnorm_block(
                inputs=bnorm_1,
                filters=3,
                activation=tf.nn.tanh(),
                name="block_4")
            print("bnorm_4 shape: {}".format(bnorm_4.shape))            
            return bnorm_4

    def _create_discriminator(self, inputs):  #  TODO(gokuls) need to figure out how to share weights
        xav_init = tf.contrib.layers.xavier_initializer
        bnorm = tf.layers.batch_normalization
        with tf.variable_scope("discriminator", reuse = True):

            def _create_conv_bnorm_block(self, inputs, name, filters):
                with tf.variable_scope(name, reuse = True):
                    conv = tf.layers.conv2d(
                        inputs=inputs,
                        filters=filters,
                        kernel_size=[5, 5],
                        strides=2,
                        kernel_initializer=xav_init(),
                        name="conv")

                    conv = tf.maximum(conv, self.lrelu_alpha * conv)
                    bnorm_op = bnorm(conv, name="bnorm")
                    return bnorm_op
            conv_1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size=[5,5], strides=2,
                    kernel_initializer=xav_init(), name = "conv_1")
            conv_1 = tf.maximum(conv, self.lrelu_alpha * conv_1)
            print("conv_1 shape: {}".format(conv_1.shape))

            bnorm_1 = _create_conv_bnorm_block(inputs = conv_1, filters = 128, name = "block_1")
            print("bnorm_1 shape: {}".format(bnorm_1.shape))

            bnorm_2 = _create_conv_bnorm_block(inputs = bnorm_1, filters = 256, name = "block_2")
            print("bnorm_2 shape: {}".format(bnorm_2.shape))

            bnorm_3 = _create_conv_bnorm_block(inputs = bnorm_2, filters = 512, name = "block_3")
            print("bnorm_3 shape: {}".format(bnorm_3.shape))

           reshaped_bnorm_3 = tf.reshape(bnorm_3, shape = [tf.shape(bnorm_3)[0], 4*4*512], name = "reshaped_bnorm_3")

           fc_1 = tf.layers.dense(inptus = reshaped_bnorm_3, units = 1, name = "fc_1")
           return fc_1

    def _compute_loss(self):
        pass


        

