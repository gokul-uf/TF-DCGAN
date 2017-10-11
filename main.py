from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os
import tensorflow as tf

from model import DCGAN
from data_utils import Processor
"""
TODO
    1. Finish the output_dir creation
    2. Create image summary in model.py
    3. Finish the tensorboard FileWriter part
    4. Finish the saved model saver
"""

flags = tf.app.flags
flags.DEFINE_string("data_dir", None, "The location of the dataset")
flags.DEFINE_string("output_dir", None, "Where should the outputs be stored")
flags.DEFINE_integer("save_every", None, "Save checkpoints every N epochs")
flags.DEFINE_integer("eval_every", None, "Generate images every N epochs")
flags.DEFINE_integer("eval_images", 100, "How many images to generate at eval")
flags.DEFINE_integer("num_steps", 1000, "Number of batchs to train on")
flags.DEFINE_integer("batch_size", 100, "Batch size")

if __name__ == "__main__":
    tf.logging.info("Starting training")
    dcgan = DCGAN()
    processor = Processor(data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.num_steps):
            train_batch = processor.get_batch().next()

            # because we can get smaller batches at file edges
            noise = np.random.randn(len(train_batch), dcgan.noise_size)

            fetches = [
                dcgan.d_train, dcgan.g_train, dcgan.d_loss, dcgan.g_loss
            ]
            feed_dict = {
                dcgan.input_images: train_batch,
                dcgan.input_noise: noise
            }
            _, _, d_loss, g_loss = sess.run(fetches, feed_dict=feed_dict)

            if i % FLAGS.eval_every == 0:
                # Let's generate some images!
                feed_dict = {
                    dcgan.input_noise:
                    np.random.randn(FLAGS.eval_images, dcgan.noise_size)
                }
                gen_output = sess.run(
                    dcgan.generator_output, feed_dict=feed_dict)
                gen_output = (gen_output * 127) + 127.0
                gen_output = gen_output.astype(np.uint8)

            if i % FLAGS.save_every == 0:
                # Save the trained model
                pass
