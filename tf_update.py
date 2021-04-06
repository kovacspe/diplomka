import tensorflow as tf
import numpy as np
w = np.arange(10).astype('float32')
weights_ph = tf.placeholder_with_default(
                w,
                shape=(10),
                name='weights_ph')
weights_var = tf.Variable(
                weights_ph,
                dtype=tf.float32,
                name='weights_var')
double = weights_var.assign(weights_var*2)

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
        gauss_kernel = tf.einsum('i,j->ij',vals,vals)
        return gauss_kernel / tf.reduce_sum(gauss_kernel)

with tf.Session() as sess:
    print(sess.run(tf.reshape(gaussian_kernel(1,0.0,1.0)[:, :, tf.newaxis, tf.newaxis],tf)))
    sess.run(
        [weights_var.initializer],
            feed_dict={weights_ph: w})
    print(sess.run(weights_var))
    sess.run(double)
    print(sess.run(weights_var))
