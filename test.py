import tensorflow as tf
import numpy as np

x = tf.constant(np.random.normal(size=(128,5),scale=0.00001))
y = tf.constant(np.random.normal(size=(128,5)))
x_mean = tf.reduce_mean(x, axis=0)
x_std = tf.math.reduce_std(x,axis=0)
y_mean = tf.reduce_mean(y, axis=0)
y_std = tf.math.reduce_std(y,axis=0)
corr_per_neuron = tf.divide(
        tf.reduce_sum((x-x_mean)*(y-y_mean), axis=0)
    ,
        tf.sqrt(tf.reduce_sum((x - x_mean)**2, axis=0))*
        tf.sqrt(tf.reduce_sum((y - y_mean)**2,axis=0))
    )
corr_per_neuron = tf.boolean_mask(corr_per_neuron,tf.logical_and(tf.greater(x_std,1e-5),tf.greater(y_std,1e-5)))

corr_no_nans = tf.where(tf.is_finite(corr_per_neuron), corr_per_neuron, tf.zeros_like(corr_per_neuron))
with tf.Session() as sess:
    print(corr_per_neuron.eval())
    print(tf.logical_and(tf.greater(x_std,1e-5),tf.greater(y_std,1e-5)).eval())