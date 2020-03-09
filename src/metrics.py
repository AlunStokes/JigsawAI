import tensorflow as tf

def freq_acc(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    a = tf.math.subtract(y_true, y_pred)
    b = tf.math.subtract(1., a)
    c = tf.cast(tf.shape(y_true)[0], tf.float32)
    d = tf.cast(tf.math.count_nonzero(b), tf.float32)
    e = tf.math.divide(d, c)
    return e
