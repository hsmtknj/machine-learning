import numpy as np
import tensorflow as tf

a = tf.Variable(1)
b = tf.Variable(2)
c = a + b

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(a.eval())
    print(b.eval())
    print(c.eval())
    
