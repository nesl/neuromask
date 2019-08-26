"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""

import tensorflow as tf

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

class InceptionModel(object):
    def __init__(self, checkpoint_path='inception_v3.ckpt'):
        self.checkpoint_path = checkpoint_path
        self.num_classes = 1001
        self.image_h = 299
        self.image_w = 299
        self.input_shape = [1,self.image_h,self.image_w, 3]
        self.initialized = False
        self.init_model()
    
    def init_model(self):
        if self.initialized:
            """ Model has already been built"""
            return
        input_node = tf.placeholder(tf.float32, shape=self.input_shape)
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                input_node, num_classes=self.num_classes, is_training=False, dropout_keep_prob=1.0)
        self.saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        self.initialized = True
    
    def restore(self, sess):
        self.saver.restore(sess, self.checkpoint_path)
        
    def __call__(self, input_node):
        """
        input_node: should be a place holder passing the input image whose shape =[1,299,299,3] and values are
            in the [0,1] range.
        """
        normalized_input = input_node * 2.0 - 1.0
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                normalized_input, num_classes=self.num_classes, is_training=False, dropout_keep_prob=1.0, reuse=True)
        predictions_node = end_points['Predictions']
        output_node = tf.argmax(predictions_node, axis=1)
        return logits, output_node
