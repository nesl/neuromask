"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

"""
import tensorflow as tf
import numpy as np

class NeuroMask(object):
    def smoothing_loss(self, m):
        if self.is_cifar:
            grad_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((3,3,1,1)).astype(np.float32)
        else:
            grad_filter = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]]).reshape((5,5,1,1)).astype(np.float32)
        grad_out = tf.nn.convolution(tf.expand_dims(tf.expand_dims(m, 2),0), grad_filter, 'VALID')
        grad_out_norm = tf.reduce_mean(tf.pow(grad_out, 2))
        return grad_out_norm
    
    def __init__(self, model, weight_limit=10, temp=0.3, learning_rate=0.03, coeffs=(50, 16, 3), is_cifar=False):
        """
        Args:
            model: The given model.
        """
        super().__init__()
        self.model = model
        self.weight_limit = weight_limit
        self.learning_rate = learning_rate
        self.temp = temp
        self.is_cifar=is_cifar
        self.lambda_pred, self.lambda_sparse, self.lambda_spatial = coeffs
        self.mask_w = tf.get_variable('maskw', shape=[self.model.image_h,self.model.image_w,1],
                                      initializer=tf.random_uniform_initializer(
                                          minval=-self.weight_limit, maxval=self.weight_limit, dtype=tf.float32))
        
        self.img_input = tf.placeholder(tf.float32, shape=self.model.input_shape, name='input')
        self.mask_var = tf.sigmoid(self.mask_w/self.temp)
        self.masked_img_input = tf.multiply(self.img_input, self.mask_var)
        
        # Feed both inputs to the provided model
        if self.is_cifar:
            logits_img = self.model(self.img_input)
            self.pred_img = tf.argmax(logits_img, axis=1)
            logits_masked = self.model(self.masked_img_input)
            self.pred_masked = tf.argmax(logits_masked, axis=1)
            
        else:
            logits_img, self.pred_img = self.model(self.img_input)
            logits_masked, self.pred_masked = self.model(self.masked_img_input)
        
        target_class_label = tf.one_hot(self.pred_img, depth=self.model.num_classes)
        pred_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=target_class_label, logits=logits_masked))
        sparse_loss = tf.reduce_mean(tf.abs(self.mask_w+self.weight_limit))
        spatial_loss = self.smoothing_loss(self.mask_w[:,:,0])
        
        self.total_loss = self.lambda_pred * pred_loss + self.lambda_sparse * sparse_loss + self.lambda_spatial * spatial_loss
        self.optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optim.minimize(self.total_loss, var_list=[self.mask_w])
        
    def init_model(self, sess):
        sess.run(tf.global_variables_initializer())
        self.model.restore(sess)
        
    def explain(self, sess, input_img, target_label = None, iters=1000):
        mask_hist = []
        feed_dict={self.img_input: [input_img]}
        cur_mask, cur_pred = sess.run([self.mask_var, self.pred_img], feed_dict={
            self.img_input: [input_img]
        })
        print('Original prediction = ', cur_pred)
        mask_hist.append(cur_mask)
        if target_label:
            feed_dict[self.pred_img] = [target_label]
        for ii in range(iters):
            cur_mask, cur_pred, cur_loss,_ = sess.run([self.mask_var, self.pred_masked  , self.total_loss, self.train_op],
                                                    feed_dict=feed_dict)
            mask_hist.append(cur_mask)
            if ii % 50 ==0:
                print('{} - {} - {}'.format(ii, cur_pred, cur_loss))
        return mask_hist