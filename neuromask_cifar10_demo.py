import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from setup_cifar import CIFAR, CIFARModel

from neuromask import NeuroMask


flags = tf.app.flags

flags.DEFINE_integer('test_idx', 1, help='Index for test example.')
flags.DEFINE_integer('num_iters', 2000, help='Number of iterations')
flags.DEFINE_float('smooth_lambda', 0.15,
                   help='Weight of smoothness loss coefficient')


def display_img(img, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(img)
    mask_min = mask.min()
    mask_max = mask.max()
    mask_vis = (mask - mask_min) / (mask_max - mask_min)
    ax[1].imshow(mask_vis, cmap='jet', alpha=0.6)
    ax[1].axis('off')
    return fig


FLAGS = flags.FLAGS
cifar_data = CIFAR()
test_idx = FLAGS.test_idx
test_img = cifar_data.test_data[test_idx]+0.5
tf.reset_default_graph()

with tf.Session() as sess:
    model = CIFARModel('cifar10_model', sess, False)
    input_holder = tf.placeholder(tf.float32, [1, 32, 32, 3], name='x')
    model_out = model(input_holder)

    mask_net = NeuroMask(model, coeffs=(
        0.4, 0.35, FLAGS.smooth_lambda), temp=1, is_cifar=True)
    mask_net.init_model(sess)
    pred_ = sess.run(model_out, feed_dict={input_holder: [test_img]})
    print('correct label = ', np.argmax(
        cifar_data.test_labels[test_idx], axis=0))
    mask_result = mask_net.explain(
        sess, test_img, target_label=None, iters=FLAGS.num_iters)
    final_mask = mask_result[-1][:, :, 0]
    display_img(test_img, final_mask)
    plt.show()

fig, axes = plt.subplots(6, 10)
axes = axes.ravel()
for i in range(60):
    axes[i].imshow(cifar_data.test_data[i]+0.5)
    axes[i].set_title('idx = {}'.format(i))
plt.show()
