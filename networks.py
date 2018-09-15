from ops import *
from config import *

class generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = leaky_relu((conv("conv1", inputs, 64, 4, 2)))
            inputs = leaky_relu(batchnorm(conv("conv2", inputs, 64, 4, 2), train_phase, "conv2"))
            inputs = leaky_relu(batchnorm(conv("conv3", inputs, 128, 4, 2), train_phase, "conv3"))
            inputs = leaky_relu(batchnorm(conv("conv4", inputs, 256, 4, 2), train_phase, "conv4"))
            inputs = leaky_relu(batchnorm(conv("conv5", inputs, 512, 4, 2), train_phase, "conv5"))
            size = inputs.shape
            inputs = fully_connected("conv2vec", inputs, 1000)
            inputs = fully_connected("vec2conv", inputs, size[1]*size[2]*size[3])
            inputs = tf.reshape(inputs, [-1, size[1], size[2], size[3]])
            inputs = tf.nn.relu(batchnorm(uconv("uconv1", inputs, 512, 4, 2), train_phase, "uconv1"))
            inputs = tf.nn.relu(batchnorm(uconv("uconv2", inputs, 256, 4, 2), train_phase, "uconv2"))
            inputs = tf.nn.relu(batchnorm(uconv("uconv3", inputs, 128, 4, 2), train_phase, "uconv3"))
            inputs = tf.nn.relu(batchnorm(uconv("uconv4", inputs, 64, 4, 2), train_phase, "uconv4"))
            inputs = tf.nn.tanh(conv("conv6", inputs, IMG_C, 4, 1))
            return inputs

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = leaky_relu((conv("conv1", inputs, 64, 4, 2, is_SN=True)))
            inputs = leaky_relu(batchnorm(conv("conv2", inputs, 128, 4, 2, is_SN=True), train_phase, "conv2"))
            inputs = leaky_relu(batchnorm(conv("conv3", inputs, 256, 4, 2, is_SN=True), train_phase, "conv3"))
            inputs = leaky_relu(batchnorm(conv("conv4", inputs, 512, 4, 2, is_SN=True), train_phase, "conv4"))
            inputs = fully_connected("fc", inputs, 1)
            return tf.sigmoid(inputs)

    def get_var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)