import tensorflow as tf
from networks import generator, discriminator
from ops import *
from utils import *
import numpy as np
from PIL import Image
import scipy.misc as misc
import os


class ContextEncoder:
    def __init__(self):
        # Paper: Context Encoders: Feature Learning by Inpainting
        self.inputs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
        self.patch = tf.placeholder(tf.float32, [None, MASK_H, MASK_W, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)
        G = generator("generator")
        D = discriminator("discriminator")
        self.patch_fake = G(self.inputs, self.train_phase)
        self.fake_logits = D(self.patch_fake, self.train_phase)
        self.real_logits = D(self.patch, self.train_phase)
        self.D_loss = -tf.reduce_mean(tf.log(self.real_logits + EPSILON) + tf.log(1 - self.fake_logits + EPSILON))
        self.G_loss = -tf.reduce_mean(tf.log(self.fake_logits + EPSILON)) + 100*tf.reduce_mean(tf.reduce_sum(tf.square(self.patch - self.patch_fake), [1, 2, 3]))
        self.D_Opt = tf.train.AdamOptimizer(2e-4).minimize(self.D_loss, var_list=D.get_var())
        self.G_Opt = tf.train.AdamOptimizer(2e-4).minimize(self.G_loss, var_list=G.get_var())
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        file_path = "./cats_bigger_than_128x128//"
        file_names = os.listdir(file_path)
        num_files = file_names.__len__()
        saver = tf.train.Saver()
        for i in range(50000):
            random_nums = np.random.randint(0, num_files, [BATCH_SIZE])
            random_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C])
            patch = np.zeros([BATCH_SIZE, MASK_H, MASK_W, IMG_C])
            batch_idx = 0
            for idx in random_nums:
                mask, X, Y = get_mask()
                img = misc.imresize(read_img_and_crop(file_path+file_names[idx]), [IMG_H, IMG_W])
                random_batch[batch_idx, :, :, :] = (img * (1 - mask) + 255 * mask) / 127.5 - 1.0
                patch[batch_idx, :, :, :] = (img[X:X + MASK_H, Y:Y + MASK_W, :]) / 127.5 - 1.0
                batch_idx += 1
            self.sess.run(self.D_Opt, feed_dict={self.inputs: random_batch, self.patch: patch, self.train_phase: True})
            self.sess.run(self.G_Opt, feed_dict={self.inputs: random_batch, self.patch: patch, self.train_phase: True})
            if i % 50 == 0:
                [D_loss, G_loss, img_patch] = self.sess.run([self.D_loss, self.G_loss, self.patch_fake], feed_dict={self.inputs: random_batch, self.patch: patch, self.train_phase: False})
                print("Iteration: %d, D_loss: %g, G_loss: %g" % (i, D_loss, G_loss))
                random_batch[-1, :, :, :][X:X+MASK_H, Y:Y+MASK_W, :] = img_patch[-1, :, :, :]
                Image.fromarray(np.uint8((random_batch[-1, :, :, :] + 1)*127.5)).save("./Results//" + str(i) + ".jpg")
            if i % 500 == 0 and i != 0:
                saver.save(self.sess, "./save_para//para.ckpt")


if __name__ == "__main__":
    CE = ContextEncoder()
    CE.train()
