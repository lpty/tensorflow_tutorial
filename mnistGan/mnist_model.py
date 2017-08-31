import os
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime


class MnistModel:

    def __init__(self):
        # mnist测试集
        self.mnist = input_data.read_data_sets('mnist/', one_hot=True)
        # 图片大小
        self.img_size = self.mnist.train.images[0].shape[0]
        # 每步训练使用图片数量
        self.batch_size = 64
        # 图片分块数量
        self.chunk_size = self.mnist.train.num_examples // self.batch_size
        # 训练循环次数
        self.epoch_size = 300
        # 抽取样本数
        self.sample_size = 25
        # 生成器判别器隐含层数量
        self.units_size = 128
        # 学习率
        self.learning_rate = 0.001
        # 平滑参数
        self.smooth = 0.1

    @staticmethod
    def generator_graph(fake_imgs, units_size, out_size, alpha=0.01):
        # 生成器与判别器属于两个网络 定义不同scope
        with tf.variable_scope('generator'):
            # 构建一个全连接层
            layer = tf.layers.dense(fake_imgs, units_size)
            # leaky ReLU 激活函数
            relu = tf.maximum(alpha * layer, layer)
            # dropout 防止过拟合
            drop = tf.layers.dropout(relu, rate=0.2)
            # logits
            # out_size应为真实图片size大小
            logits = tf.layers.dense(drop, out_size)
            # 激活函数 将向量值限定在某个区间 与 真实图片向量类似
            # 这里tanh的效果比sigmoid好一些
            # 输出范围(-1, 1) 采用sigmoid则为[0, 1]
            outputs = tf.tanh(logits)
            return logits, outputs

    @staticmethod
    def discriminator_graph(imgs, units_size, alpha=0.01, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # 构建全连接层
            layer = tf.layers.dense(imgs, units_size)
            # leaky ReLU 激活函数
            relu = tf.maximum(alpha * layer, layer)
            # logits
            # 判断图片真假 out_size直接限定为1
            logits = tf.layers.dense(relu, 1)
            # 激活函数
            outputs = tf.sigmoid(logits)
            return logits, outputs

    @staticmethod
    def loss_graph(real_logits, fake_logits, smooth):
        # 生成器图片loss
        # 生成器希望判别器判断出来的标签为1
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits) * (1 - smooth)))
        # 判别器识别生成器图片loss
        # 判别器希望识别出来的标签为0
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        # 判别器识别真实图片loss
        # 判别器希望识别出来的标签为1
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits) * (1 - smooth)))
        # 判别器总loss
        dis_loss = tf.add(fake_loss, real_loss)
        return gen_loss, fake_loss, real_loss, dis_loss

    @staticmethod
    def optimizer_graph(gen_loss, dis_loss, learning_rate):
        # 所有定义变量
        train_vars = tf.trainable_variables()
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        # optimizer
        # 生成器与判别器作为两个网络需要分别优化
        gen_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(dis_loss, var_list=dis_vars)
        return gen_optimizer, dis_optimizer

    def train(self):
        # 真实图片与混淆图片
        # 不确定输入图片数量 用None
        real_imgs = tf.placeholder(tf.float32, [None, self.img_size], name='real_imgs')
        fake_imgs = tf.placeholder(tf.float32, [None, self.img_size], name='fake_imgs')

        # 生成器
        gen_logits, gen_outputs = self.generator_graph(fake_imgs, self.units_size, self.img_size)
        # 判别器对真实图片
        real_logits, real_outputs = self.discriminator_graph(real_imgs, self.units_size)
        # 判别器对生成器图片
        # 公用参数所以要reuse
        fake_logits, fake_outputs = self.discriminator_graph(gen_outputs, self.units_size, reuse=True)

        # 损失
        gen_loss, fake_loss, real_loss, dis_loss = self.loss_graph(real_logits, fake_logits, self.smooth)
        # 优化
        gen_optimizer, dis_optimizer = self.optimizer_graph(gen_loss, dis_loss, self.learning_rate)

        # 开始训练
        saver = tf.train.Saver()
        step = 0
        # 指定占用GPU比例
        # tensorflow默认占用全部GPU显存 防止在机器显存被其他程序占用过多时可能在启动时报错
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch_size):
                for _ in range(self.chunk_size):
                    batch_imgs, _ = self.mnist.train.next_batch(self.batch_size)
                    batch_imgs = batch_imgs * 2 - 1
                    # generator的输入噪声
                    noise_imgs = np.random.uniform(-1, 1, size=(self.batch_size, self.img_size))
                    # 优化
                    _ = sess.run(gen_optimizer, feed_dict={fake_imgs: noise_imgs})
                    _ = sess.run(dis_optimizer, feed_dict={real_imgs: batch_imgs, fake_imgs: noise_imgs})
                    step += 1
                # 每一轮结束计算loss
                # 判别器损失
                loss_dis = sess.run(dis_loss, feed_dict={real_imgs: batch_imgs, fake_imgs: noise_imgs})
                # 判别器对真实图片
                loss_real = sess.run(real_loss, feed_dict={real_imgs: batch_imgs, fake_imgs: noise_imgs})
                # 判别器对生成器图片
                loss_fake = sess.run(fake_loss, feed_dict={real_imgs: batch_imgs, fake_imgs: noise_imgs})
                # 生成器损失
                loss_gen = sess.run(gen_loss, feed_dict={fake_imgs: noise_imgs})

                print(datetime.now().strftime('%c'), ' epoch:', epoch, ' step:', step, ' loss_dis:', loss_dis,
                      ' loss_real:', loss_real, ' loss_fake:', loss_fake, ' loss_gen:', loss_gen)
            model_path = os.getcwd() + os.sep + "mnist.model"
            saver.save(sess, model_path, global_step=step)

    def gen(self):
        # 生成图片
        sample_imgs = tf.placeholder(tf.float32, [None, self.img_size], name='sample_imgs')
        gen_logits, gen_outputs = self.generator_graph(sample_imgs, self.units_size, self.img_size)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            sample_noise = np.random.uniform(-1, 1, size=(self.sample_size, self.img_size))
            samples = sess.run(gen_outputs, feed_dict={sample_imgs: sample_noise})
        with open('samples.pkl', 'wb') as f:
            pickle.dump(samples, f)

    @staticmethod
    def show():
        # 展示图片
        with open('samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        plt.show()
