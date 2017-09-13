import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skipGramVec.text import Text


class Model:

    def __init__(self):
        self.train_text = Text()
        self.batch = self.train_text.batch()
        self.batch_size = self.train_text.batch_size
        self.chunk_size = self.train_text.chunk_size
        self.vocab_size = len(self.train_text.vocab)
        # 权重矩阵维度 即最终每个词对应向量维度
        self.embedding_size = 200
        # 负采样数量
        self.sample_size = 100
        # 循环次数
        self.epoch_size = 10
        # 可视化单词数量
        self.viz_words = 100

    def embedding(self, inputs=None):
        # 将int_word转化为embedding_size维度的向量
        # 这也是模型训练完后我们最终想要的矩阵
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            embed = tf.nn.embedding_lookup(embedding, inputs) if inputs is not None else None
        return embedding, embed

    def softmax(self):
        w = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1))
        b = tf.Variable(tf.zeros(self.vocab_size))
        return w, b

    def loss(self, w, b, labels, embed):
        # 采用负样本采样 加快收敛速度
        return tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=w, biases=b, labels=labels, inputs=embed,
                                                         num_sampled=self.sample_size, num_classes=self.vocab_size))

    def optimizer(self, loss):
        return tf.train.AdamOptimizer().minimize(loss)

    def train(self):
        # 输入和标签
        # [batch_size, num_dim]
        inputs = tf.placeholder(tf.int32, [None], name='inputs')
        # [batch_size, num_true]
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

        # embedding
        embedding, embed = self.embedding(inputs)
        # softmax
        w, b = self.softmax()

        # loss
        loss = self.loss(w, b, labels, embed)
        # optimizer
        optimizer = self.optimizer(loss)

        # train
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(self.epoch_size):
            batch = self.train_text.batch()
            for batch_x, batch_y in batch:
                feed = {inputs: batch_x, labels: np.array(batch_y)[:, None]}
                train_loss, _ = sess.run([loss, optimizer], feed_dict=feed)
                print(datetime.datetime.now().strftime('%c'), ' epoch:', epoch, 'step:', step, ' train_loss:', train_loss)
                step += 1
        model_path = os.getcwd() + os.sep + "skipGramVec.model"
        saver.save(sess, model_path, global_step=step)
        sess.close()

    def gen(self):
        embedding, _ = self.embedding()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            embedding = sess.run(embedding)
        # 词向量
        data = embedding[:self.viz_words, :]
        # 高维词向量降维
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        embed_tsne = tsne.fit_transform(data)
        # 显示
        plt.subplots(figsize=(10, 10))
        for idx in range(self.viz_words):
            plt.scatter(*embed_tsne[idx, :], color='steelblue')
            plt.annotate(self.train_text.int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
        plt.show()
