import os
import tensorflow as tf
from datetime import datetime
from mnistRnn.settings import mnist, batch_size, width, height, rnn_size, out_size


def weight_variable(shape, w_alpha=0.01):
    '''
    增加噪音，随机生成权重
    :param shape:
    :param w_alpha:
    :return:
    '''
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, b_alpha=0.1):
    '''
    增加噪音，随机生成偏置项
    :param shape:
    :param b_alpha:
    :return:
    '''
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def rnn_graph(x, rnn_size, out_size, width, height):
    '''
    循环神经网络计算图
    :param x:
    :param rnn_size:
    :param out_size:
    :param width:
    :param height:
    :return:
    '''
    # 权重及偏置
    w = weight_variable([rnn_size, out_size])
    b = bias_variable([out_size])
    # LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    # 原排列[0,1,2]transpose为[1,0,2]代表前两维装置，如shape=(1,2,3)转为shape=(2,1,3)
    # 这里的实际意义是把所有图像向量的相同行号向量转到一起，如x1的第一行与x2的第一行
    x = tf.transpose(x, [1, 0, 2])
    # reshape -1 代表自适应，这里按照图像每一列的长度为reshape后的列长度
    x = tf.reshape(x, [-1, width])
    # split默任在第一维即0 dimension进行分割，分割成height份，这里实际指把所有图片向量按对应行号进行重组
    x = tf.split(x, height)
    # 这里RNN会有与输入层相同数量的输出层，我们只需要最后一个输出
    outputs, status = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    y_conv = tf.add(tf.matmul(outputs[-1], w), b)

    return y_conv


def optimize_graph(y, y_conv):
    '''
    优化计算图
    :param y:
    :param y_conv:
    :return:
    '''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    return optimizer


def accuracy_graph(y, y_conv):
    '''
    偏差计算图
    :param y:
    :param y_conv:
    :return:
    '''
    correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def train():
    '''
    rnn训练
    :return:
    '''
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height, width])
    y = tf.placeholder(tf.float32)
    # rnn模型
    y_conv = rnn_graph(x, rnn_size, out_size, width, height)
    # 最优化
    optimizer = optimize_graph(y, y_conv)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)

    # 启动会话.开始训练
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    step = 0
    acc_rate = 0.98
    while 1:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, height, width])
        session.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # 每训练10次测试一次
        if step % 10 == 0:
            batch_x_test = mnist.test.images
            batch_y_test = mnist.test.labels
            batch_x_test = batch_x_test.reshape([-1, height, width])
            acc = session.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc >= acc_rate:
                model_path = os.getcwd() + os.sep + str(acc_rate) + "mnist.model"
                saver.save(session, model_path, global_step=step)
                break
        step += 1
    session.close()


if __name__ == '__main__':
    train()
