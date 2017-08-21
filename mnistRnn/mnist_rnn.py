import tensorflow as tf
import numpy as np
from mnistRnn.rnn_train import rnn_graph
from mnistRnn.settings import mnist, width, height, rnn_size, out_size


def mnist2text(image_list, height, width, rnn_size, out_size):
    '''
    mnist数字向量转为文本
    :param image_list:
    :param height:
    :param width:
    :param rnn_size:
    :param out_size:
    :return:
    '''
    x = tf.placeholder(tf.float32, [None, height, width])
    y_conv = rnn_graph(x, rnn_size, out_size, width, height)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(y_conv, 1)
        vector_list = sess.run(predict, feed_dict={x: image_list})
        vector_list = vector_list.tolist()
        return vector_list


if __name__ == '__main__':
    batch_x_test = mnist.test.images
    batch_x_test = batch_x_test.reshape([-1, height, width])

    batch_y_test = mnist.test.labels
    batch_y_test = list(np.argmax(batch_y_test, 1))

    pre_y = list(mnist2text(batch_x_test, height, width, rnn_size, out_size))
    for text in batch_y_test:
        print('Label:', text, ' Predict:', pre_y[batch_y_test.index(text)])

