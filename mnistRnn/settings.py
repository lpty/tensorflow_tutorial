from tensorflow.examples.tutorials.mnist import input_data

# minst测试集
mnist = input_data.read_data_sets('mnist/', one_hot=True)
# 每次使用100条数据进行训练
batch_size = 100
# 图像向量
width = 28
height = 28
# LSTM隐藏神经元数量
rnn_size = 256
# 输出层one-hot向量长度的
out_size = 10
