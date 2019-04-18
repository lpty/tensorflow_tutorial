import random
random.seed(20190417)
import numpy as np
import tensorflow as tf
from siameseCBOW.siamese_cbow import SiameseCBOW
from c2v import C2V

c2v = C2V.load('c2v.model')
word_list = c2v.wv.index2word
word_list.append('PAD')
index2word = {index: word for index, word in enumerate(c2v.wv.index2word)}
word2index = {word: index for index, word in enumerate(c2v.wv.index2word)}

matrix = c2v.wv.syn0
matrix = np.row_stack((matrix, np.zeros(shape=(100,))))
# new_c2v = dict()
# with open('c2v.model_0_3.txt', 'r', encoding='utf-8') as f:
#     dim = int(next(f).rstrip().split()[1])
#     for line in f:
#         line = line.rstrip().split(' ')
#         token = line[0]
#         if not token: continue
#         new_c2v[token] = np.asarray(list(map(float, line[1:])))
# matrix = np.array([new_c2v.get(word, np.zeros(shape=(100,))) for _, word in enumerate(c2v.wv.index2word)])

train_set = open('train.txt', encoding='utf-8').readlines()
test_set = open('test.txt', encoding='utf-8').readlines()

sentence_size = 20
vocab_size = len(index2word)
embed_size = 100
margin = 1
learning_rate = 0.001
is_train = True
batch_size = 32
epoch_num = 20


def processor(sentence_list):

    def pad(matrix):
        return list(map(lambda l: l + [word2index['PAD']] * (sentence_size - len(l)), matrix))

    def get_length(x):
        zero = np.zeros((sentence_size, embed_size))
        zero[:len(x)] = 1
        return zero

    def _get(get_sentence_list):
        anchor, left, right, label = [], [], [], []
        for line in get_sentence_list:
            sentences = line.replace('\n', '').split(',')
            anchor.append([word2index[c] for c in sentences[0] if c in word2index][:20])
            left.append([word2index[c] for c in sentences[1] if c in word2index][:20])
            right.append([word2index[c] for c in sentences[2] if c in word2index][:20])
            label.append([int(sentences[3]), int(sentences[4])])
        anchor_length = [get_length(sentence) for sentence in anchor]
        left_length = [get_length(sentence) for sentence in left]
        right_length = [get_length(sentence) for sentence in right]
        return pad(anchor), pad(left), pad(right), label, anchor_length, left_length, right_length

    random.shuffle(sentence_list)

    for start, end in zip(range(0, len(sentence_list), batch_size),
                          range(batch_size, len(sentence_list), batch_size)):
        get_sentence_list = sentence_list[start: end]
        yield _get(get_sentence_list)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    model = SiameseCBOW(sentence_size, vocab_size, embed_size,
                        margin, learning_rate, is_train)

    saver = tf.train.Saver()
    # if os.path.exists(model_path):
    #     saver.restore(sess, tf.train.latest_checkpoint(model_path))
    # else:
    #     sess.run(tf.global_variables_initializer())
    sess.run(tf.global_variables_initializer())

    sess.run(tf.assign(model.embedding, matrix))

    for epoch in range(epoch_num):
        train_process = processor(train_set)
        test_process = processor(test_set)
        loss_sum, acc_sum, count = 0, 0, 0
        for process in train_process:
            train_anchor = process[0]
            train_left = process[1]
            train_right = process[2]
            train_label = process[3]

            train_anchor_length = process[4]
            train_left_length = process[5]
            train_right_length = process[6]

            loss, acc, _ = sess.run(
                [model.loss, model.accuracy, model.optimize],
                feed_dict={model.anchor: train_anchor,
                           model.left: train_left,
                           model.right: train_right,
                           model.label: train_label,
                           model.anchor_length: train_anchor_length,
                           model.left_length: train_left_length,
                           model.right_length: train_right_length,
                           })
            loss_sum += loss
            acc_sum += acc
            count += 1
        print("count:", count, "loss:", loss_sum/count, "acc:", acc_sum/count)

        eval_loss_sum, eval_acc_sum, eval_count = 0, 0, 0
        for process in test_process:
            test_anchor = process[0]
            test_left = process[1]
            test_right = process[2]
            test_label = process[3]

            test_anchor_length = process[4]
            test_left_length = process[5]
            test_right_length = process[6]

            loss, acc = sess.run(
                [model.loss, model.accuracy],
                feed_dict={model.anchor: test_anchor,
                           model.left: test_left,
                           model.right: test_right,
                           model.label: test_label,
                           model.anchor_length: test_anchor_length,
                           model.left_length: test_left_length,
                           model.right_length: test_right_length,
                           })
            eval_loss_sum += loss
            eval_acc_sum += acc
            eval_count += 1
        print("eval_loss:", eval_loss_sum/eval_count, "eval_acc:", eval_acc_sum/eval_count)

        embeddings = sess.run(model.embedding)

        dim = embeddings.shape[-1]
        f = open(f'c2v.model_{epoch}_{eval_acc_sum}.txt', 'w', encoding='utf-8')
        f.write(" ".join([str(len(c2v.wv.index2word) - 1), str(dim)]))
        f.write("\n")

        # get word and weight matrix row index i
        for i, word in enumerate(c2v.wv.index2word):
            f.write(word)
            f.write(" ")
            # get vector a index i
            f.write(" ".join(map(str, list(embeddings[i, :]))))
            f.write("\n")
        f.close()