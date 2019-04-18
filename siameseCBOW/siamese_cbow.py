import tensorflow as tf


class SiameseCBOW:

    def __init__(self, sentence_size, vocab_size, embed_size, margin, learning_rate, is_train):
        self.margin = margin
        self.anchor = tf.placeholder(name='anchor', shape=(None, sentence_size), dtype=tf.int32)
        self.anchor_length = tf.placeholder(name='anchor_length', shape=(None, sentence_size, embed_size), dtype=tf.float32)

        self.left = tf.placeholder(name='left', shape=(None, sentence_size), dtype=tf.int32)
        self.left_length = tf.placeholder(name='left_length', shape=(None, sentence_size, embed_size), dtype=tf.float32)

        self.right = tf.placeholder(name='right', shape=(None, sentence_size), dtype=tf.int32)
        self.right_length = tf.placeholder(name='right_length', shape=(None, sentence_size, embed_size), dtype=tf.float32)

        self.label = tf.placeholder(name='label', shape=(None, 2), dtype=tf.int32)
        self.embedding = tf.get_variable(name='embedding', shape=(vocab_size, embed_size), dtype=tf.float32)

        self.prediction, self.loss = self.inference()
        self.accuracy = self.compute_accuracy(self.prediction)

        if is_train:
            self.optimize = self.optimizer(self.loss, learning_rate)

    def inference(self):
        # embed layer
        anchor_embed = tf.nn.embedding_lookup(self.embedding, self.anchor) * self.anchor_length
        left_embed = tf.nn.embedding_lookup(self.embedding, self.left) * self.left_length
        right_embed = tf.nn.embedding_lookup(self.embedding, self.right) * self.right_length

        # average layer
        anchor_length = tf.expand_dims(tf.reduce_sum(tf.reduce_mean(self.anchor_length, axis=2), axis=1), axis=1)
        anchor_vector = tf.reduce_sum(anchor_embed, axis=1)/anchor_length

        left_length = tf.expand_dims(tf.reduce_sum(tf.reduce_mean(self.left_length, axis=2), axis=1), axis=1)
        left_vector = tf.reduce_sum(left_embed, axis=1)/left_length

        right_length = tf.expand_dims(tf.reduce_sum(tf.reduce_mean(self.right_length, axis=2), axis=1), axis=1)
        right_vector = tf.reduce_sum(right_embed, axis=1)/right_length

        # cosine layer
        anchor_left = compute_cosine_similarity(anchor_vector, left_vector)
        anchor_right = compute_cosine_similarity(anchor_vector, right_vector)
        cosine = tf.stack([anchor_left, anchor_right], axis=1)

        # soft max layer
        prediction = tf.argmax(tf.nn.softmax(cosine), axis=1, name="predictions")

        # loss
        # cross entropy
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=cosine)
        # loss = tf.reduce_mean(loss)

        # contrastive_loss
        label_left, label_right = tf.unstack(self.label, axis=1)
        loss_left = compute_contrastive_loss(anchor_vector, left_vector, label_left, self.margin)
        loss_right = compute_contrastive_loss(anchor_vector, right_vector, label_right, self.margin)
        loss = loss_left + loss_right

        return prediction, loss

    def optimizer(self, loss, learning_rate):
        optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gvs = optimizer.compute_gradients(loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
        #               for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs)
        return optimize

    def compute_accuracy(self, prediction):
        correct_prediction = tf.equal(prediction, tf.argmax(self.label, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        return accuracy


def compute_cosine_similarity(x1, x2):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))

    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))

    x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)

    return x1_x2 / x1_norm / x2_norm


def compute_euclidean_distance(x1, x2):
    d = tf.reduce_sum(tf.square(x1 - x2), 1)
    return d


def compute_contrastive_loss(x1, x2, label, margin):
    x1 = tf.nn.l2_normalize(x1, dim=1)
    x2 = tf.nn.l2_normalize(x2, dim=1)
    label = tf.to_float(label)
    one = tf.constant(1.0)

    d = compute_euclidean_distance(x1, x2)
    d_sqrt = tf.sqrt(compute_euclidean_distance(x1, x2))
    first_part = tf.multiply(label, d)  # (Y)*(d)

    max_part = tf.square(tf.maximum(margin - d_sqrt, 0))
    second_part = tf.multiply(one-label, max_part)  # (1-Y) * max(margin - d, 0)

    loss = 0.5 * tf.reduce_mean(first_part + second_part)

    return loss


def _test():
    import numpy as np

    sentence_size = 5
    vocab_size = 100
    embed_size = 10
    margin = 1.25
    learning_rate = 0.001
    is_train = True
    batch_size = 8
    model = SiameseCBOW(sentence_size, vocab_size, embed_size,
                        margin, learning_rate, is_train)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.random.randint(low=0, high=99, size=(batch_size, sentence_size), dtype=np.int32)
            input_x_length = np.ones((batch_size, sentence_size, embed_size), dtype=np.int32)

            input_left = np.random.randint(low=0, high=99, size=(batch_size, sentence_size), dtype=np.int32)
            input_left_length = np.ones((batch_size, sentence_size, embed_size), dtype=np.int32)

            input_right = np.random.randint(low=0, high=99, size=(batch_size, sentence_size), dtype=np.int32)
            input_right_length = np.ones((batch_size, sentence_size, embed_size), dtype=np.int32)

            input_y = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0]], dtype=np.int32)
            loss, acc, _ = sess.run(
                [model.loss, model.accuracy, model.optimize],
                feed_dict={model.anchor: input_x,
                           model.left: input_left,
                           model.right: input_right,
                           model.label: input_y,
                           model.anchor_length: input_x_length,
                           model.left_length: input_left_length,
                           model.right_length: input_right_length,
                           })
            print("loss:", loss, "acc:", acc)


if __name__ == '__main__':
    _test()
