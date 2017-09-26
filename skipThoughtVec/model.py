import tensorflow as tf
from skipThoughtVec.story import Story


class Model:

    def __init__(self):
        self.story = Story()
        self.vocab = self.story.vocab
        self.batch_size = self.story.batch_size - 2
        self.chunk_size = self.story.chunk_size
        self.embedding_dim = 300
        self.num_units = 500
        self.learning_rate = 0.001
        self.epoch = 25
        self.sample_size = 50

    # soft_max参数
    @staticmethod
    def soft_max_variable(num_units, vocab_size, reuse=False):
        # 共享变量
        with tf.variable_scope('soft_max', reuse=reuse):
            w = tf.get_variable("w", [num_units, vocab_size])
            b = tf.get_variable("b", [vocab_size])
        return w, b

    # 构建输入
    @staticmethod
    def build_inputs():
        with tf.variable_scope('inputs'):
            # 句子
            encode = tf.placeholder(tf.int32, shape=[None, None], name='encode')
            encode_length = tf.placeholder(tf.int32, shape=[None, ], name='encode_length')
            # 句子的前一句
            decode_pre_x = tf.placeholder(tf.int32, shape=[None, None], name='decode_pre_x')
            decode_pre_y = tf.placeholder(tf.int32, shape=[None, None], name='decode_pre_y')
            decode_pre_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_pre_length')
            # 句子的后一句
            decode_post_x = tf.placeholder(tf.int32, shape=[None, None], name='decode_post_x')
            decode_post_y = tf.placeholder(tf.int32, shape=[None, None], name='decode_post_y')
            decode_post_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_post_length')
        return encode, decode_pre_x, decode_pre_y, decode_post_x, decode_post_y, encode_length, decode_pre_length, decode_post_length

    # 构建输入embedding
    def build_word_embedding(self, encode, decode_pre_x, decode_post_x):
        # embedding
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable(name='embedding', shape=[len(self.vocab), self.embedding_dim],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))
            encode_emb = tf.nn.embedding_lookup(embedding, encode, name='encode_emb')
            decode_pre_emb = tf.nn.embedding_lookup(embedding, decode_pre_x, name='decode_pre_emb')
            decode_post_emb = tf.nn.embedding_lookup(embedding, decode_post_x, name='decode_post_emb')
        return encode_emb, decode_pre_emb, decode_post_emb

    # 构建encoder
    def build_encoder(self, encode_emb, length, train=True):
        batch_size = self.batch_size if train else 1
        with tf.variable_scope('encoder'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
            initial_state = cell.zero_state(batch_size, tf.float32)
            _, final_state = tf.nn.dynamic_rnn(cell, encode_emb, initial_state=initial_state, sequence_length=length)
        return initial_state, final_state

    # 构建decoder
    def build_decoder(self, decode_emb, length, state, scope='decoder', reuse=False):
        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
            outputs, final_state = tf.nn.dynamic_rnn(cell, decode_emb, initial_state=state, sequence_length=length)
        x = tf.reshape(outputs, [-1, self.num_units])
        w, b = self.soft_max_variable(self.num_units, len(self.vocab), reuse=reuse)
        logits = tf.matmul(x, w) + b
        prediction = tf.nn.softmax(logits, name='predictions')
        return logits, prediction, final_state

    # 构建loss
    def build_loss(self, logits, targets, scope='loss'):
        with tf.variable_scope(scope):
            y_one_hot = tf.one_hot(targets, len(self.vocab))
            y_reshaped = tf.reshape(y_one_hot, [-1, len(self.vocab)])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        return loss

    # 构建optimizer
    def build_optimizer(self, loss, scope='optimizer'):
        with tf.variable_scope(scope):
            grad_clip = 5
            # 使用clipping gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    def build(self):
        # 输入
        encode, decode_pre_x, decode_pre_y, decode_post_x, decode_post_y, encode_length, decode_pre_length, decode_post_length = self.build_inputs()
        # embedding
        encode_emb, decode_pre_emb, decode_post_emb = self.build_word_embedding(encode, decode_pre_x, decode_post_x)

        # encoder构建
        initial_state, final_state = self.build_encoder(encode_emb, encode_length)

        # 前一句decoder
        pre_logits, pre_prediction, pre_state = self.build_decoder(decode_pre_emb, decode_pre_length, final_state, scope='decoder_pre')
        pre_loss = self.build_loss(pre_logits, decode_pre_y, scope='decoder_pre_loss')
        pre_optimizer = self.build_optimizer(pre_loss, scope='decoder_pre_op')

        # 后一句decoder
        post_logits, post_prediction, post_state = self.build_decoder(decode_post_emb, decode_post_length, final_state, scope='decoder_post', reuse=True)
        post_loss = self.build_loss(post_logits, decode_post_y, scope='decoder_post_loss')
        post_optimizer = self.build_optimizer(post_loss, scope='decoder_post_op')

        inputs = {'initial_state': initial_state, 'encode': encode, 'encode_length': encode_length,
                  'decode_pre_x': decode_pre_x, 'decode_pre_y': decode_pre_y, 'decode_pre_length': decode_pre_length,
                  'decode_post_x':  decode_post_x, 'decode_post_y': decode_post_y, 'decode_post_length': decode_post_length}
        decode_pre = {'pre_optimizer': pre_optimizer, 'pre_loss': pre_loss, 'pre_state': pre_state}
        decode_post = {'post_optimizer': post_optimizer, 'post_loss': post_loss, 'post_state': post_state}

        return inputs, decode_pre, decode_post
