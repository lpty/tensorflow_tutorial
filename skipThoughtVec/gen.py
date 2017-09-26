import tensorflow as tf
import numpy as np
from skipThoughtVec.model import Model


def main():
    model = Model()
    # 输入
    encode, _, _, decode_post_x, _, encode_length, _, decode_post_length = model.build_inputs()
    # embedding
    encode_emb, _, decode_post_emb = model.build_word_embedding(encode, decode_post_x, decode_post_x)
    # encoder构建
    initial_state, final_state = model.build_encoder(encode_emb, encode_length, train=False)
    _, post_prediction, post_state = model.build_decoder(decode_post_emb, decode_post_length, final_state, scope='decoder_post')

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    new_state = sess.run(initial_state)
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    encode_x = [[model.story.word_to_int[c] for c in '掠夺']]
    samples = [[] for _ in range(model.sample_size)]
    samples[0] = encode_x[0]
    for i in range(model.sample_size):
        decode_x = [[model.story.word_to_int['<GO>']]]
        while decode_x[0][-1] != model.story.word_to_int['<EOS>']:
            feed = {encode: encode_x, encode_length: [len(encode_x[0])], initial_state: new_state,
                    decode_post_x: decode_x, decode_post_length: [len(decode_x[0])]}
            predict, state = sess.run([post_prediction, post_state], feed_dict=feed)
            int_word = np.argmax(predict, 1)[-1]
            decode_x[0] += [int_word]
        samples[i] += decode_x[0][1:-1]
        encode_x = [samples[i]]
        new_state = state
        print(''.join([model.story.int_to_word[sample] for sample in samples[i]]))
    sess.close()

if __name__ == '__main__':
    main()
