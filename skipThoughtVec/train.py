import os
import datetime
import tensorflow as tf
from skipThoughtVec.model import Model


def main():
    model = Model()
    inputs, decode_pre, decode_post = model.build()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    new_state = sess.run(inputs['initial_state'])
    for epoch in range(model.epoch):
        # 训练数据生成器
        batches = model.story.batch()
        for encode_x, decode_pre_x, decode_pre_y, decode_post_x, decode_post_y, encode_length, decode_pre_length, decode_post_length in batches:
            if len(encode_x) != model.batch_size: continue
            feed = {inputs['initial_state']: new_state, inputs['encode']: encode_x, inputs['encode_length']: encode_length,
                    inputs['decode_pre_x']: decode_pre_x, inputs['decode_pre_y']: decode_pre_y, inputs['decode_pre_length']: decode_pre_length,
                    inputs['decode_post_x']: decode_post_x, inputs['decode_post_y']: decode_post_y, inputs['decode_post_length']: decode_post_length}
            _, pre_loss, _, _, post_loss, new_state = sess.run([decode_pre['pre_optimizer'], decode_pre['pre_loss'], decode_pre['pre_state'],
                                                                decode_post['post_optimizer'], decode_post['post_loss'], decode_post['post_state']], feed_dict=feed)
            print(datetime.datetime.now().strftime('%c'), ' epoch:', epoch, ' step:', step, ' pre_loss', pre_loss, ' post_loss', post_loss)
            step += 1
    model_path = os.path.join(os.getcwd(), "skipThought.model")
    saver.save(sess, model_path, global_step=step)
    sess.close()

if __name__ == '__main__':
    main()
