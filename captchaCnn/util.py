import numpy as np
from captchaCnn.captcha_gen import gen_captcha_text_and_image
from captchaCnn.captcha_gen import CAPTCHA_LIST, CAPTCHA_LEN, CAPTCHA_HEIGHT, CAPTCHA_WIDTH


def convert2gray(img):
    '''
    图片转为黑白，3维转1维
    :param img:
    :return:
    '''
    if len(img.shape) > 2:
        img = np.mean(img, -1)
    return img


def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    '''
    验证码文本转为向量
    :param text:
    :param captcha_len:
    :param captcha_list:
    :return:
    '''
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len): vector[captcha_list.index(text[i])+i*len(captcha_list)] = 1
    return vector


def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    '''
    验证码向量转为文本
    :param vec:
    :param captcha_list:
    :param size:
    :return:
    '''
    # if np.size(np.shape(vec)) is not 1:
    #     raise ValueError('向量限定为1维')
    # vec = np.reshape(vec, (size, -1))
    # vec_idx = np.argmax(vec, 1)
    vec_idx = vec
    text_list = [captcha_list[v] for v in vec_idx]
    return ''.join(text_list)


def wrap_gen_captcha_text_and_image(shape=(60, 160, 3)):
    '''
    返回特定shape图片
    :param shape:
    :return:
    '''
    while True:
        t, im = gen_captcha_text_and_image()
        if im.shape == shape: return t, im


def next_batch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    '''
    获取训练图片组
    :param batch_count:
    :param width:
    :param height:
    :return:
    '''
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    # 返回该训练批次
    return batch_x, batch_y

if __name__ == '__main__':
    x, y = next_batch(batch_count=1)
    print(x, y)



