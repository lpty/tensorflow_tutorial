import os
import numpy as np
from glob import glob
from collections import Counter


class Story:

    def __init__(self):
        self.data_name = 'data'
        self.sentence_list = self._get_sentence_list()
        self.vocab, self.word_to_int, self.int_to_word = self._get_vocab()
        self.batch_size = 128
        self.chunk_size = len(self.sentence_list) // self.batch_size

    def _get_sentence_list(self):
        # 读取本地文本
        path = os.path.join(os.getcwd(), self.data_name, '*.txt')
        file_name = glob(path)
        file_list = [self._get_file_content(f) for f in file_name]
        # 处理特殊符号
        process_words = self._process_words(file_list)
        sentence_list = [p for p in process_words.split('\\') if len(p)]
        # 处理出现频次过多的句子
        return self._process_sentence_list(sentence_list)

    @staticmethod
    def _process_sentence_list(sentence_list, t=1e-5, threshold=0.5):
        sentence_count = Counter(sentence_list)
        total_count = len(sentence_list)
        # 计算句子频率
        sentence_freqs = {w: c / total_count for w, c in sentence_count.items()}
        # 计算被删除的概率
        prob_drop = {w: 1 - np.sqrt(t / sentence_freqs[w]) for w in sentence_count}
        # 剔除出现频率太高的句子
        sentence_list = [w for w in sentence_list if prob_drop[w] < threshold]
        return sentence_list

    @staticmethod
    def _get_file_content(filename):
        with open(filename, encoding='utf-8') as f:
            file_content = f.read()
        return file_content

    @staticmethod
    def _process_words(file_list):
        words = ''.join(file_list)
        vocab = sorted(set(words))
        mask = vocab[:110]+vocab[-57:]
        mark = ['!', ',', ':', ';', '?', '~', '…', '、', '。', '.', '？', '；', '：', '．', '，', '！']
        for m in mask:
            words = words.replace(m, '\\') if m in mark else words.replace(m, '')
        return words

    def _get_vocab(self):
        # 生成词字典
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        words = ''.join(self.sentence_list)
        vocab = sorted(set(words))+special_words
        word_to_int = {w: i for i, w in enumerate(vocab)}
        int_to_word = {i: w for i, w in enumerate(vocab)}
        return vocab, word_to_int, int_to_word

    @staticmethod
    def _get_target(sentences, index, window_size=1):
        # 获取句子相邻句子
        start = index - window_size if (index - window_size) > 0 else 0
        end = index + 2*window_size
        targets = set(sentences[start:index] + sentences[index+1:end])
        return list(targets)

    def get_vector(self, batch):
        # 句子转化为向量
        return [self.word_to_int.get(word, self.word_to_int['<UNK>']) for word in batch]

    def to_full_batch(self, batch):
        # 对每个batch进行补全
        max_length = max(self.get_batch_length(batch))
        batch_size = len(batch)
        full_batch = np.full((batch_size, max_length), self.word_to_int['<PAD>'], np.int32)
        for row in range(batch_size): full_batch[row, :len(batch[row])] = batch[row]
        return full_batch

    @staticmethod
    def get_batch_length(batch):
        # 获取batch中每个向量长度
        return [len(i) for i in batch]

    def batch(self):
        #
        start, end = 0, self.batch_size
        for _ in range(self.chunk_size):
            batch_x, batch_y, batch_z = [], [], []
            sentences = self.sentence_list[start:end]
            for index in range(self.batch_size):
                x = sentences[index]
                targets = self._get_target(sentences, index)
                if len(targets) < 2: continue
                y, z = targets
                batch_x.extend([x])
                batch_y.extend([y])
                batch_z.extend([z])
            encode_x = [self.get_vector(x) for x in batch_x]
            encode_length = self.get_batch_length(encode_x)
            encode_x = self.to_full_batch(encode_x)

            decode_pre_x = [self.get_vector(['<GO>']) + self.get_vector(y) for y in batch_y]
            decode_pre_y = [self.get_vector(y) + self.get_vector(['<EOS>']) for y in batch_y]
            decode_pre_length = self.get_batch_length(decode_pre_x)
            decode_pre_x = self.to_full_batch(decode_pre_x)
            decode_pre_y = self.to_full_batch(decode_pre_y)

            decode_post_x = [self.get_vector(['<GO>']) + self.get_vector(z) for z in batch_z]
            decode_post_y = [self.get_vector(z) + self.get_vector(['<EOS>']) for z in batch_z]
            decode_post_length = self.get_batch_length(decode_post_x)
            decode_post_x = self.to_full_batch(decode_post_x)
            decode_post_y = self.to_full_batch(decode_post_y)

            yield encode_x, decode_pre_x, decode_pre_y, decode_post_x, decode_post_y, encode_length, decode_pre_length, decode_post_length
            start += self.batch_size
            end += self.batch_size

if __name__ == '__main__':
    s = Story()
    b = s.batch()
    for _ in b:
        pass
