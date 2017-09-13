import os
import numpy as np
from collections import Counter
from glob import glob


class Text:

    def __init__(self):
        self.data_name = 'text_c10'
        self.file_list = self._get_list()
        self.text_list = [self._get_text(file_name) for file_name in self.file_list]
        self.vocab, self.words, self.vocab_to_int, self.int_to_vocab = self._get_words()
        self.batch_size = 200
        self.chunk_size = len(self.words) // self.batch_size

    def _get_list(self):
        # 获取文本列表
        path = os.path.join(os.getcwd(), self.data_name, '*', '*.txt')
        return glob(path)

    def _get_text(self, file_name):
        # 获取文本内容
        f = open(file_name, 'r', encoding='utf-8')
        text = self._process_text(f.read())
        return text

    def _get_words(self, freq=15, t=1e-5, threshold=0.981):
        # 所有词
        all_word = ''.join(self.text_list).split()
        word_counts = Counter(all_word)
        # 剔除出现频率低的词, 减少噪音
        words = [word for word in all_word if word_counts[word] > freq]

        # 统计单词出现频次
        word_counts = Counter(words)
        total_count = len(words)
        # 计算单词频率
        word_freqs = {w: c / total_count for w, c in word_counts.items()}
        # 计算被删除的概率
        prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in word_counts}
        # 剔除出现频率太高的词
        train_words = [w for w in words if prob_drop[w] < threshold]

        vocab = sorted(set(train_words))
        vocab_to_int = {w: c for c, w in enumerate(vocab)}
        int_to_vocab = {c: w for c, w in enumerate(vocab)}
        return vocab, train_words, vocab_to_int, int_to_vocab

    @staticmethod
    def _get_target(words, index, window_size=8):
        # 获取上下文单词
        window = np.random.randint(1, window_size+1)
        start = index - window if (index - window) else 0
        end = index + window
        targets = set(words[start:index] + words[index+1:end])
        return list(targets)

    def _get_vector(self, words):
        return [self.vocab_to_int[word] for word in words]

    @staticmethod
    def _process_text(text):
        marks = ['.', ',', '"', ';', '!', '?', '(', ')', '--', ':', '-']
        for mark in marks:
            text = text.replace(mark, '')
        return text

    def batch(self):
        # 生成器
        start, end = 0, self.batch_size
        for _ in range(self.chunk_size):
            batch_x, batch_y = [], []
            words = self.words[start:end]
            for index in range(self.batch_size):
                x = words[index]
                y = self._get_target(words, index)
                batch_x.extend([x] * len(y))
                batch_y.extend(y)
            yield self._get_vector(batch_x), self._get_vector(batch_y)
            start += self.batch_size
            end += self.batch_size

if __name__ == '__main__':
    T = Text()
    b = T.batch()
    for bx, by in b:
        print(bx)
        print(by)
