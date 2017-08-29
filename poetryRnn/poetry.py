import numpy as np


class Poetry:

    def __init__(self):
        self.poetry_file = 'poetry.txt'
        self.poetry_list = self._get_poetry()
        self.poetry_vectors, self.word_to_int, self.int_to_word = self._gen_poetry_vectors()
        self.batch_size = 64
        self.chunk_size = len(self.poetry_vectors) // self.batch_size

    def _get_poetry(self):
        with open(self.poetry_file, "r", encoding='utf-8') as f:
            poetry_list = [line for line in f]
        return poetry_list

    def _gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetry_list)+' '))
        # 每一个字符分配一个索引 为后续诗词向量化做准备
        int_to_word = {i: word for i, word in enumerate(words)}
        word_to_int = {v: k for k, v in int_to_word.items()}
        to_int = lambda word: word_to_int.get(word)
        poetry_vectors = [list(map(to_int, poetry)) for poetry in self.poetry_list]
        return poetry_vectors, word_to_int, int_to_word

    def batch(self):
        # 生成器
        start = 0
        end = self.batch_size
        for _ in range(self.chunk_size):
            batches = self.poetry_vectors[start:end]
            # 输入数据 按每块数据中诗句最大长度初始化数组，缺失数据补全
            x_batch = np.full((self.batch_size, max(map(len, batches))), self.word_to_int[' '], np.int32)
            for row in range(self.batch_size): x_batch[row, :len(batches[row])] = batches[row]
            # 标签数据 根据上一个字符预测下一个字符 所以这里y_batch数据应为x_batch数据向后移一位
            y_batch = np.copy(x_batch)
            y_batch[:, :-1], y_batch[:, -1] = x_batch[:, 1:], x_batch[:, 0]
            yield x_batch, y_batch
            start += self.batch_size
            end += self.batch_size

if __name__ == '__main__':
    data = Poetry().batch()
    for x, y in data:
        print(x)
