from poetryRnn.poetry_model import PoetryModel


if __name__ == '__main__':
    poetry = PoetryModel()
    poetry.train(epoch=20)
