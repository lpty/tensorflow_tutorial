from mnistGan.mnist_model import MnistModel

if __name__ == '__main__':
    mnist = MnistModel()
    mnist.gen()
    mnist.show()
