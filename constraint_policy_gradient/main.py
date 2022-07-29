from network import NN
import numpy as np


def main():

    # random seed
    np.random.seed(10)

    nn = NN([2, 1])

    nn.train([[[[1, 2]], [0], [1]]])


if __name__ == '__main__':
    main()
