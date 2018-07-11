import numpy as np


def apply_activation(hidden_in):
    tanh_vec = np.vectorize(np.tanh)
    return tanh_vec(hidden_in)


class Autoencoder:

    def __init__(self, number_of_hidden_nodes):
        if number_of_hidden_nodes <= 0:
            raise ValueError('The number of hidden nodes must be an integer grater than 1')
        self._hidden_layer_size = number_of_hidden_nodes

    def get_hidden_layer_size(self):
        return self._hidden_layer_size

    def train(self, data, max_iter=10000):
        datapoint_size = data.shape[1]

        encoding_weights = []
        for i in range(self._hidden_layer_size):
            hidden_node_weights = []
            for j in range(datapoint_size):
                hidden_node_weights.append(np.random.normal())
            encoding_weights.append(hidden_node_weights)
        encoding_weights = np.matrix(encoding_weights)

        decoding_weights = []
        for i in range(self._hidden_layer_size):
            hidden_node_weights = []
            for j in range(datapoint_size):
                hidden_node_weights.append(np.random.normal())
            decoding_weights.append(hidden_node_weights)
        decoding_weights = np.matrix(decoding_weights)

        hidden_in = np.matmul(encoding_weights, data)

        hidden_out = apply_activation(hidden_in)


        print('Training autoencoder with maximum of,', max_iter, 'iterations')