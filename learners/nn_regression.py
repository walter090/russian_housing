import tensorflow as tf


class DeepRegressor(object):
    def __init__(self, n_layers, keep_prob=0.9, hidden_layer_spec=None):
        """
        constructor
        :param n_layers: number of hidden layers
        :param keep_prob: keep probability for dropout
        :param hidden_layer_spec: list of layers in the form of list of
         dictionaries of {'num_features': , 'keep_prob': }
        Note that if this argument is set, the previous two 
        arguments are ignored, otherwise all hidden layers shall have the
        same setup. This list should only include hidden layers, as the name
        indicates
        """
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        self.hidden_layer_spec = hidden_layer_spec
        self.layers_w_b = []

        if hidden_layer_spec is not None:
            if hidden_layer_spec.__class__ is not list:
                raise ValueError('layers argument must be a list')
            self.n_layers = len(hidden_layer_spec)
            self.keep_prob = None

    def build_net(self, x):
        """
        This is the method that builds the multilayer neural network for regression
        :param x: training data
        :return: output layer
        """
        n_input = x.shape[2]  # number of input features
        n_output = 1  # number of output neurons

        # The following code constructs the list of weights and biases,
        # since the input and output layers are not included in the hidden_layer_spec,
        # their weights and biases are added before and after the loop.
        input_weights = self.init_weights_biases([n_input, self.hidden_layer_spec[0]['num_features']])
        input_biases = self.init_weights_biases([self.hidden_layer_spec[0]['num_features']])
        self.layers_w_b.append({'weights': input_weights, 'biases': input_biases})

        layer = tf.add(tf.matmul(x, input_weights), input_biases)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, keep_prob=self.hidden_layer_spec[0]['keep_prob'])

        for i in range(self.n_layers - 1):
            weights = self.init_weights_biases([self.hidden_layer_spec[i]['num_features'],
                                               self.hidden_layer_spec[i + 1]['num_features']])

            biases = self.init_weights_biases(shape=[self.hidden_layer_spec[i + 1]['num_features']])

            self.layers_w_b.append({'weights': weights, 'biases': biases})

            layer = tf.add(tf.matmul(layer, weights), biases)
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=self.hidden_layer_spec[i+1]['keep_prob'])

        output_weights = self.init_weights_biases([self.hidden_layer_spec[-1]['num_features'], n_output])

        output_biases = self.init_weights_biases([n_output])

        self.layers_w_b.append({'weights': output_weights, 'biases': output_biases})

        output_layer = tf.add(tf.matmul(layer, output_weights), output_biases)
        return output_layer

    @staticmethod
    def init_weights_biases(shape):
        """
        initialize weights or biases according to the given shape
        :param shape: shape of the variable
        :return: TensorFlow variable
        """
        init = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(init)
