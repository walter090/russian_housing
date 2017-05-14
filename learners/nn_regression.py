import tensorflow as tf


class DeepRegressor(object):
    def __init__(self, n_layers, keep_prob=0.9, layers=None):
        """
        constructor
        :param n_layers: number of hidden layers
        :param keep_prob: keep probability for dropout
        :param layers: list of layers in the form of list of
         dictionaries of {'num_features': , 'keep_prob': }
        if value to key 'keep_prob' is None, the layer does not have 
        a dropout. Note that if this argument is set, the previous two 
        arguments are ignored
        """
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        self.layers = layers

        if layers is not None:
            if layers.__class__ is not list:
                raise ValueError('layers argument must be a list')
            self.n_layers = len(layers)
            self.keep_prob = None

    def build_net(self):
        raise NotImplementedError
