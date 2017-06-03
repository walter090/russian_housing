import tensorflow as tf


class DeepRegressor(object):
    def __init__(self, n_layers, n_input_features, keep_prob=0.95, hidden_layer_spec=None):
        """Constructor
        :param n_layers: number of hidden layers
        :param n_input_features: number of input features
        :param keep_prob: keep probability for dropout
        :param hidden_layer_spec: list of layers in the form of list of
         dictionaries of {'num_features': , 'keep_prob': }
        Note that if this argument is set, the previous two 
        arguments are ignored, otherwise all hidden layers shall have the
        same setup. This list should only include hidden layers, as the name
        indicates
        """
        self.n_layers = n_layers
        self.keep_prob_value = keep_prob
        self.hidden_layer_spec = hidden_layer_spec
        self.layers_w_b = []
        self.keep_prob = tf.placeholder(dtype='float')
        self.x = tf.placeholder(dtype='float', shape=[None, len(n_input_features)])
        self.y = tf.placeholder(dtype='float', shape=[None])

        if hidden_layer_spec is not None:
            if hidden_layer_spec.__class__ is not list:
                raise ValueError('layers argument must be a list')
            self.n_layers = len(hidden_layer_spec)
            self.keep_prob = None

    def build_net(self, x):
        """This is the method that builds the multilayer neural network for regression
        :param x: Training data
        :return: output layer (prediction)
        """
        n_input = x.shape[1]  # number of input features
        n_output = 1  # number of output neurons

        # The following code constructs the list of weights and biases,
        # since the input and output layers are not included in the hidden_layer_spec,
        # their weights and biases are added before and after the loop.
        input_weights = self.__init_weights_biases([n_input, self.hidden_layer_spec[0]['num_features']])
        input_biases = self.__init_weights_biases([self.hidden_layer_spec[0]['num_features']])
        self.layers_w_b.append({'weights': input_weights, 'biases': input_biases})

        layer = tf.add(tf.matmul(x, input_weights), input_biases)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, keep_prob=self.hidden_layer_spec[0]['keep_prob'])

        for i in range(self.n_layers - 1):
            weights = self.__init_weights_biases([self.hidden_layer_spec[i]['num_features'],
                                                  self.hidden_layer_spec[i + 1]['num_features']])

            biases = self.__init_weights_biases(shape=[self.hidden_layer_spec[i + 1]['num_features']])

            self.layers_w_b.append({'weights': weights, 'biases': biases})

            layer = tf.add(tf.matmul(layer, weights), biases)
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=self.hidden_layer_spec[i + 1]['keep_prob'])

        output_weights = self.__init_weights_biases([self.hidden_layer_spec[-1]['num_features'], n_output])

        output_biases = self.__init_weights_biases([n_output])

        self.layers_w_b.append({'weights': output_weights, 'biases': output_biases})

        output_layer = tf.add(tf.matmul(layer, output_weights), output_biases)
        return output_layer

    @staticmethod
    def __init_weights_biases(shape):
        """Initialize weights or biases according to the given shape
        :param shape: shape of the variable
        :return: TensorFlow variable
        """
        init = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def launch(self,
               batch_size,
               x_train,
               y_train,
               optimizer_class=tf.train.AdamOptimizer,
               train_epochs=1000,
               verbose_every=50):
        """This function launches the session and starts training the
        built neural net
        :param verbose_every: int, this argument sets the number of epochs before a
        status display
        :param y_train: training data target label
        :param x_train: training data features
        :param batch_size: int, size of each batch
        :param train_epochs: number of epochs to train the model before stopping
        :param optimizer_class: Tensorflow optimizer class, default AdamOptimizer
        :return: None
        """
        predictions = self.build_net(self.x)
        cost = tf.reduce_mean(tf.square(predictions - self.y))
        optimizer = optimizer_class().minimize(cost)
        correct_predictions = tf.equal(predictions, y_train)
        train_accuracy = tf.reduce_mean(correct_predictions)

        total_train_accuracy = 0
        total_cost = 0

        initializer = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(initializer)
            total_batch = len(x_train) // batch_size

            # Training cycles
            for epoch in range(train_epochs):
                for batch in range(total_batch - 1):
                    x_batch = x_train[batch * batch_size: (batch + 1) * batch_size]
                    y_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
                    _, c, pred, acc = session.run([optimizer, cost, predictions, train_accuracy],
                                                  feed_dict={self.x: x_batch,
                                                             self.y: y_batch,
                                                             self.keep_prob: self.keep_prob_value})
                    total_train_accuracy += acc / total_batch
                    total_cost += c / total_batch

                if epoch % verbose_every == 0:
                    print 'At epoch {0}, Total training accuracy: {1:.2f}, total cost: {2:.2f}'\
                        .format(epoch, total_train_accuracy, total_cost)
            print 'Training complete'
