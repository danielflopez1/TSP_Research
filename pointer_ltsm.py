from keras import initializers
from keras.activations import tanh, softmax
from keras.layers import Recurrent
from keras.engine import InputSpec
import keras.backend as K
import numpy as np

class PointerLSTM(Recurrent):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM, self).__init__(*args, **kwargs)
        self._trainable_weights = []


    def time_distributed_dense(self,x, w, b=None, dropout=None,
                                input_dim=None, output_dim=None,
                                timesteps=None, training=None):
        """Apply `y . w + b` for every temporal slice y of x.
        # Arguments
            x: input tensor.
            w: weight matrix.
            b: optional bias vector.
            dropout: wether to apply dropout (same dropout mask
                for every temporal slice of the input).
            input_dim: integer; optional dimensionality of the input.
            output_dim: integer; optional dimensionality of the output.
            timesteps: integer; optional number of timesteps.
            training: training phase tensor or boolean.
        # Returns
            Output tensor.
            """

        if not input_dim:
            self.input_dim = K.shape(x)[2]
        if not timesteps:
            timesteps = K.shape(x)[1]
        if not output_dim:
            self.output_dim = K.shape(w)[1]

        if dropout is not None and 0. < dropout < 1.:
            # apply the same dropout pattern at every timestep
            ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
            dropout_matrix = K.dropout(ones, dropout)
            expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
            x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

        # collapse time dimension and batch dimension together
        x = K.reshape(x, (-1, input_dim))
        x = K.dot(x, w)
        if b is not None:
            x = K.bias_add(x, b)
        # reshape to 3D tensor
        if K.backend() == 'tensorflow':
            x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
            x.set_shape([None, None, output_dim])
        else:
            x = K.reshape(x, (-1, timesteps, output_dim))
        return x
    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        init = initializers.get('orthogonal')
        self.W1 = init((self.hidden_shape, 1))
        self.W2 = init((self.hidden_shape, 1))
        self.vt = init((input_shape[1], 1))
        print("--------", [self.W1, self.W2, self.vt])
        self._trainable_weights += [self.W1, self.W2, self.vt]

    def get_initial_states(self, X):
        self.input_dim = K.shape(X)[2]
        self.output_dim = K.shape(X)[1]
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = self.time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = self.time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])
