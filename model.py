
import tensorflow as tf
import numpy as np

from collections import namedtuple

from tensorflow.contrib.rnn import RNNCell as RNNCell


def poisson(rate):
    return tf.cast(tf.less(tf.random_uniform(rate.get_shape()), rate), rate.dtype)

def exp_poisson(u, dt=0.001):
    return poisson(dt * tf.exp(u))
                

def xavier_init(fan_in, fan_out, const=0.5):
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999, reuse=False):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope, reuse=reuse):
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(1.0), trainable=False)
        offset = tf.get_variable('offset', [size], trainable=False)

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

    return tf.cond(training, batch_statistics, population_statistics)


class PredictiveUnit(RNNCell):
    State = namedtuple("State", ["u", "a", "a_m", "e", "dF"])
    Output = namedtuple("Output", ["u", "a", "a_m", "e", "reconstruction"])


    def __init__(self, input_size, layer_size, feedback_size, c, act, fb_factor, is_training, Finput=None):
        self._layer_size = layer_size
        self._input_size = input_size
        self._feedback_size = feedback_size
        self._c = c
        self._act = act
        
        self._Finput = Finput
        self._params = None
        self._is_training = is_training
        self._fb_factor = fb_factor

    @property
    def state_size(self):
        return PredictiveUnit.State(self._layer_size, self._layer_size, self._layer_size, self._input_size, (self._input_size, self._layer_size))

    @property
    def output_size(self):
        return PredictiveUnit.Output(self._layer_size, self._layer_size, self._layer_size, self._input_size, self._input_size)

    @property
    def layer_size(self):
        return self._layer_size
    
    @property
    def input_size(self):
        return self._input_size

    @property
    def feedback_size(self):
        return self._feedback_size

    def _init_parameters(self):
        if self._Finput is not None: return (self._Finput, )
        
        c = self._c

        # init = tf.nn.l2_normalize(xavier_init(self._input_size, self._layer_size, c.weight_init_factor), 0)
        init = xavier_init(self._input_size, self._layer_size, c.weight_init_factor)
        
        return (
            tf.Variable(init),
        )

    @property
    def F(self):
        assert not self._params is None
        return self._params[0]
    
    def __call__(self, input, s, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._params is None:
                self._params = self._init_parameters()

            x, feedback = input[0], input[1]
            c = self._c

            F = self._params[0]

            x_hat = tf.matmul(s.a, tf.transpose(F))

            e = x - x_hat

            fb = self._fb_factor * feedback

            ff = tf.matmul(e, F)

            # ff = batch_norm(ff, "ff", self._is_training, epsilon=1e-03, decay=0.9)

            # fb = batch_norm(fb, "fb", self._is_training, epsilon=1e-03, decay=0.9)
            
            dudt = ff + fb

            # dudt = batch_norm(dudt, "PU", self._is_training, epsilon=1e-03, decay=0.9)

            u_new = s.u + c.step * dudt/c.tau
            
            if c.adaptive:
                a_new = self._act(u_new - s.a_m)
            else:
                a_new = self._act(u_new)
            
            # a_new = batch_norm(a_new, "a_new", self._is_training, epsilon=1e-01, decay=0.99)
            
            new_dF = s.dF + c.grad_accum_rate * tf.matmul(tf.transpose(e), a_new)
            
            x_hat_new = tf.matmul(a_new, tf.transpose(F))
            
            new_a_m = s.a_m + (c.adapt_gain*s.a - s.a_m)/c.tau_m

            
            return (
                PredictiveUnit.Output(u_new, a_new, new_a_m, x_hat_new-x, x_hat_new),
                PredictiveUnit.State(u_new, a_new, new_a_m, x_hat_new-x, new_dF)
            )


class OutputUnit(PredictiveUnit):
    # @property
    # def state_size(self):
    #     return PredictiveUnit.State(self._layer_size, self._layer_size, self._layer_size, (self._input_size, self._layer_size))

    # @property
    # def output_size(self):
    #     return PredictiveUnit.Output(self._layer_size, self._layer_size, self._input_size, self._layer_size)

    def __call__(self, input, s, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._params is None:
                self._params = self._init_parameters()

            x, a_target = input[0], input[1]
            c = self._c

            F = self._params[0]

            u_new = tf.matmul(x, F)

            a_new = self._act(u_new)

            # e_y = -tf.gradients(
            #     tf.nn.softmax_cross_entropy_with_logits(logits=u_new ,labels=a_target), u_new
            # )[0]
            
            e_y = a_target - a_new
            
            # e = tf.matmul(e_y, tf.transpose(F))
            e = tf.matmul(e_y, tf.transpose(tf.nn.l2_normalize(F, 0)))
            
            new_dF = s.dF + c.grad_accum_rate * (tf.matmul(tf.transpose(x), e_y) ) #- c.regularization * F)
            
            return (
                PredictiveUnit.Output(u_new, a_new, a_new, e, a_new),
                PredictiveUnit.State(u_new, a_new, a_new, e, new_dF)
            )



class FeedbackNet(RNNCell):
    def __init__(self, *cells):
        self._cells = cells

        self._c = self._cells[0]._c

    @property
    def output_size(self):
        return tuple(cell.output_size for cell in self._cells)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    def __call__(self, input, states, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c = self._c
        
            xt, yt = input

            xit = xt
            
            new_states = []
            new_outputs = []
            for i, (cell, state) in enumerate(zip(self._cells, states)):
                last_cell = i == len(self._cells)-1
                
                feedback_to_cell = states[i+1].e if not last_cell else yt
                
                it, ns = cell(
                    (
                        xit, 
                        feedback_to_cell, 
                    ), 
                    state,
                    scope="cell_{}".format(i)
                )

                new_outputs.append(it)
                new_states.append(ns)

                xit = it.a


            return tuple(new_outputs), tuple(new_states)

    @property
    def cells(self):
        return self._cells
    


def reset_state_fn(state):
    return PredictiveUnit.State(
        state.u,
        state.a,
        state.a_m,
        state.e,
        np.zeros(state.dF.shape)
    )

