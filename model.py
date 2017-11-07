
import tensorflow as tf
import numpy as np

from collections import namedtuple

from tensorflow.contrib.rnn import RNNCell as RNNCell


def poisson(rate):
    return tf.cast(tf.less(tf.random_uniform(rate.get_shape()), rate), rate.dtype)

def exp_poisson(u, dt=0.001):
    return poisson(dt * tf.exp(u))
                



class PredictiveUnit(RNNCell):
    State = namedtuple("State", ["u", "a", "e", "dF"])
    Output = namedtuple("Output", ["u", "a", "e", "reconstruction"])


    def __init__(self, input_size, layer_size, feedback_size, c, act, Finput=None):
        self._layer_size = layer_size
        self._input_size = input_size
        self._feedback_size = feedback_size
        self._c = c
        self._act = act
        
        self._Finput = Finput
        self._params = None

    @property
    def state_size(self):
        return PredictiveUnit.State(self._layer_size, self._layer_size, self._input_size, (self._input_size, self._layer_size))

    @property
    def output_size(self):
        return PredictiveUnit.Output(self._layer_size, self._layer_size, self._input_size, self._input_size)

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
        return (
            tf.get_variable("F", [self._input_size, self._layer_size], 
                # initializer=tf.uniform_unit_scaling_initializer(factor=c.weight_init_factor)
                initializer=tf.random_uniform_initializer(0.0, c.weight_init_factor),
            ),
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

            # e = tf.Print(e, [feedback])
            
            u_new = s.u + c.step * (
                tf.matmul(e, F) + c.fb_factor * feedback
            )/c.tau

            a_new = self._act(u_new)

            new_dF = s.dF + c.grad_accum_rate * tf.matmul(tf.transpose(e), a_new)
            
            return (
                PredictiveUnit.Output(u_new, a_new, e, x_hat),
                PredictiveUnit.State(u_new, a_new, e, new_dF)
            )


class OutputUnit(PredictiveUnit):
    # @property
    # def state_size(self):
    #     return PredictiveUnit.State(self._layer_size, self._layer_size, self._layer_size, (self._input_size, self._layer_size))

    # @property
    # def output_size(self):
    #     return PredictiveUnit.Output(self._layer_size, self._layer_size, self._layer_size, self._layer_size)

    def __call__(self, input, s, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._params is None:
                self._params = self._init_parameters()

            x, a_target = input[0], input[1]
            c = self._c

            F = self._params[0]

            u_new = tf.matmul(x, F)

            a_new = self._act(u_new)

            e_y = a_target - a_new

            e = tf.matmul(e_y, tf.transpose(F))
            

            new_dF = s.dF + c.grad_accum_rate * tf.matmul(tf.transpose(x), e_y)
            
            return (
                PredictiveUnit.Output(u_new, a_new, e, a_new),
                PredictiveUnit.State(u_new, a_new, e, new_dF)
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
        state.e,
        np.zeros(state.dF.shape)
    )

