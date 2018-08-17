# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:44:32 2018

@author: yxr
"""
from tensorflow.python.ops.rnn_cell_impl import *
import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class yxrBasicLSTMCell(BasicLSTMCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self,
               num_units,
               num_skillls,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.

      When restoring from CudnnLSTM-trained checkpoints, must use
      `CudnnCompatibleLSTMCell` instead.
    """
    super(BasicLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

    self.num_skills=num_skillls

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    #input_depth = inputs_shape[1].value
    input_depth = inputs_shape[1].value-1   #减去了time这一位
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * num_units]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)

    print('inputs_shape:',inputs.shape)

    batch_size = tf.shape(inputs)[0]
    x = tf.slice(inputs, [0,1], [batch_size, 2*self.num_skills])
    # #t = tf.slice(inputs, [0,0], [batch_size,1])
    inputs=x

    # Map elapse time in days or months
    #T = self.map_elapse_time(t)

    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, h], 1), self._kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = math_ops.add
    multiply = math_ops.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    print('shape:',inputs.shape,'\n')
    return new_h, new_state


  # def map_elapse_time(self, t):
  #     c1 = tf.constant(1, dtype=tf.float32)  # 1
  #     c2 = tf.constant(2.7183, dtype=tf.float32) # e

  #     # T = tf.multiply(self.wt, t) + self.bt
  #     T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
  #     Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)
  #     T = tf.matmul(T, Ones)

  #     return T
