import numpy as np
import tensorflow as tf

def linear(input_, output_size, name, stddev=0.002, bias_start=0.0,
           with_w=False, reuse=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(name, reuse=reuse):
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size], tf.float32,
        initializer=tf.constant_initializer(bias_start)
    )
    bias = tf.get_variable(
        "bias", [output_size], initializer=tf.constant_initializer(bias_start)
    )
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def qlinear(input_, output_size, name, stddev=0.002, bias_start=0.0,
             with_w=False, reuse=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(name, reuse=reuse):
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size], tf.float32,
         tf.random_normal_initializer(stddev=stddev)
    )
    bias_init = np.tile(np.array([1,0,0,0]), output_size // 4-1)
    bias_init = np.concatenate([bias_init, np.array([0,0,0,0])])
    bias = tf.get_variable(
        "bias", [output_size], initializer=tf.constant_initializer(bias_init)
    )
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def conv1d(input_, output_dim, k_w=25, d_w=2, name="conv1d",
           padding="SAME", reuse=False):
  with tf.variable_scope(name, reuse=reuse):
    initializer = tf.contrib.layers.xavier_initializer()
    w = tf.get_variable("w", [k_w, input_.get_shape()[-1], output_dim],
                        initializer=initializer)
    conv = tf.nn.conv1d(input_, w, stride=d_w, padding=padding)

    biases = tf.get_variable("biases", [output_dim],
                             initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def batch_norm(inputs, name, train=True, decay=0.9, epsilon=1e-5, reuse=False):
  return tf.contrib.layers.batch_norm(inputs=inputs, decay=decay,
                                      updates_collections=None,
                                      epsilon=epsilon, is_training=train,
                                      reuse=reuse, scope=name,scale=True)


def instance_norm(input_, name="instance_norm"):
  with tf.variable_scope(name):
    depth = input_.get_shape()[-1]

    scale_init = tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32)
    scale = tf.get_variable("scale", [depth], initializer=scale_init)

    offset_init = tf.constant_initializer(0.0)
    offset = tf.get_variable("offset", [depth], initializer=offset_init)

    mean, variance = tf.nn.moments(input_, axes=[1], keep_dims=True)

    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input_ - mean) * inv
    return scale*normalized + offset


def relu(x):
  return tf.nn.relu(x)


def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def sigmoid(x):
  return tf.nn.sigmoid(x)


def get_vels(input_, n_joints, dmean, dstd, omean, ostd):
  joints = tf.reshape(
      input_[:,:,:-4], input_.shape[:-1].as_list() + [n_joints, 3]
  ) * dstd[None] + dmean[None]
  root_x = input_[:,:,-4] * ostd[0,0] + omean[0,0]
  root_y = input_[:,:,-3] * ostd[0,1] + omean[0,1]
  root_z = input_[:,:,-2] * ostd[0,2] + omean[0,2]
  root_r = input_[:,:,-1] * ostd[0,3] + omean[0,3]

  rotation = np.repeat(
      np.array([[[1., 0., 0., 0.]]]), int(input_.shape[0]), axis=0
  ).astype("float32")
  rotation = tf.constant(rotation)
  translation = np.repeat(
      np.array([[[0., 0., 0.]]]), int(input_.shape[0]), axis=0
  ).astype("float32")
  translation = tf.constant(translation)
  axis = np.repeat(
      np.array([[0., 1., 0.]]), int(input_.shape[0]), axis=0
  ).astype("float32")
  axis = tf.constant(axis)
  joints_list = []

  for t in xrange(int(joints.shape[1])):
    joints_list.append(q_mul_v(rotation, joints[:,t,:,:]))
    joints_x = joints_list[-1][:,:,0:1] + translation[:,0:1,0:1]
    joints_y = joints_list[-1][:,:,1:2] + translation[:,0:1,1:2]
    joints_z = joints_list[-1][:,:,2:3] + translation[:,0:1,2:3]
    joints_list[-1] = tf.concat([joints_x, joints_y, joints_z], axis=-1)

    rotation = q_mul_q(from_angle_axis(-root_r[:,t], axis), rotation)

    translation += q_mul_v(
        rotation,
        tf.concat(
            [root_x[:,t:t+1], root_y[:,t:t+1], root_z[:,t:t+1]], axis=-1
        )[:,None,:]
    )

  joints = tf.reshape(
      tf.stack(joints_list, axis=1), input_.shape[:-1].as_list() + [-1]
  )
  return joints[:,1:,:] - joints[:,:-1,:], joints#[:,:-1,:]


def q_mul_v(a, b):
  vs = tf.concat(
      [tf.zeros(b.shape[:-1].as_list() + [1]), b], axis=-1
  )
  return q_mul_q(a, q_mul_q(vs, q_neg(a)))[...,1:4]


def q_mul_q(a, b):
  sqs, oqs = q_broadcast(a, b)

  q0 = sqs[...,0:1]; q1 = sqs[...,1:2];
  q2 = sqs[...,2:3]; q3 = sqs[...,3:4];
  r0 = oqs[...,0:1]; r1 = oqs[...,1:2];
  r2 = oqs[...,2:3]; r3 = oqs[...,3:4];

  qs0 = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
  qs1 = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
  qs2 = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
  qs3 = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

  return tf.concat([qs0, qs1, qs2, qs3], axis=-1)


def q_neg(a):
  return a * np.array([[[1, -1, -1, -1]]])


def q_broadcast(sqs, oqs):
  if int(sqs.shape[-2]) == 1:
    sqsn = []
    for l in xrange(oqs.shape[-2]):
      sqsn.append(sqs)
    sqs = tf.concat(sqsn, axis=-2)

  if int(oqs.shape[-2]) == 1:
    oqsn = []
    for l in xrange(sqs.shape[-2]):
      oqsn.append(oqs)
    oqs = tf.concat(oqsn, axis=-2)

  return sqs, oqs


def from_angle_axis(angles, axis):
  axis = axis / (tf.sqrt(tf.reduce_sum(axis ** 2, axis=-1)) + 1e-10)[...,None]
  sines = tf.sin(angles / 2.0)[...,None]
  cosines = tf.cos(angles / 2.0)[...,None]
  return tf.concat([cosines, axis * sines], axis=-1)[:,None,:]


def gaussian_noise(input_, input_mean, input_std, stddev):
  noise = tf.random_normal(
      shape=input_.shape, mean=0.0, stddev=stddev, dtype=tf.float32
  )
  noisy_input = noise + input_ * input_std + input_mean
  return (noisy_input - input_mean) / input_std

