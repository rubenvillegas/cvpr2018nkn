"""Based on Daniel Holden code from http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing"""

import tensorflow as tf

class FK(object):
  def __init__(self):
    pass

  def transforms_multiply(self, t0s, t1s):
    return tf.matmul(t0s, t1s)

  def transforms_blank(self, rotations):
    diagonal = tf.diag([1.0, 1.0, 1.0, 1.0])[None, None, :, :]
    ts = tf.tile(
        diagonal, [int(rotations.shape[0]), int(rotations.shape[1]), 1, 1]
    )

    return ts

  def transforms_rotations(self, rotations):
    q_length = tf.sqrt(tf.reduce_sum(tf.square(rotations), axis=-1))
    qw = rotations[...,0] / q_length
    qx = rotations[...,1] / q_length
    qy = rotations[...,2] / q_length
    qz = rotations[...,3] / q_length

    """Unit quaternion based rotation matrix computation"""
    x2 = qx + qx; y2 = qy + qy; z2 = qz + qz;
    xx = qx * x2; yy = qy * y2; wx = qw * x2;
    xy = qx * y2; yz = qy * z2; wy = qw * y2;
    xz = qx * z2; zz = qz * z2; wz = qw * z2;

    dim0 = tf.stack(values=[1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)
    dim1 = tf.stack(values=[xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)
    dim2 = tf.stack(values=[xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)
    m = tf.stack(values=[dim0, dim1, dim2], axis=-2)

    return m

  def transforms_local(self, positions, rotations):
    transforms = self.transforms_rotations(rotations)
    transforms = tf.concat(values=[transforms, positions[:,:,:,None]], axis=-1)
    zeros = tf.zeros([int(transforms.shape[0]), int(transforms.shape[1]), 1, 3])
    ones = tf.ones([int(transforms.shape[0]), int(transforms.shape[1]), 1, 1])
    zerosones = tf.concat(values=[zeros, ones], axis=-1)
    transforms = tf.concat(values=[transforms, zerosones], axis=-2)
    return transforms

  def transforms_global(self, parents, positions, rotations):
    locals = self.transforms_local(positions, rotations)
    globals = self.transforms_blank(rotations)

    globals = tf.concat([locals[:,0:1], globals[:,1:]], axis=1)
    globals = tf.split(globals, int(globals.shape[1]), axis=1)
    for i in range(1, positions.shape[1]):
      globals[i] = self.transforms_multiply(
          globals[parents[i]][:,0], locals[:,i]
      )[:,None,:,:]

    return tf.concat(values=globals, axis=1)

  def run(self, parents, positions, rotations):
    positions = self.transforms_global(parents, positions, rotations)[:,:,:,3]
    return positions[:,:,:3] / positions[:,:,3,None]

