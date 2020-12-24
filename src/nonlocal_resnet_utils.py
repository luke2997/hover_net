import tensorflow as tf
from resnet_utils import conv2d_same
from resnet_v1 import resnet_arg_scope
slim = tf.contrib.slim


def instantiate_block(net, block):
  with tf.variable_scope(block.scope, values=[net]) as sc:
    for i, unit in enumerate(block.args):
      with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
        net = block.unit_fn(net, rate=1, **unit)
    return slim.utils.collect_named_outputs(None, sc.name, net)


def nonlocal_dot(net, depth, embed=True, softmax=False, maxpool=2, scope=None):

  with tf.variable_scope(scope, 'nonlocal', values=[net]) as sc:
    with slim.arg_scope([slim.conv2d], normalizer_fn=None):
      if embed:
        a = conv2d_same(net, depth, 1, stride=1, scope='embA')
        b = conv2d_same(net, depth, 1, stride=1, scope='embB')
      else:
        a, b = net, net
      g_orig = g = conv2d_same(net, depth, 1, stride=1, scope='g')
    if maxpool is not False and maxpool > 1:
      b = slim.max_pool2d(b, [maxpool, maxpool], stride=maxpool, scope='pool')
      g = slim.max_pool2d(g, [maxpool, maxpool], stride=maxpool, scope='pool')

    # Flatten from (B,H,W,C) to (B,HW,C) or similar
    a_flat = tf.reshape(a, [tf.shape(a)[0], -1, tf.shape(a)[-1]])
    b_flat = tf.reshape(b, [tf.shape(b)[0], -1, tf.shape(b)[-1]])
    g_flat = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
    a_flat.set_shape([a.shape[0], a.shape[1] * a.shape[2] if None not in a.shape[1:3] else None, a.shape[-1]])
    b_flat.set_shape([b.shape[0], b.shape[1] * b.shape[2] if None not in b.shape[1:3] else None, b.shape[-1]])
    g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])
    # Compute f(a, b) -> (B,HW,HW)
    f = tf.matmul(a_flat, tf.transpose(b_flat, [0, 2, 1]))
    if softmax:
        f = tf.nn.softmax(f)
    else:
        f = f / tf.cast(tf.shape(f)[-1], tf.float32)
    # Compute f * g ("self-attention") -> (B,HW,C)
    fg = tf.matmul(f, g_flat)
    # Expand and fix the static shapes TF lost track of.
    fg = tf.reshape(fg, tf.shape(g_orig))
    # fg.set_shape(g.shape)  # NOTE: This actually appears unnecessary.

    # Go back up to the original depth, add residually, zero-init.
    #with slim.arg_scope([slim.conv2d],
    #                    weights_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.batch_norm], param_initializers={'gamma': tf.zeros_initializer()}):
      fg = conv2d_same(fg, net.shape[-1], 1, stride=1, scope='fgup')
    net = net + fg

    return slim.utils.collect_named_outputs(None, sc.name, net)
