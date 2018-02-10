from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, \
    BatchNormalization, Dropout, LeakyReLU, ThresholdedReLU

from models.specs import TFNetwork
from lib.nn.losses import weighted_sigmoid, weighted_l1


def unet_encdoer(x, size, scope, down_layers=5, flat_layers=1, dropout=0.0, act_fn=LeakyReLU, batch_norm=True, reuse=None):
    ep_collection = '%s_end_points' % scope
    _upconv = []

    with tf.variable_scope(scope, scope, [x], reuse=reuse) as sc:
        net = x
        # Multiply chans and half spatial size
        for i in range(1, down_layers + 1):
            conv, pool = encoder_block(net, size=size, scope='conv%d' % i, ep_collection=ep_collection,
                                       dropout=dropout, act_fn=act_fn, batch_norm=batch_norm, reuse=reuse)
            net = pool
            size *= 2
            _upconv.append(conv)

        # flat region
        for j in range(i+1, i+1 + flat_layers):
            conv = encoder_block(net, size=size, pool=False, scope='conv%d' % j, ep_collection=ep_collection, reuse=reuse)
            net = conv

    return net, _upconv


def unet_decoder(x, out_size, upconvs, scope, last_act_fn=None, act_fn='relu', dropout=0.0, batch_norm=True, reuse=None):
    ep_collection = '%s_end_points' % scope
    with tf.variable_scope(scope, scope, [x], reuse=reuse) as sc:
        net = x
        last, rest = upconvs[::-1][-1], upconvs[::-1][:-1]
        for n, upconv in enumerate(rest):
            conv = decoder_block(net, upconv, scope='conv%d' % (n+1), act_fn=act_fn, reuse=reuse,
                                 batch_norm=batch_norm, dropout=dropout, ep_collection=ep_collection)
            net = conv

        net = decoder_block(net, last, size=out_size, scope='conv_last', dropout=0.0, batch_norm=False, reuse=reuse,
                               act_fn=act_fn, ep_collection=ep_collection)
        tf.add_to_collection(ep_collection, net)
        logits = Conv2D(out_size, (1, 1), activation=last_act_fn, padding='same')(net)
        tf.add_to_collection(ep_collection, logits)
    return logits


def encoder_block(x, scope, size, ksize=(3, 3), pool_size=(2, 2), act_fn=LeakyReLU, reuse=None, ep_collection='end_points',
                  pool=True, batch_norm=False, dropout=0.0):
    with tf.variable_scope(scope, scope, [x], reuse=reuse) as sc:
        if batch_norm:
            x = BatchNormalization()(x, training=True)
            tf.add_to_collection(ep_collection, x)
        conv = Conv2D(size, ksize, activation=None, padding='same')(x)
        conv = act_fn(0.2)(conv)
        tf.add_to_collection(ep_collection, conv)
        conv = Conv2D(size, ksize, activation=None, padding='same')(conv)
        conv = act_fn(0.2)(conv)
        tf.add_to_collection(ep_collection, conv)
        if pool:
            pool = MaxPooling2D(pool_size=pool_size)(conv)
            tf.add_to_collection(ep_collection, pool)
            return conv, pool
    return conv


def decoder_block(x, y, scope, size=None, upconv=True, ksize=(3, 3), upsize=(2, 2), upstirdes=(2, 2), act_fn='relu',
                  ep_collection='end_points', reuse=None, batch_norm=True, dropout=0.0):
    if size is None:
        base_size = x.get_shape().as_list()[-1]
        size = int(base_size / 2)
    with tf.variable_scope(scope, scope, [x], reuse=reuse) as sc:
        x = ThresholdedReLU(theta=0.0)(x)
        uped = Conv2DTranspose(size, upsize, strides=upstirdes, padding='same')(x) if upconv else x

        uped, y = reconcile_feature_size(uped, y)
        up = concatenate([uped, y], axis=3)
        tf.add_to_collection(ep_collection, up)

        conv = Conv2D(size, ksize, activation=act_fn, padding='same')(up)
        tf.add_to_collection(ep_collection, conv)

        conv = Conv2D(size, ksize, activation=act_fn, padding='same')(conv)
        tf.add_to_collection(ep_collection, conv)

        if batch_norm:
            conv = BatchNormalization()(conv, training=True)
            tf.add_to_collection(ep_collection, conv)
        if dropout > 0.0:
            conv = Dropout(dropout)(conv)
            tf.add_to_collection(ep_collection, conv)
    return conv


def reconcile_feature_size(x, y):
    x_shape = np.array(x.get_shape().as_list()[1:3], np.int32)
    y_shape = np.array(y.get_shape().as_list()[1:3], np.int32)
    x_pad_sizes = np.maximum(0, y_shape - x_shape)
    if np.any(x_pad_sizes > 0):
        x = tf.pad(x, paddings=[[0, 0], [0, x_pad_sizes[0]], [0, x_pad_sizes[1]], [0, 0]])
    y_pad_sizes = np.maximum(0, x_shape - y_shape)
    if np.any(y_pad_sizes > 0):
        y = tf.pad(y, paddings=[[0, 0], [0, y_pad_sizes[0]], [0, y_pad_sizes[1]], [0, 0]])
    return x, y


# def heatmap_loss(y, y_hat, weight_pos, weight_neg, reduction=tf.reduce_mean, name='heatmap_loss'):
#     with tf.name_scope(name):
#         unpacked_sigmoid_loss = weighted_sigmoid(predictions=y_hat, raw_labels=y, w_pos=weight_pos, w_neg=weight_neg)
#         cls_loss = reduction(unpacked_sigmoid_loss, name='unet_hmap_loss')
#     return cls_loss


def reconstruction_loss(y, y_hat, name='reconstruction_loss'):
    with tf.name_scope(name):
        rec_loss = weighted_l1(predictions=y_hat, images=y)
    return rec_loss


class UnetSmoother(TFNetwork):

    def __init__(self, out_size, size, down_layers, **kwargs):
        super(UnetSmoother, self).__init__(**kwargs)
        self.out_size = out_size
        self.down_layers = down_layers
        self.size = size
        self._calls = 0

        self.smooth_fmap = None
        self.upconvs = None

    def unet(self, hmap):
        reuse = True if self._calls > 0 else None
        self._calls += 1
        smooth_fmap, upconvs = unet_encdoer(hmap, size=self.size, down_layers=self.down_layers, scope='%s_enc' % self.scope, reuse=reuse)
        smooth_logits = unet_decoder(smooth_fmap, out_size=self.out_size, upconvs=upconvs, scope='%s_dec' % self.scope, reuse=reuse)
        return smooth_logits


if __name__ == '__main__':
    pass



