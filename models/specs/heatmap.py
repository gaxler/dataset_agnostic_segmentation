from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.keras.api.keras import backend as Kb

from models.specs import TFNetwork
from lib.nn.losses import weighted_xent_with_reshape, weighted_sigmoid
from models.specs import resnet_from_blocks, resnet_v2_block, get_vars

# def feature_map(x, scope,  build_phocs=False, arch='base', dropout=1.0, **kwargs):
#     if arch == 'base' and build_phocs:
#         blocks = [
#                 resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
#                 resnet_v2_block('block2', base_depth=128, num_units=3, stride=1),
#                 resnet_v2_block('block3', base_depth=256, num_units=3, stride=1),
#                 resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
#                 resnet_v2_block('block5', base_depth=512, num_units=3, stride=1),
#                 resnet_v2_block('block6', base_depth=512, num_units=3, stride=1),
#         ]
#     elif arch == 'small':
#         blocks = [
#             resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
#             resnet_v2_block('block2', base_depth=128, num_units=3, stride=1),
#             resnet_v2_block('block3', base_depth=256, num_units=3, stride=1),
#             resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
#             # resnet_v2_block('block5', base_depth=512, num_units=3, stride=1),
#             # resnet_v2_block('block6', base_depth=512, num_units=3, stride=1),
#         ]
#     else:
#         blocks = [
#             resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
#             resnet_v2_block('block2', base_depth=128, num_units=10, stride=1),
#             resnet_v2_block('block3', base_depth=256, num_units=3, stride=1)]
#
#     feature_map.scope = scope
#     net, end_points = resnet_from_blocks(x, blocks, scope=scope)
#     net = slim.dropout(net, keep_prob=dropout, scope='%s_dropout' % scope, is_training=Kb.learning_phase())
#     return net


def heatmap(x, scope, output_size=2, L2_reg=0.0, reuse=None, train_mode=True, linear_output=True, act_func=tf.nn.relu, **kwargs):
    def _args_scope():
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=act_func,
                            weights_regularizer=slim.l2_regularizer(L2_reg)):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    with slim.arg_scope(_args_scope()):
        with tf.variable_scope(scope, 'hmap', [x], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, slim.conv2d_transpose],
                                outputs_collections=end_points_collection):
                # Allow smoother downsampling of chanels
                net = slim.conv2d(x, 128, [3, 3], scope='conv1')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn1')
                net = slim.conv2d_transpose(net, 64, kernel_size=[3, 3], stride=[2, 2], scope='upconv2')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn2')
                net = slim.conv2d_transpose(net, 32, kernel_size=[3, 3], stride=[2, 2], scope='upconv3')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn3')

                trim_hmap = pad_features(net, size=16, scope='conv4')
                trim_hmap = slim.batch_norm(trim_hmap, is_training=train_mode, scope='bn4')
                trim_hmap = slim.conv2d(trim_hmap, 8, [3, 3], scope='conv5')
                trim_hmap = slim.batch_norm(trim_hmap, is_training=train_mode, scope='bn5')
                trim_hmap = slim.conv2d(trim_hmap, 4, [3, 3], scope='conv6')
                trim_hmap = slim.conv2d(trim_hmap, output_size, kernel_size=[3, 3], scope='conv7')
                if linear_output:
                    trim_hmap = slim.conv2d(trim_hmap, output_size, kernel_size=[1, 1], activation_fn=None, scope='conv8')

    return trim_hmap


def pad_features(inputs, size, scope, pad_x=4, pad_y=0, kernel=None, reuse=None):
    """
    Legacy, Kept for reproducibility
    """
    if kernel is None:
        kernel = [3, 3]
    with tf.variable_scope(scope, 'bottlneck', [inputs], reuse=reuse) as sc:
        bottle = slim.conv2d_transpose(inputs, size, kernel_size=kernel, stride=[2, 2], scope='upconv_bottleneck')
        bottle = tf.pad(bottle, paddings=[[0, 0], [0, 0], [0, pad_x], [0, pad_y]], mode='CONSTANT', name='pad_bottleneck')
    return bottle


def heatmap_loss_xent(y, y_hat, weight_pos, weight_neg, with_border=False):
    unpacked_xent_loss = weighted_xent_with_reshape(predictions=y_hat, raw_labels=y, w_pos=weight_pos, w_neg=weight_neg, with_border=with_border)
    cls_loss = tf.reduce_mean(unpacked_xent_loss, name='cls_loss')
    return cls_loss


def heatmap_loss_sigmoid(y, y_hat, weight_pos, weight_neg, reduction=tf.reduce_mean, name='heatmap_loss'):
    with tf.name_scope(name):
        unpacked_sigmoid_loss = weighted_sigmoid(predictions=y_hat, raw_labels=y, w_pos=weight_pos, w_neg=weight_neg)
        cls_loss = reduction(unpacked_sigmoid_loss, name='unet_hmap_loss')
    return cls_loss


class FeatureMap(TFNetwork):
    """
    This Network can generate two version of ResNet feature extractors (base and small).
    ResNet is constructed using tf.Slim
    """

    def __init__(self, **kwargs):
        super(FeatureMap, self).__init__(**kwargs)
        self.end_points = None

        self._calls = 0

    def base(self, x):
        blocks = [
            resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2_block('block2', base_depth=128, num_units=3, stride=1),
            resnet_v2_block('block3', base_depth=256, num_units=3, stride=1),
            resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
            resnet_v2_block('block5', base_depth=512, num_units=3, stride=1),
            resnet_v2_block('block6', base_depth=512, num_units=3, stride=1),
        ]

        return self._net(x, blocks)

    def small(self, x):
        blocks = [
            resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2_block('block2', base_depth=128, num_units=3, stride=1),
            resnet_v2_block('block3', base_depth=256, num_units=3, stride=1),
        ]

        return self._net(x, blocks)

    def _net(self, inputs, blocks):
        reuse = True if self._calls > 0 else None

        net, end_points = resnet_from_blocks(inputs, blocks, scope=self.scope, reuse=reuse)
        net = slim.dropout(net, keep_prob=(1-self.args.dropout), scope='%s_dropout' % self.scope, is_training=Kb.learning_phase())
        return net


class HeatMap(TFNetwork):

    def __init__(self, output_size, linear_output=True, **kwargs):
        super(HeatMap, self).__init__(**kwargs)
        self.linear_output = linear_output
        self.output_size = output_size
        self._calls = 0

    def heatmap(self, x):
        reuse = True if self._calls > 0 else None
        self._calls += 1
        return heatmap(x, scope=self.scope, output_size=self.output_size, L2_reg=self.args.heatmap_L2_reg, train_mode=True, linear_output=self.linear_output,
                       reuse=reuse)
