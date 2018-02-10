from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from lib.nn.losses import smooth_l1, weighted_loss_with_reshape
from models.specs import get_vars, TFNetwork


def regression_features(x, scope, L2_reg=0.0, reuse=None, train_mode=True, act_func=tf.nn.relu, **kwargs):
    def _args_scope():
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=act_func,
                            weights_regularizer=slim.l2_regularizer(L2_reg)
                            ):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    with slim.arg_scope(_args_scope()):
        with tf.variable_scope(scope, scope, [x], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):

                net = slim.conv2d(x, 32, [3, 3], stride=1, scope='conv1')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn1')
                net = slim.repeat(net, 3, slim.conv2d, 32, [3, 3], scope='conv11')

                net = slim.conv2d(net, 64, [3, 3], stride=2, padding='VALID', scope='conv2')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn2')
                net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv21')

                #x2
                net = slim.conv2d(net, 128, [3, 3], stride=2, padding='VALID', scope='conv3')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn3')
                net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv31')

                net = slim.conv2d(net, 128, [3, 3], stride=2, padding='VALID', scope='conv4')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn4')
                net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv41')

                # Here we go down to quarter
                net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn5')

                net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn6')

                net = slim.conv2d(net, 1024, [3, 3], scope='conv7')
                net = slim.batch_norm(net, is_training=train_mode, scope='bn7')

                # TODO: custom size image support - make padding dynamic
                net = tf.pad(net, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]], mode='CONSTANT', name='reg_features')

    return net


def box_shifts(x, scope, L2_reg=0.0, reuse=None, train_mode=True, act_func=tf.nn.relu, **kwargs):
    """
    1x1 convolution on the regression featuremap, to generate box_shifts predictions
    """
    def _args_scope():
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=act_func,
                            weights_regularizer=slim.l2_regularizer(L2_reg)
                            ):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    with slim.arg_scope(_args_scope()):
        with tf.variable_scope(scope, scope, [x], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                reg = slim.conv2d(x, 4, [1, 1], scope='reg_targets', activation_fn=None)

                pred_shape = reg.get_shape().as_list()
                new_shape = [pred_shape[0], pred_shape[1] * pred_shape[2], pred_shape[3]]
                reg_predictions_reshape = tf.reshape(reg, new_shape, name='reg_targets_reshape')

    return reg_predictions_reshape


def boxes_from_shifts_op(shifts, rel_points, inputs, scale_factor, scope, debug=False):
    """
    Transform relative points on a document to produce predicted boxes
    :param shifts: regression network output
    :param rel_points: relative points are anchor points spread uniformly on an image
    :param inputs: images tensor
    :param scale_factor: a parameter that shifts are scaled at during training
    :param scope:
    :param debug:
    :return:
    """
    with tf.variable_scope(scope, 'boxes_from_shifts', values=[shifts]):
        inp_shape = inputs.get_shape().as_list()[1:3][::-1]
        scale_tensor = tf.cast(tf.reshape(tf.concat([inp_shape, inp_shape], axis=0, name='scale_tensor'), [1, 1, 4]),
                               tf.float32)
        shifts = tf.div(shifts, scale_factor, name='scaled_shifts')

        x1 = tf.expand_dims(rel_points[:, :, 0] - shifts[:, :, 0], dim=-1, name='x1')
        x2 = tf.expand_dims(rel_points[:, :, 0] + shifts[:, :, 1], dim=-1, name='x2')

        y1 = tf.expand_dims(rel_points[:, :, 1] - shifts[:, :, 2], dim=-1, name='y1')
        y2 = tf.expand_dims(rel_points[:, :, 1] + shifts[:, :, 3], dim=-1, name='y2')

        boxes = tf.concat([y1, x1, y2, x2], axis=-1, name='predicted_boxes')
        output = boxes

        if debug:
            deb_output = locals()
            return output, deb_output

        return output


def add_batch_id_as_dim(boxes):
    """
    add batch id as extra dimension and squeeze tensor degree
    :param boxes: (Tensor) of degree 3 (batch_idx, num_boxes, 4) 
    :return: (Tensor) of degree 2 (num_boxes*batch_size, 5) - the first dimension is batch_idx \in [0, batch_size - 1]
    """
    box_shape = boxes.get_shape().as_list()
    batch_idx = np.floor(np.arange(box_shape[0] * box_shape[1]) / float(box_shape[1]))[:, np.newaxis]
    stacked_boxes = tf.reshape(boxes, [-1, 4])
    pred_boxes = tf.concat([tf.ones((box_shape[0] * box_shape[1], 1), tf.float32) * batch_idx, stacked_boxes], axis=-1, name='pred_boxes')
    return pred_boxes


def filter_boxes_on_size(boxes, target_size, min_side_len_pixels=8, scope='filter_boxes_on_size'):
    """
    Return boxes that are largers than a certain threshold area in pixels
    expects box format of (y1, x1, y2, x2)
    :param target_size: (tuple) Target (width, height)
    :param boxes: (Tensor) of degree 2 (num_boxes*batch_size, 5)
    :param min_side_len_pixels: (int) minimum side length
    :return: (Tensor) of degree 2 (?, 5) - filtered boxes
    """
    with tf.variable_scope(scope, 'filter_boxes_on_size', values=[boxes]):
        width = tf.subtract(boxes[:, 4], boxes[:, 2], name='boxes_width')
        height = tf.subtract(boxes[:, 3], boxes[:, 1], name='boxes_height')
        dx = tf.greater_equal(width, min_side_len_pixels / float(target_size[0]), name='t1')
        dy = tf.greater_equal(height, min_side_len_pixels / float(target_size[1]), name='t1')

        idx = tf.where(tf.logical_and(dx, dy), name='area_filter')[:, 0]
        selected_boxes = tf.gather(boxes, idx, name='selected_boxes')

    return selected_boxes


def reg_loss(y, y_hat, inside_box_flags, weight_pos, weight_neg, batch_size, scope):
    all_losses = []
    for i in range(batch_size):
        with tf.name_scope('%s/loss_b%d' % (scope, i), [y, y_hat]):

            # Find points that are well inside a word boundigng box
            flag_cond = tf.where(inside_box_flags >= 0)
            in_predictions = tf.squeeze(tf.gather_nd(y_hat, flag_cond), name='in_predictions')
            in_boxes = tf.squeeze(tf.gather_nd(y, flag_cond), name='in_boxes')
            in_labels = tf.squeeze(tf.gather_nd(inside_box_flags, flag_cond), name='in_labels')
            loss_vector = smooth_l1(in_predictions, in_boxes)
            w_loss_vector = weighted_loss_with_reshape(loss_vector, in_labels, w_pos=weight_pos, w_neg=weight_neg)
            b_regression_loss = tf.reduce_mean(w_loss_vector, name='b_reg_loss')
            all_losses.append(b_regression_loss)
    regression_loss = tf.reduce_mean(all_losses, name='reg_loss')
    # regression_loss = tf.div(tf.add_n(all_losses, name='sum_reg_loss'), batch_size, name='reg_loss')
    return regression_loss


class BoxRegression(TFNetwork):

    def __init__(self, **kwargs):
        super(BoxRegression, self).__init__(**kwargs)
        self._reg_calls = 0
        self._shift_calls = 0

    def features(self, x):
        reg_features = regression_features(x, scope=self.scope, L2_reg=self.args.box_reg_L2_reg, reuse=self.get_reuse(self._reg_calls))
        self._reg_calls += 1
        return reg_features

    def box_shifts(self, x):
        shifts = box_shifts(x, scope=self.scope, L2_reg=self.args.box_reg_L2_reg, reuse=self.get_reuse(self._shift_calls))
        self._shift_calls += 1
        return shifts

    def shifts_to_boxes(self, x, anchor_points, images):
        pred_boxes = boxes_from_shifts_op(x, anchor_points, images, scope='pred_boxes', scale_factor=self.args.regression_target_scaling_factor)
        pred_boxes_with_batch_idx = add_batch_id_as_dim(pred_boxes)
        return pred_boxes_with_batch_idx

    def filter_boxes_on_size(self, batched_boxes):
        return filter_boxes_on_size(batched_boxes, self.args.target_size)

