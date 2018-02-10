"""
Various tensorflow utilities
"""

import tensorflow as tf


def weighted_sigmoid(predictions, raw_labels, w_pos, w_neg):
    with tf.name_scope('weighted_sigmoid'):
        lab_shape = tf.shape(raw_labels)
        w_pos = tf.fill(lab_shape, w_pos)
        w_neg = tf.fill(lab_shape, w_neg)
        loss_weights = tf.where(tf.greater(raw_labels, 0.), w_pos, w_neg)
        labels = tf.cast(raw_labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions, name='unpacked')
        loss = tf.multiply(loss_weights, loss, name='weighted_loss')
    return loss


def weighted_l1(predictions, images):
    with tf.name_scope('weighted_sigmoid'):
        loss = tf.losses.absolute_difference(images, predictions, weights=1.0, scope='weighted_loss')
    return loss



def weighted_xent_with_reshape(predictions, raw_labels, w_pos, w_neg, with_border=False):
    with tf.name_scope('weighted_xent_with_reshape'):
        raw_labels = tf.reshape(raw_labels, [-1])
        lab_shape = tf.shape(raw_labels)
        w_pos = tf.fill(lab_shape, w_pos)
        w_neg = tf.fill(lab_shape, w_neg)
        loss_weights = tf.where(tf.greater(raw_labels, 0.), w_pos, w_neg)
        logits = tf.reshape(predictions, [-1, 2]) if not with_border else tf.reshape(predictions, [-1, 3])
        labels = tf.cast(raw_labels, tf.int32)
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='unpacked_cls_loss')
        xent = tf.multiply(loss_weights, xent, name='weighted_loss')
    return xent


def smooth_l1(x, y):
    with tf.name_scope('smooth_l1'):
        diff = tf.subtract(x, y, name='diff')
        l1 = tf.abs(x - y, name='abs_diff')
        l1_smooth = tf.where(tf.greater(l1, 1), tf.subtract(l1, 0.5), tf.multiply(0.5, tf.square(diff)), name='loss')
    return l1_smooth


def weighted_loss_with_reshape(loss_vector, pos_neg_indicator, w_pos, w_neg):
    with tf.name_scope('weighted_loss_with_reshape'):
        pos_neg_indicator = tf.reshape(pos_neg_indicator, [-1])
        lv_shape = tf.shape(loss_vector)
        w_pos = tf.fill(lv_shape, w_pos)
        w_neg = tf.fill(lv_shape, w_neg)
        loss_weights = tf.where(tf.greater(pos_neg_indicator, 0.), w_pos, w_neg)
        weighted_loss_vector = tf.multiply(loss_weights, loss_vector, name='weighted_loss')
    return weighted_loss_vector