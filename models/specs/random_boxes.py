from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import tensorflow as tf

from lib.bbox import bbox_overlaps


def random_boxes_ops(gt_boxes, scope, num_classes, default_image_size, boxes_per_class, batch_size, gt_phocs=None, phoc_dim=0, iou_cls_lower_bound=None,
                     tf_format_in=True, tf_format_out=True):
    """
    Make random boxes generation ops with tf.pyfunc
    """
    if isinstance(boxes_per_class, int):
        boxes_per_class = [boxes_per_class]*num_classes
    assert isinstance(boxes_per_class, list), "Pass list of int as boxes_per_class"

    with tf.variable_scope('%s/random_boxes' % scope, values=[gt_boxes]):
        func = partial(random_rois, image_size=default_image_size, num_classes=num_classes, num_boxes_per_class=boxes_per_class, lower_bound=iou_cls_lower_bound,
                       tf_format_in=tf_format_in, tf_format_out=tf_format_out)

        # TODO: This fails if we can't find sufficient number of boxes. It seems pyfunc makes it difficult to get the actual dimensions of what's comming out of it.
        # TODO: refactor random_rois to be fully TF based
        random_boxes_shape = (batch_size * sum(boxes_per_class), 6)

        if gt_phocs is not None:
            rois_and_labels, phocs = tf.py_func(func, [gt_boxes, gt_phocs], [tf.float32, tf.float32])
            # Adding 1 (at the beginning) to represent batch
            phocs.set_shape((batch_size * sum(boxes_per_class), phoc_dim))
            phocs = tf.identity(phocs, name='rnd_phoc_assignment')
        else:
            rois_and_labels = tf.py_func(func, [gt_boxes], tf.float32, tf.float32)

        rois_and_labels.set_shape(random_boxes_shape)
        rois = tf.identity(rois_and_labels[:, :5], name='box_filter_rois')
        box_labels = tf.cast(rois_and_labels[:, -1], tf.int32, name='box_filter_labels')

        if gt_phocs is not None:
            return rois, box_labels, phocs

        return rois, box_labels, None


def random_rois(batch_gt_boxes, batch_embeddings=None, image_size=(900, 1200), num_classes=2, num_boxes_per_class=20,
                random_boxes_per_word=10, std_range=30, lower_bound=None,
                tf_format_in=False, tf_format_out=False):
    """

    :param batch_gt_boxes: (num_boxes, 5) => [batch_idx, x1, y1, x2, y2]
    :param batch_embeddings: (num_boxes, embedding_size+1) => e.g. [batch_id, v^T \in R^540]
    :param image_size:
    :param num_classes:
    :param num_boxes_per_class:
    :param lower_bound: (float) => lower bound of first IoU class
    :param tf_format_in:
    :param tf_format_out:

    :return:
    """
    assert batch_gt_boxes.shape[1] == 5 and batch_gt_boxes.ndim == 2, 'Pass gt_boxes in [batch, x,y,x,y] format'
    assert isinstance(num_classes, int) and num_classes > 1, 'Must have at least 2 classes'

    if tf_format_in:
        # Switch to abs boxes coordinates
        batch_gt_boxes = tf_format_to_abs(batch_gt_boxes, image_size)

    RANDS_PER_WORD = random_boxes_per_word
    RANDOM_STD_RANGE = std_range
    # gt_randoms = _generate_random_boxes_around_gt(gt_boxes, RANDS_PER_WORD, pixel_std=5)
    batch_size = batch_gt_boxes.astype(np.int32)[:, 0].max() + 1
    batch_rois = []
    assigned_embeddings = []
    for n in range(batch_size):
        bidx = np.where(batch_gt_boxes[:, 0] == n)[0]
        gt_boxes = batch_gt_boxes[bidx, 1:]
        gt_randoms = np.vstack([_generate_random_boxes_around_gt(gt_boxes, RANDS_PER_WORD, pixel_std=i + 0.5) for i in range(0, RANDOM_STD_RANGE, 2)])
        rand_randoms = _generate_random_relative_boxes(RANDS_PER_WORD * gt_boxes.shape[0]) * np.array(image_size * 2)
        rois = np.vstack((gt_randoms, rand_randoms)).astype(np.float32)
        # Clamp to image
        rois[:, ::2] = np.minimum(np.maximum(rois[:, ::2], 0), image_size[0])
        rois[:, 1::2] = np.minimum(np.maximum(rois[:, 1::2], 0), image_size[1])
        ovlps = bbox_overlaps(rois.astype(np.float32), gt_boxes.astype(np.float32))
        scores = ovlps.max(1).flatten()
        # NOTICE: the following assumes classes can be 5 or 2 by default, for any other num_classes you should set a lower_bound that makes sense
        lower_bound = (0.35 if num_classes == 5 else 0.2) if lower_bound is None else lower_bound
        class_bins = np.linspace(lower_bound, 1., num_classes + 1)[1:]
        func = partial(_box_scoring_helper, bins=class_bins)
        labels = np.array(map(func, scores))
        keep = _label_filter_picker_helper(num_classes=num_classes, labels=labels, num_boxes_per_class=num_boxes_per_class)
        rois = rois[keep, :]
        labels = labels[keep]
        rois = np.hstack((np.ones(rois.shape[0], np.float32)[:, np.newaxis]*n, rois, labels[:, np.newaxis])).astype(np.float32)
        batch_rois.append(rois)

        if batch_embeddings is not None:
            assigned_words = ovlps.argmax(1).flatten()
            embeddings = batch_embeddings[bidx, 1:]
            embeddings = embeddings[assigned_words, :]
            # Aligned embedding with randomly selected idx
            embeddings = embeddings[keep, :]
            assigned_embeddings.append(embeddings)

    batch_rois = np.vstack(batch_rois)

    if tf_format_out:
        new_rois = batch_rois[:, 1:-1][:, [1, 0, 3, 2]] / np.array(list(image_size) * 2)[::-1]
        batch_rois[:, 1:-1] = new_rois

    if batch_embeddings is not None:
        assigned_embeddings = np.vstack(assigned_embeddings)
        return batch_rois, assigned_embeddings
    return batch_rois


def _label_filter_picker_helper(num_classes, labels, num_boxes_per_class):
    """
    Randomly selects boxes
    num_boxes_per_class can be an int or an iterable. Gives control over class distribution during training
    """
    assert isinstance(num_classes, (int, tuple, list)), "num_boxes_per_class must be integer or iterable of ints"
    if isinstance(num_boxes_per_class, int):
        num_boxes_per_class = [num_boxes_per_class]*num_classes
    else:
        assert len(num_boxes_per_class) == num_classes, "num_boxes_per_class len must equal num_classes"
    output = []
    for indx in range(num_classes):
        valid_labels = np.where(labels == indx)[0]
        if valid_labels.shape[0] < 1:
            continue
        output.append(np.random.choice(valid_labels, size=num_boxes_per_class[indx], replace=True))
    output = np.concatenate(output)
    return output


def _box_scoring_helper(score, bins=None):
    if bins is None:
        bins = [0.2, 0.4, 0.6, 0.8, 1.]
    score = min(.99, score)
    i = 0
    while score > bins[i]:
        i += 1
    return i


def _generate_random_boxes_around_gt(gt_boxes, samples_around_each_box, pixel_std):
    flat_random_boxes = []
    for box in gt_boxes:
        mu = box[:4]
        z = np.random.randn(*((samples_around_each_box,) + mu.shape))
        sigma = np.ones_like(mu) * pixel_std
        random_boxes = mu + z * sigma
        flat_random_boxes.append(random_boxes)
    flat_random_boxes = np.vstack(flat_random_boxes)
    return flat_random_boxes


def _generate_random_relative_boxes(num_boxes):
    def random_point(size):
        x1 = np.random.rand(size)
        x2 = (1 - x1) * np.random.rand(size) + x1
        return np.vstack([x1, x2]).T

    random_boxes = np.zeros(shape=(num_boxes, 4))

    random_boxes[:, ::2] = random_point(num_boxes)
    random_boxes[:, 1::2] = random_point(num_boxes)
    return random_boxes.astype(np.float32)


def tf_format_to_abs(batch_boxes, image_size):
    """
    Turn tf formated relative boxes to absolute xyxy
    :param batch_boxes: boxes in [batch_id, rel_y1, rel_x1, rel_y2, rel_x2] or [rel_y1, rel_x1, rel_y2, rel_x2]
    :param image_size: (image_width, image_height)
    :return:
    """
    if batch_boxes.shape[-1] == 5:
        new_cords = batch_boxes[:, [0, 2, 1, 4, 3]]
        scale_vec = np.array([1] + list(image_size)*2)
    elif batch_boxes.shape[-1] == 4:
        new_cords = batch_boxes[:, [1, 0, 3, 2]]
        scale_vec = list(image_size) * 2
    else:
        raise ValueError('Pass 4 or 5 dim box coords')

    abs_cords = new_cords * scale_vec
    return abs_cords


def untrim_boxes(boxes, trim):
    """
    we trim boxes boxes during training. it's a good idea to scale them a bit to produce the final predictions
    :param boxes: nd.array (num_boxes, d) d \in {4,5}
    :param trim: (float)
    """
    box_dim = 1*(boxes.shape[1] == 5)
    new_boxes = []
    for i, box in enumerate(boxes):
            dx = int((box[box_dim + 2] - box[box_dim + 0]) * trim)
            box[box_dim + 0] -= dx
            box[box_dim + 2] += dx
            dy = int((box[box_dim + 3] - box[box_dim + 1]) * trim)
            box[box_dim + 1] -= dy
            box[box_dim + 3] += dy
            new_boxes.append(box)
    return np.vstack(new_boxes)


if __name__ == '__main__':
    gt_boxes = np.array([[0, 100, 200, 250, 300]])
    gt_phoc = np.hstack((np.array([[0]]), np.random.randint(0, 2, 540).reshape(1, -1)))
    tf_rois, phocs = random_rois(gt_boxes, batch_embeddings=gt_phoc, tf_format_out=True, num_classes=5)
    z = random_rois(tf_rois[:, :-1], tf_format_in=True, tf_format_out=False, num_classes=5)
    print ('/end')


