from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from lib.phocs import phoc_letters_and_digits


def from_image_to_heatmap(image, gt_boxes, meta_image, names, trim=0., abs_size=10, with_border=False, **kwargs):
    """
    Draw heatmap from bounding boxes.
    :param image:
    :param gt_boxes: (num_boxes, 4) -> (x1,y1,x2,y2)
    :param meta_image: MetaImage class
    :param name: name to added as key in pipeline output dict
    :param trim: percentage of trimming from each side of the image
    :param abs_size: minimum size of which triming won't be performed
    :param with_border: If True, mark trimmed area as periphery class. (bkg=0, periphery=1, fg=2)
    :return:
            name: (str)
            t_map (nd.array) (im_h, im_w, 1)
    """
    image = image.copy()
    gt_boxes = gt_boxes.copy()
    height, width = image.shape[:2]

    # Currently only supports full image heatmap
    scale = 1.

    t_map = np.zeros((height, width))
    # Trim protection
    for i, box in enumerate(gt_boxes):
        if with_border:
            box = np.array(box).astype(np.int32)
            t_map[box[1]:box[3], box[0]:box[2]] = 1
        if np.diff(box[::2]) > abs_size:
            dx = int((box[2] - box[0]) * trim)
            box[0] += dx
            box[2] -= dx
        if np.diff(box[1::2]) > abs_size:
            dy = int((box[3] - box[1]) * trim)
            box[1] += dy
            box[3] -= dy
        cords = np.array(box / float(scale)).astype(np.int32)

        t_map[cords[1]:cords[3], cords[0]:cords[2]] = 2 if with_border else 1

    t_map = t_map[:, :, np.newaxis]
    return names, t_map


def regression_bbox_targets(image, gt_boxes, meta_image, names, fmap_w, fmap_h, **kwargs):
    """
    We wish to assign a [width_left, width_right, height_left, height_right] coordinates to each
    point in the feature map. Feature map point is defined by its center coordinate.
    If a center coordinate is outside of a gt_box than we assign a zero box target to it (box with w,h of 0)
    If the center is in more than one box we will ignore the point (no need in fuzzy gradients messing out updates)
    Output:
        An np.array of targets (fmap_w*fmap_h, 4)
        An np.array of labels (fmap_w*fmap_h, ) where 1 - gt_box, 0 - no box, -1 - ignore this point
    """
    image = image.copy()
    gt_boxes = gt_boxes.copy()
    t_w = image.shape[1]
    t_h = image.shape[0]
    pixels_per_fmap = t_w / fmap_w

    # Save computations with a static var. Assumes image dim and feature map dim remians constant after init
    try:
        relative_pts = regression_bbox_targets.relative_points
    except:
        sh_x, sh_y = np.meshgrid(np.arange(fmap_w), np.arange(fmap_h))
        pts = np.vstack((sh_x.ravel(), sh_y.ravel())).transpose()
        cntr_pts = pts + np.array([0.5]*2, np.float32)[np.newaxis, :]
        relative_pts = cntr_pts / np.array([fmap_w, fmap_h], np.float32)[np.newaxis, :]
        regression_bbox_targets.relative_points = relative_pts

    relative_boxes = gt_boxes  # / np.array([t_w, t_h]*2, np.float32)[np.newaxis, :]

    assignment_map = np.zeros((relative_pts.shape[0], gt_boxes.shape[0]), np.float32)
    target = np.zeros((relative_pts.shape[0], 4), np.float32)

    for bnum, box in enumerate(gt_boxes):
        adj_box = (box / pixels_per_fmap).astype(np.int32)
        sh_x, sh_y = np.meshgrid(range(max(0, adj_box[0] - 2), min(fmap_w, adj_box[2] + 2)),
                                 range(max(0, adj_box[1] - 2), min(fmap_h, adj_box[3] + 2)))
        idx = [i + fmap_w * j for i, j in zip(sh_x.ravel(), sh_y.ravel())]
        run_pts = relative_pts[idx, :]
        run_pts = run_pts * [t_w, t_h]
        for pt_idx, p in enumerate(run_pts):
            rel_box = relative_boxes[bnum, :]
            assignment_map[idx[pt_idx], bnum] = _point_in_box(p, rel_box)
            # if self._point_in_box(p, rel_box) > 0:
            #     print ('sd')
            target[idx[pt_idx], :] = [p[0] - rel_box[0], rel_box[2] - p[0], p[1] - rel_box[1], rel_box[3] - p[1]]

    # Points we are intrested in
    relevant_points = np.ones(assignment_map.shape[0])
    # if more than 1 gt_box is assigned - ignore this point (intersection are bad points anyway)
    relevant_points[np.where(assignment_map.sum(axis=1) > 1)] = -1
    # if some of the coords are negative. ignore point
    relevant_points[np.where(target.min(axis=1) < 0)] = -1
    # if no gt_box is assigned than this point should get the zero target
    relevant_points[np.where(assignment_map.sum(axis=1) < 1)] = 0
    relevant_points = relevant_points[:, np.newaxis]

    target *= (relevant_points > 0) * 1
    target /= np.array([t_w, t_w, t_h, t_h])[np.newaxis, :]

    relevant_points = relevant_points.ravel()
    return names, (target, relevant_points)


def _point_in_box(point, box):
    """
    :param point: np.array([x,y])
    :param box: np.array([x1,y1, x2, y2])
    :return: Bool
    """
    truth_list = [(point[i] <= box[i::2][1]) & (point[i] >= box[i::2][0]) for i in range(2)]
    return int(all(truth_list))


def phoc_embedding(image, gt_boxes, meta_image, names=('phocs', 'tf_format_gt_boxes'), **kwargs):
    """
    Build PHOC embedding out of proposed boxes.
    In case we know about bad segmentation, we ingore those boxes and hence return new filtered gt_boxes
    """
    image = image.copy()
    gt_boxes = gt_boxes.copy()
    filtered_gt_boxes, phocs, _ = _phocs_from_metaimage(meta_image, gt_boxes)
    # In case we have no good boxes in the page just return None. pipeline will skip it
    if filtered_gt_boxes is None or phocs is None:
        return names, None

    image_size = image.shape[:2]
    tf_format_gt_boxes = filtered_gt_boxes[:, [1, 0, 3, 2]] / np.array(list(image_size) * 2)

    return names, (phocs, tf_format_gt_boxes)


def tf_boxes(image, gt_boxes, meta_image, names=('tf_format_gt_boxes', ), **kwargs):
    """
    Build PHOC embedding out of proposed boxes.
    In case we know about bad segmentation, we ingore those boxes and hence return new filtered gt_boxes
    """
    image = image.copy()
    gt_boxes = gt_boxes.copy()
    image_size = image.shape[:2]
    tf_format_gt_boxes = gt_boxes[:, [1, 0, 3, 2]] / np.array(list(image_size) * 2)
    return names, (tf_format_gt_boxes, )


def _phocs_from_metaimage(meta_image, augmented_gt_boxes):
    """ Take a meta-image, encodes words to PHOC and returns gt_boxes with good segmentation"""
    boxes_idx, words = meta_image.get_good_words_and_boxes_idx()
    if boxes_idx is not None and words is not None:
        gt_boxes = augmented_gt_boxes[boxes_idx, :]
        phocs, phoc_dim = phoc_letters_and_digits(words)
        return gt_boxes, phocs, phoc_dim
    _, phoc_dim = phoc_letters_and_digits([])
    return None, None, phoc_dim





