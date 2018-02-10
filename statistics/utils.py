
from functools import partial
import json
from pathlib2 import Path

import numpy as np
import cv2


def save_stat_page(name, pred_stat, pred_img, save_path):
    name = Path(name).stem
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    stat_path = save_path / ('%s.json' % name)
    img_path = save_path / ('%s.png' % name)
    json.dump(pred_stat, stat_path.open('wb'))
    cv2.imwrite(str(img_path), pred_img)
    return


def get_coverage_mapping(overlaps):
    """
    Find best coverage of gt_boxes by predicted boxes
    :param overlaps: assumes (boxes, gt_boxes)
    :return:
    """
    rows = overlaps.shape[0]
    cols = overlaps.shape[1]
    t = partial(_1d_to_2d, fast_shift=cols)
    flat_overlaps = overlaps.flatten()
    indx_best_fits = np.argsort(flat_overlaps)[::-1]
    coverage_map = {}
    row_bucket = np.zeros(rows)
    col_bucket = np.zeros(cols)
    for ind in indx_best_fits:
        row, col = t(ind)
        if row_bucket[row] == 0 and col_bucket[col] == 0:
            row_bucket[row] = 1
            col_bucket[col] = 1
            coverage_map[col] = row
    return coverage_map


def _1d_to_2d(idx, fast_shift):
    slow = np.floor(idx / fast_shift).astype(np.int32)
    fast = idx % fast_shift
    return slow, fast
