from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import cv2

from lib.show_images import debugShowBoxes


class BaseContoursHeatmap(object):

    cv_thresh = cv2.THRESH_BINARY
    cv_contour_method = cv2.CHAIN_APPROX_NONE
    contour_mode = cv2.RETR_TREE

    def __init__(self):
        pass

    def determenistic_boxes(self, orig, hmap, thresh=0.7, draw=False):
        dfunc = partial(self._deterministic_threshold, thresh=thresh)
        return self._base_get_bboxes(thresh_func=dfunc, orig=orig, hmap=hmap, draw=draw)

    def edge_boxes(self, orig, hmap, draw=False):
        return self._base_get_bboxes(thresh_func=self._edges_thresh, orig=orig, hmap=hmap, draw=draw)

    def _base_get_bboxes(self, thresh_func, orig, hmap, draw=False):
        o_shape = orig.shape
        h_shape = hmap.shape
        edges = thresh_func(hmap=hmap)
        conts = self._get_contours(threshed_hmap=edges)
        boxes = self._bboxes_from_contours(conts=conts)
        if boxes.shape[0] > 0:
            scales = [o_shape[0] / float(h_shape[0]), o_shape[1]/float(h_shape[1])]
            scales = np.array(scales+scales)
            boxes = boxes*scales
            if draw:
                debugShowBoxes(orig, boxes=boxes, wait=3000)
            return boxes
        return np.zeros(shape=(1, 4))
   

    def _deterministic_threshold(self, hmap, thresh=0.7, scale=255):
        hmap = (hmap*scale).astype(np.uint8)
        _, thresh = cv2.threshold(hmap, int(scale * thresh), scale, self.cv_thresh)
        return thresh

    def _edges_thresh(self, hmap, thresh=0.5, scale=255):
        hmap = (hmap * scale).astype(np.uint8)
        edges = cv2.Canny(hmap, scale*thresh, scale)
        return edges

    def _binomial_threshold(self, hmap):
        orig_shape = hmap.shape
        p = hmap.flatten()
        thresh = np.random.binomial(n=1, p=p).reshape(shape=orig_shape).astype(np.uint8)
        return thresh

    def _get_contours(self, threshed_hmap):
        # support diffrenet versio of cv2.findContours
        try:
            _, poly, _ = cv2.findContours(threshed_hmap, self.contour_mode, self.cv_contour_method)
        except:
            poly, _ = cv2.findContours(threshed_hmap, self.contour_mode, self.cv_contour_method)

        return poly

    def _bboxes_from_contours(self, conts):
        xywh = map(cv2.boundingRect, conts)
        xyxy = map(xywh_to_xyxy, xywh)
        return np.array(xyxy)


def xywh_to_xyxy(box):
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return np.array([box[0], box[1], x2, y2])
