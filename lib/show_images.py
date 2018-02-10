from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2


def debugShowBoxes(debugimg, boxes=None, gt_boxes=None, wait=0, titles=None, save_path=None, dont_show=False):

    if boxes is None:
        boxes = []
    if gt_boxes is None:
        gt_boxes = []

    for i, bbox in enumerate(gt_boxes):
        cv2.rectangle(debugimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)

    for i, bbox in enumerate(boxes):
        cv2.rectangle(debugimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), thickness=1)
        if titles:
            t = str(titles[i]) if isinstance(titles, list) else titles
            fontFace = 0
            fontScale = .2 * (debugimg.shape[0] / 720.0)
            thickness = 1
            fg = (0, 120, 0)
            textSize, baseline = cv2.getTextSize(t, fontFace, fontScale, thickness)
            cv2.putText(debugimg, t, (int(bbox[0]), int(bbox[1] + textSize[1] + baseline / 2)),
                        fontFace, fontScale, fg, thickness)

    if not dont_show:
        if wait >= 0:
            if save_path is not None:
                save_path = str(save_path)
                cv2.imwrite(save_path, debugimg)
            else:
                cv2.imshow('debug', debugimg)
                cv2.waitKey(wait)
        else:
            cv2.imshow('debug', debugimg)
            cv2.waitKey()

    return debugimg