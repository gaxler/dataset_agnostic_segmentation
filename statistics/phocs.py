

import numpy as np

from lib.bbox import bbox_overlaps
from lib.phocs import phoc_letters_and_digits

from lib.show_images import debugShowBoxes

from utils import get_coverage_mapping, save_stat_page


def update_phoc_stats(meta_images, doc_images, gt_boxes, pred_boxes, pred_phocs, save_path):
    """"""
    # PHOC evaluation is supported for a single document batch
    meta_image = meta_images[0]
    doc_image = doc_images[0, :]

    good_idx, good_words = meta_image.get_good_words_and_boxes_idx()
    pred_stats, pred_img = phoc_eval_page(doc_image, pred_boxes[:, 1:], pred_phocs, gt_boxes[good_idx, 1:], good_words)
    save_stat_page(meta_image.path, pred_img=pred_img, pred_stat=pred_stats, save_path=save_path)

    pred_img_batch = pred_img[np.newaxis, :, :, :]

    return pred_img_batch


def phoc_eval_page(page_image, pred_boxes, pred_phocs, gt_boxes, gt_words):
    """

    :param page_image:
    :param pred_boxes:
    :param pred_phocs:
    :param gt_boxes:
    :param gt_words: (list) of str contating gt-words
    :param image_transform:
    :param o2o_score_func:
    :return:
    """
    page_stats = {}
    overlaps = bbox_overlaps(gt_boxes.astype(np.float32), pred_boxes.astype(np.float32))
    gt_to_pred_map = get_coverage_mapping(overlaps.T)
    pred_phocs = np.atleast_2d(pred_phocs)
    pred_boxes = np.atleast_2d(pred_boxes)

    output_titles = []
    output_boxes = []
    # Check each gt_box
    for ind in range(gt_boxes.shape[0]):
        word_stats = {}
        gt_box = gt_boxes[ind, :]
        gt_word = gt_words[ind]
        pred_ind = gt_to_pred_map.get(ind, None)

        word_stats['gt'] = gt_box.tolist()
        phocs, dim = phoc_letters_and_digits([gt_word])
        word_stats['gt_phoc'] = phocs[0, :].tolist()
        word_stats['text'] = gt_word

        if pred_ind is not None:
            pred_box = pred_boxes[pred_ind, :]
            pred_phoc = pred_phocs[pred_ind, :]
            o2o = overlaps[ind, pred_ind]
            output_boxes.append(pred_box)
            output_titles.append('%s[%d]' % (gt_word, o2o * 100))
            word_stats['pred'] = pred_box.tolist()
            word_stats['pre_phoc'] = pred_phoc.tolist()
            word_stats['cover'] = o2o
        page_stats['word_%d' % ind] = word_stats

    # Do stats for all un-assigned words
    for idx in set(range(pred_boxes.shape[0])) - set(gt_to_pred_map.values()):
        word_stats = {}
        best_gt_id = np.argmax(overlaps[:, idx])
        gt_box = gt_boxes[best_gt_id, :]
        gt_word = gt_words[best_gt_id]
        pred_ind = idx

        word_stats['gt'] = gt_box.tolist()
        phocs, dim = phoc_letters_and_digits([gt_word])
        word_stats['gt_phoc'] = phocs[0, :].tolist()
        word_stats['text'] = gt_word

        if pred_ind is not None:
            pred_box = pred_boxes[pred_ind, :]
            pred_phoc = pred_phocs[pred_ind, :]
            o2o = overlaps[ind, pred_ind]
            output_boxes.append(pred_box)
            output_titles.append('%s[%d]' % (gt_word, o2o*100))
            word_stats['pred'] = pred_box.tolist()
            word_stats['pre_phoc'] = pred_phoc.tolist()
            word_stats['cover'] = o2o
        page_stats['word_red_%d' % idx] = word_stats

    page_stats['predictions'] = pred_boxes.shape[0]
    page_stats['gt_boxes'] = gt_boxes.shape[0]

    preds_image = debugShowBoxes(page_image.copy(), boxes=output_boxes, gt_boxes=gt_boxes, titles=output_titles, dont_show=True)

    return page_stats, preds_image


