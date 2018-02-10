
import numpy as np

from lib.bbox import bbox_overlaps

from utils import get_coverage_mapping, save_stat_page
from lib.show_images import debugShowBoxes


def update_segmentation_stats(meta_images, doc_images, gt_boxes, params, pred_boxes, binary_icdar=True, viz=False, save_path=None, test_phase=False, untrim_icdar=0.1):
    """
    """
    batch_size = doc_images.shape[0]

    batch_imgs = []
    for n in range(batch_size):
        bidx = np.where(pred_boxes[:, 0] == n)[0]
        predictions = pred_boxes[bidx, 1:]
        meta = meta_images[n]
        gtidx = np.where(gt_boxes[:, 0] == n)[0]
        gt_boxes = gt_boxes[gtidx, 1:]

        # If ICDAR is being evaluated use fg pixel IoU calc and untrim boxes
        use_pixel_level = 'icdar' in str(meta.path) and binary_icdar
        predictions = untrim_boxes(predictions, trim=untrim_icdar) if use_pixel_level else predictions

        if test_phase and save_path is not None:
            pred_stats, pred_img = page_eval(doc_images[n, :], predictions, gt_boxes, use_pixel_level=use_pixel_level, output_all=params.output_all)
            save_stat_page(name=meta.path, pred_img=pred_img, pred_stat=pred_stats, save_path=save_path)

        if viz:
            viz_img = doc_images[n, :].copy()
            box_viz_img = debugShowBoxes(viz_img, boxes=predictions, gt_boxes=gt_boxes, wait=0, dont_show=True)
            batch_imgs.append(box_viz_img[np.newaxis, :])

    box_viz_img = np.vstack(batch_imgs) if len(batch_imgs) > 0 else None
    return box_viz_img


def page_eval(page_image, pred_boxes, gt_boxes, use_pixel_level=True, output_all=False):
    page_stats = {}
    overlaps = bbox_overlaps(gt_boxes.astype(np.float32), pred_boxes.astype(np.float32))
    gt_to_pred_map = get_coverage_mapping(overlaps.T)
    inv_page_binary = _inverse_binary(page_image, thresh=0.99)

    output_titles = []
    output_boxes = []
    # Check each gt_box
    for ind in range(gt_boxes.shape[0]):
        word_stats = {}
        gt_box = gt_boxes[ind, :]
        pred_ind = gt_to_pred_map.get(ind, None)
        if pred_ind is None:
            continue
        pred_box = pred_boxes[pred_ind, :]
        if use_pixel_level:
            o2o = pixel_iou(gt_box=gt_box, box=pred_box, binary_image=inv_page_binary)
        else:
            o2o = overlaps[ind, pred_ind]
        output_boxes.append(pred_box)
        output_titles.append('%4.3f' % o2o)
        word_stats['gt'] = gt_box.tolist()
        word_stats['pred'] = pred_box.tolist()
        word_stats['cover'] = o2o
        page_stats['word_%d' % ind] = word_stats

    if output_all:
        for ind in range(pred_boxes.shape[0]):
            pred_box = pred_boxes[ind, :]
            output_boxes.append(pred_box)
            output_titles.append('-')
            word_stats['pred'] = pred_box.tolist()
            page_stats['box_%d' % ind] = word_stats

    page_stats['predictions'] = pred_boxes.shape[0]
    page_stats['gt_boxes'] = gt_boxes.shape[0]
    preds_image = debugShowBoxes(page_image.copy(), boxes=output_boxes, gt_boxes=gt_boxes, titles=output_titles, dont_show=True)

    return page_stats, preds_image


def untrim_boxes(boxes, trim):
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


def train_viz(batch, rnd_boxes, rnd_labels, phoc_lab_thresh=0, unnormalize=None):
    rnd_images = []
    for i in range(batch['image'].shape[0]):
        idx = np.where(rnd_boxes[:, 0] == i)[0]
        batch_boxes = rnd_boxes[idx, :]
        gt_idx = np.where(batch['gt_boxes'][:, 0] == i)[0]
        batch_gt_boxes = batch['gt_boxes'][gt_idx, :]
        batch_labels = rnd_labels[idx]
        good_boxes = batch_boxes[np.where(batch_labels >= phoc_lab_thresh)[0], :]
        input_img = batch['image'].copy()[i, :]
        if unnormalize is not None:
            input_img = input_img*unnormalize
        rnd_image = debugShowBoxes(input_img, boxes=good_boxes[:, 1:], gt_boxes=batch_gt_boxes[:, 1:], dont_show=True)
        rnd_images.append(rnd_image[np.newaxis, :])
    return np.vstack(rnd_images)


def pixel_iou(gt_box, box, binary_image):
    gt_box = gt_box.astype(np.int32)
    # Number of black pixels in gt_box
    gt_pixels = np.sum(binary_image[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]])
    # Calculate intersection box between gt and prediction
    int_box = np.array(_intersection_box(gt_box, box), dtype=np.int32)
    # Number of black pixels in intersection box
    intersect_pixels = np.sum(binary_image[int_box[1]:int_box[3], int_box[0]:int_box[2]]).astype(np.float32)
    # rate of pixels covered by prediction
    o2o = intersect_pixels / gt_pixels
    return o2o


def _inverse_binary(binary_image, thresh=0.5, scale=255.):
    """ Turn binary picture so that black pixels will be valued 1. and white at 0."""
    x = np.sum(binary_image, axis=2)
    sliced_image = x / (scale*binary_image.shape[2])
    inv_binary = (np.ones_like(sliced_image) - sliced_image)
    inverse_img = (inv_binary > thresh)*1
    return inverse_img


def _intersection_box(box1, box2):
    return max(box1[0], box2[0]), \
           max(box1[1], box2[1]), \
           min(box1[2], box2[2]), \
           min(box1[3], box2[3])
