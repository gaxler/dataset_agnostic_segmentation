from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from numba import jit, uint8, float32, int32

from lib.show_images import debugShowBoxes


class AugmentationBase(object):

    def __init__(self, target_width, target_height, apply_prob=1.0, debug=False):
        self.apply_prob = apply_prob
        self._tw = target_width
        self._th = target_height

        self._debug = debug

    def apply(self, image, boxes, meta_image):
        raise NotImplementedError('Inherit and implement')

    def image_resize(self, image, boxes, target_x=None, target_y=None):

        target_x = self._tw if target_x is None else target_x
        target_y = self._th if target_y is None else target_y

        if float(image.shape[0]) / float(image.shape[1]) < target_y / target_x:
            f = float(target_x) / image.shape[1]
            dsize = (target_x, int(image.shape[0] * f))
        else:
            f = float(target_y) / image.shape[0]
            dsize = (int(image.shape[1] * f), target_y)

        image = cv2.resize(image, dsize=dsize)

        scaled_boxes = boxes * np.atleast_2d(np.array([f, f, f, f]))

        if self._debug:
            pass

        return image, scaled_boxes

    def border_replicate(self, image, boxes, border_type=cv2.BORDER_REPLICATE):
        """
        :param border_type: cv2 borderType 
        :return: image of target_size
        """
        target_x = self._tw
        target_y = self._th
        resized_image = cv2.copyMakeBorder(image,
                                           top=0,
                                           left=0,
                                           right=target_x - image.shape[1],
                                           bottom=target_y - image.shape[0],
                                           borderType=border_type)

        return resized_image, boxes



class PartilPage(AugmentationBase):
    """ 
    Crop part of the image and only keep boxes that are fully inside the cropped region
    """

    def apply(self, image, boxes, meta_image):
        if np.random.rand() > self.apply_prob:
            return image, boxes, meta_image
        idx = np.array([])
        count = 0
        # try five times
        while not idx.size and count < 5:
            count += 1
            idx, z = self.box_filter(boxes, image.shape, (2*self._th, 2*self._tw))
            idx = np.array(idx)
        # if failed just skip the cropping
        if idx.size < 5 or max(z) < 1:
            return image, boxes, meta_image
        random_img = image[z[1]:z[3], z[0]:z[2], :]
        new_boxes = boxes[idx, :]
        new_boxes = adjust_boxes(new_boxes, (z[0], z[1]))
        meta_image._old_boxes = boxes
        meta_image.bboxes = new_boxes
        if len(meta_image.words) > 0:
            meta_image._old_words = meta_image.words
            meta_image.words = [meta_image.words[ind] for ind in idx]
        return random_img, new_boxes, meta_image

    @staticmethod
    def sq_size(limit, size):
        dy = int(size[0] / 2)
        dx = int(size[1] / 2)
        y = np.random.randint(dy, max(limit[0] - dy, dy+1))
        x = np.random.randint(dx, max(limit[1] - dx, dx+1))
        return x - dx, y - dy, x + dx, y + dy

    @staticmethod
    def box_filter(boxes, limits, size, border=20):
        z = PartilPage.sq_size(limits, size)
        # Pick boxes that fall inside new image boundaries
        good_idx = np.where((boxes[:, 0] > z[0]) & (boxes[:, 1] > z[1]) & (boxes[:, 2] < z[2]) & (boxes[:, 3] < z[3]))[0]
        # If boundaries are empty...
        if good_idx.shape[0] < 1:
            return [], (0, 0, 0, 0)
        good_boxes = boxes[good_idx, :]
        limits_of_good_boxes = np.concatenate((good_boxes[:, :2].min(0), good_boxes[:, 2:].max(0)))

        new_z = np.array([max(limits_of_good_boxes[0] - border, 0), max(limits_of_good_boxes[1] - border, 0),
                          min(limits_of_good_boxes[2] + border, limits[1]), min(limits_of_good_boxes[3] + border, limits[0])])\
            .astype(np.int32)
        return good_idx, new_z

@jit
def adjust_boxes(boxes, start_point):
    new_boxes = boxes - np.array(start_point*2)[np.newaxis, :]
    return new_boxes


class Resize(AugmentationBase):
    """ 
    Plain resize - keeps aspect ration and uses border replication to fit target image size.
    If image size is same as target size this augmentation will produce identity
    """

    def apply(self, image, boxes, meta_image):
        new_image, new_boxes = self.image_resize(image, boxes)
        final_image, final_boxes = self.border_replicate(new_image, new_boxes)
        return final_image, final_boxes, meta_image


class Slant(AugmentationBase):
    def __init__(self, apply_prob=0.3, slant_prob=0.5, **kwargs):
        super(Slant, self).__init__(**kwargs)
        self.slant_prob = slant_prob
        self._apply_prob = apply_prob

    def apply(self, image, boxes, meta_image):
        if np.random.rand() < self._apply_prob:
            aug_img = self.augment_boxes(image, boxes.astype(np.int32), self.slant_prob)
            return aug_img, boxes, meta_image
        return image, boxes, meta_image

    @staticmethod
    # @jit(uint8(uint8, int32, float32), nogil=True)
    def augment_boxes(image, boxes, prob):
        for b in boxes:
            img = image[b[1]:b[3], b[0]:b[2], :]
            p = np.random.rand()
            # if p > prob:
            augmented = Slant.img_aug(img[:, :, 0])
            augmented = expand(augmented)
            # else:
            #     augmented = img
            image[b[1]:b[3], b[0]:b[2]] = augmented
        return image

    @staticmethod
    # @jit(uint8(uint8), nogil=True)
    def img_aug(img):
        h, w = img.shape
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img


class DialationErosio(AugmentationBase):
    """
    Dialate or erode word images
    """
    def __init__(self, dialation_prob=0.5, skip_gray_prob=0.0, apply_prob=0.3, **kwargs):
        super(DialationErosio, self).__init__(**kwargs)
        self.skip_gray_prob = skip_gray_prob
        self.apply_prob = apply_prob
        self.dialation_prob = dialation_prob

    def apply(self, image, boxes, meta_image):
        if np.random.rand() < self.skip_gray_prob:
            return image, boxes, meta_image
        # tic = time.time()
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # toc1 = time.time() - tic
        # print ('Gray Scale: %4.3f' % toc1)
        if np.random.rand() > self.apply_prob:
            gray_scale = expand(gray_scale)
            return gray_scale, boxes, meta_image
        kernel = np.ones((5, 5), np.uint8)
        gray_scale = augment_boxes(gray_scale, boxes.astype(np.int32), kernel, self.dialation_prob)
        # toc2 = time.time() - tic - toc1
        # print('Agument: %4.3f' % toc2)
        # gray_scale = np.repeat(gray_scale[:,:, np.newaxis], 3, axis=2)
        gray_scale = expand(gray_scale)
        q = 0.4 + 0.2*np.random.rand()
        gray_scale = np.array(q*gray_scale + (1-q)*image, dtype=np.uint8)
        # print('expand: %4.3f' % toc3)
        return gray_scale, boxes, meta_image

@jit(uint8(uint8, int32, uint8, float32), nogil=True)
def augment_boxes(gray_scale, boxes, kernel, prob):
    for b in boxes:
        img = gray_scale[b[1]:b[3], b[0]:b[2]]
        p = np.random.rand()
        if p < prob and img.shape[0] > 1 and img.shape[1] > 1:
        # if img.shape[0] > 1 and img.shape[1] > 1:
            augmented = cv2.dilate(img, kernel=kernel, iterations=1)
        else:
            augmented = img
        gray_scale[b[1]:b[3], b[0]:b[2]] = augmented
    return gray_scale

@jit(uint8(uint8), nogil=True)
def expand(gray_scale):
    gray_scale = np.repeat(gray_scale[:, :, np.newaxis], 3, axis=2)
    return gray_scale


class KeepWordsOnly(AugmentationBase):
    """
    Embed a small version of the image inside a black box of target size
    This should be helpful for learning to identify small scripts
    """

    def __init__(self, target_width, target_height, debug=False, **kwargs):
        super(KeepWordsOnly, self).__init__(target_width, target_height, debug)
        self._low_bound, self._high_bound = kwargs.get('ratio_bounds', (0.2, 0.7))
        self._prob = kwargs.get('prob', 1.)

    def apply(self, image, boxes, meta_image):
        mins = boxes.min(0).astype(np.int32) - 5
        maxs = boxes.max(0).astype(np.int32) + 5

        new_image = image[mins[1]:maxs[3], mins[0]:maxs[2], :]
        if new_image.shape[0] < 10 or new_image.shape[1] < 10:
            black_image = np.zeros(shape=image.shape, dtype=image.dtype)
            black_image[mins[1]:maxs[3], mins[0]:maxs[2], :] = image[mins[1]:maxs[3], mins[0]:maxs[2], :]
            return black_image, boxes
        new_boxes = boxes - np.array([mins[0], mins[1], mins[0], mins[1]])
        image = None
        return new_image, new_boxes, meta_image


class ImageEmbed(AugmentationBase):
    """
    Embed a small version of the image inside a black box of target size
    This should be helpful for learning to identify small scripts
    """

    def __init__(self, target_width, target_height, debug=False, **kwargs):
        super(ImageEmbed, self).__init__(target_width, target_height, debug)
        self._low_bound, self._high_bound = kwargs.get('ratio_bounds', (0.2, 0.7))
        self._prob = kwargs.get('prob', 1.)

    def apply(self, image, boxes, meta_image):
        if np.random.rand() >= self._prob:
            return image, boxes, meta_image
        black_image = np.zeros(shape=(self._th, self._tw, 3), dtype=image.dtype)
        tw, th, x0, y0 = self.pick_random_size()
        scale_image, scaled_boxes = self.image_resize(image, boxes, target_x=tw, target_y=th)
        black_image[y0:(y0 + scale_image.shape[0]), x0:(x0 + scale_image.shape[1]), :] = scale_image
        translated_boxes = scaled_boxes + [x0, y0, x0, y0]
        return black_image, translated_boxes, meta_image

    def pick_random_size(self):
        ratio = self._low_bound + (self._high_bound - self._low_bound) * np.random.rand()
        tw = np.int32(ratio * self._tw)
        th = np.int32(ratio * self._th)

        x0 = np.random.randint(0, self._tw)
        y0 = np.random.randint(0, self._th)
        while not (x0 + tw < self._tw and y0 + th < self._th):
            x0 = np.random.randint(0, self._tw)
            y0 = np.random.randint(0, self._th)

        return tw, th, x0, y0

class GaussianNoise(AugmentationBase):

    def __init__(self, target_width, target_height, debug=False, **kwargs):
        super(GaussianNoise, self).__init__(target_width, target_height, debug)
        # Noise apply probability
        self._prob = kwargs.get('prob', 0.5)

    def apply(self, image, boxes, meta_image):
        # Noise apply probability
        if np.random.rand() >= self._prob:
            return image, boxes, meta_image
        new_img = image * np.random.normal(np.ones(image.shape[:-1] + (1, )), 0.15, image.shape[:-1] + (1, ))
        return new_img, boxes, meta_image


class BoxRearange(AugmentationBase):

    @staticmethod
    def box_mover(box, start_point=(0, 0)):
        box = np.array(box)
        box[2] -= box[0] - start_point[0]
        box[0] -= box[0] - start_point[0]
        box[3] -= box[1] - start_point[1]
        box[1] -= box[1] - start_point[1]
        return box

    def apply(self, image, boxes, meta_image):
        new_format = WordAranger(image, boxes, meta_image)
        return new_format.get_new_page()


def split(box):
    if np.random.rand() < 0.5:
        s = np.random.randint(box[0], box[2])
        box1 = [box[0], box[1], s, box[3]]
        box2 = [s, box[1], box[2], box[3]]
    else:
        s = np.random.randint(box[1], box[3])
        box1 = [box[0], box[1], box[2], s]
        box2 = [box[0], s, box[2], box[3]]
    return box1, box2


class ImageSplitter(object):
    def __init__(self, image_shape, max_h, max_w, max_splits=1):
        self.max_w = max_w
        self.max_h = max_h
        self.max_splits = max_splits
        self.h = image_shape[0]
        self.w = image_shape[1]
        self._boxes = [(0, 0, self.w, self.h)]
        self._splits = 0

    def get_next_box(self):
        box = self._boxes.pop(0)
        if self._splits < self.max_splits and (box[2] - box[0] > 2*self.max_w and box[3] - box[1] > 2*self.max_h):
            self._splits += 1
            boxes = self.split(box)
            for b in boxes:
                self._boxes.append(b)
            return self.get_next_box()
        init_x = box[0]
        init_y = box[1]
        final_x = box[2]
        final_y = box[3]
        line_arrays = init_y + np.cumsum(np.random.randint(self.max_h, self.max_h + 10, int((final_y - init_y) / self.max_h) - 1))
        return init_x, final_x, init_y, final_y, line_arrays

    def split(self, box):
        s = 0
        return_boxes = []
        indicator = (np.random.rand() < 0.5) #and box[2] - box[0] > 2*self.max_w) or (box[2] - box[0] > 2*self.max_w and box[3] - box[1] < 2*self.max_h)
        if indicator:
            while len(return_boxes) < 1:
                s = np.random.randint(box[0] + self.max_w, box[2] - self.max_w)
                box1 = (box[0], box[1], s, box[3])
                box2 = (s, box[1], box[2], box[3])
                if box1[2] - box1[0] > self.max_w:
                    return_boxes.append(box1)
                if box2[2] - box2[0] > self.max_w:
                    return_boxes.append(box2)
        else:
            while s - box[1] < self.max_h and box[3] - s < self.max_h:
                s = np.random.randint(box[1] + self.max_h, box[3] - self.max_h)
                box1 = (box[0], box[1], box[2], s)
                box2 = (box[0], s, box[2], box[3])
                if box1[3] - box1[1] > self.max_h:
                    return_boxes.append(box1)
                if box2[3] - box2[1] > self.max_h:
                    return_boxes.append(box2)

        return return_boxes


class WordAranger(object):

    def __init__(self, image, boxes, meta_image, fill_meta_images=None):
        self.fill_meta_images = fill_meta_images
        self.meta_image = meta_image
        self.boxes = np.array(boxes).tolist()
        self._permutation = np.random.permutation(range(len(self.boxes))).tolist()
        self.image = image
        self._new_word_list = []
        self._new_bboxes = []

        self.w = image.shape[1]
        self.h = image.shape[0]
        self.empty = self.image is None

        if self.empty:
            dh = np.random.randint(1200, 4500)
            self.image = np.ones((dh, int(dh/1.33), 3))*128.
        self.canvas = self._get_canvas()

    def get_new_page(self):
        self.fill_page()
        self.meta_image.words = self._new_word_list
        bboxes = np.array(self._new_bboxes)
        self.meta_image.bboxes = bboxes
        return self.canvas.astype(np.uint8), bboxes, self.meta_image

    def fill_page(self):
        zero_point_x = np.random.randint(5, 100)
        zero_point_y = np.random.randint(5, 100)
        end = [self.w, zero_point_y]

        abs_max_h = 0

        while end[1] < self.h - abs_max_h:
            strt = [zero_point_x, end[1]]
            line_gap = np.random.randint(5, 50)
            max_h = self.fill_line(strt, end)
            if max_h == strt[1]:
                break
            abs_max_h = max(max_h - end[1], abs_max_h)
            end[1] = max_h + line_gap
            strt[1] = max_h + line_gap

    def _get_canvas(self):
        img = self.image
        dx = np.mean(img, axis=(0, 1))[np.newaxis, np.newaxis, :]
        dy = np.ones(img.shape, dtype=np.uint8) * 227
        dy[:, :, :] = dx
        dy = dy + 0.02 * dy * np.random.randn(*dy.shape[:-1])[:, :, np.newaxis]
        return dy

    def fill_canvas(self, new_box, old_box):
        img = self.image
        dy = self.canvas
        new_box = np.array(new_box, dtype=np.int32)
        old_box = np.array(old_box, dtype=np.int32)
        gamma = np.random.beta(1.1, 0.1)
        dy[new_box[1]:new_box[3], new_box[0]:new_box[2], :] = (gamma)*img[old_box[1]:old_box[3], old_box[0]:old_box[2], :] + \
                                                              (1-gamma)*dy[new_box[1]:new_box[3], new_box[0]:new_box[2], :]

    def fill_line(self, strt, end):
        gappines = np.random.randint(20, 150)
        box, word_dict = self.get_next_word()
        max_h = strt[1]

        if box is None or word_dict is None:
            return end[1]

        while strt[0] + box[2] - box[0] < end[0]:
            new_box = WordAranger.box_mover(box, strt)
            gap = np.random.randint(5, gappines)
            strt[0] += (new_box[2] - new_box[0] + gap)
            max_h = max(new_box[3], max_h)

            word_dict['box'] = new_box.tolist()
            self._new_word_list.append(word_dict)
            self._new_bboxes.append(new_box)

            self.fill_canvas(new_box, box)

            box, word_dict = self.get_next_word()
            if box is None and word_dict is None:
                break

        if box is not None and word_dict is not None:
            self.boxes.append(box)
            self.meta_image.words.append(word_dict)
        return max_h

    def get_next_word(self):
        if len(self._permutation):
            idx = self._permutation.pop()
            box = self.boxes[idx]
            word_dict = self.meta_image.words[idx]
            return box, word_dict
        return None, None

    @staticmethod
    def box_mover(box, start_point=(0, 0)):
        box = np.array(box)
        box[2] -= box[0] - start_point[0]
        box[0] -= box[0] - start_point[0]
        box[3] -= box[1] - start_point[1]
        box[1] -= box[1] - start_point[1]
        return box



# @jit
def reorder_boxes(image, boxes):
    dx = np.mean(image, axis=(0, 1))[np.newaxis, np.newaxis, :]
    dy = np.ones(image.shape, dtype=np.uint8) * 227
    dy[:, :, :] = dx
    dy = dy + 0.02 * dy * np.random.randn(*dy.shape[:-1])[:, :, np.newaxis]
    mw = np.diff(boxes[:, ::2]).max()
    mh = np.diff(boxes[:, 1::2]).max()
    splitter = ImageSplitter(image.shape, mh, mw, max_splits=1)
    init_x, final_x, init_y, final_y, line_arrays = splitter.get_next_box()
    next_x = init_x + 5
    j = 0
    gamma = 0.95
    new_boxes = []
    for n, b in enumerate(boxes):
        box_img = image[b[1]:b[3], b[0]:b[2], :].astype(np.uint8)
        line_limit = next_x + (b[2] - b[0])
        while line_limit > final_x or final_y < line_arrays[j] + (b[3] - b[1]):
            j = j + 1
            if j > len(line_arrays) - 1:
                init_x, final_x, init_y, final_y, line_arrays = splitter.get_next_box()
                j = 0
            next_x = init_x + 5
            line_limit = next_x + (b[2] - b[0])

        new_box = BoxRearange.box_mover(b, (next_x, line_arrays[j] + rnd_shift()))
        new_boxes.append(new_box)
        dy[new_box[1]:new_box[3], new_box[0]:new_box[2], :] = gamma * box_img + (1 - gamma) * dy[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        next_x = new_box[2] + max(rnd_shift(), 0)

    return dy, np.array(new_boxes)


def rnd_shift(phi=5):
    return np.random.randint(-phi, phi)


def test_augmentations():
    import data
    from data.simple_pipe import PipelineBase
    from data.data_extenders import phoc_embedding, from_image_to_heatmap, regression_bbox_targets, tf_boxes

    iamdb = data.IamDataset('datasets/iamdb')
    iamdb.run()
    it = iamdb.get_iterator(infinite=True)

    pipeline = PipelineBase(it, batch_size=1, target_x=900, target_y=1200)
    pipeline.add_augmentation(BoxRearange)
    pipeline.add_augmentation(DialationErosio, apply_prob=0.2)
    pipeline.add_augmentation(Slant, apply_prob=0.2)
    pipeline.add_augmentation(Resize)

    # Heatmap
    pipeline.add_extender('heatmap', from_image_to_heatmap, in_batch='vstack', trim=0.2)
    # Regression
    pipeline.add_extender(('reg_target', 'reg_flags'), regression_bbox_targets, in_batch='vstack', fmap_w=112,
                          fmap_h=150)
    # TF Boxes
    pipeline.add_extender(('tf_gt_boxes',), tf_boxes, in_batch='hstack')
    # Phoc Extenders
    pipeline.add_extender(('phocs', 'tf_gt_boxes'), phoc_embedding, in_batch='hstack')
    pipeline.run(num_producers=1)

    for i in range(100):
        batch = pipeline.pull_data()

        img, boxes = batch['image'][0, :], batch['gt_boxes']
        debugShowBoxes(img / 255. / 255., boxes=boxes[:, 1:], wait=300)

if __name__ == '__main__':
    test_augmentations()



