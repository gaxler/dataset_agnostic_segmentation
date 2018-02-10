from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Queue import Queue
from pathlib2 import Path
from threading import Thread
from functools import partial

import cv2
import numpy as np

from data.augmentations import AugmentationBase, Resize


class Producer(Thread):
    DEBUG = False

    def __init__(self, get_data_func, pull_size=1, in_queue=None, out_queue=None, tag=None):
        """
        get_data_func must support iterable
        empty list should be returned in case there's not output
        """

        self.pull_size = pull_size
        name = 'Producer' + ('' if not tag else ' %d' % tag)
        super(Producer, self).__init__(name=name)
        self.daemon = True
        self.should_stop = False
        self.thread_func = get_data_func
        self.out_queue = out_queue
        self.in_queue = in_queue

    def run(self):
        while not self.should_stop:
            if self.in_queue is not None:
                if self.pull_size > 1:
                    inputs = [self.in_queue.get(block=True, timeout=None) for _ in range(self.pull_size)]
                else:
                    inputs = self.in_queue.get(block=True, timeout=None)
                outputs = self.thread_func(inputs)
                outputs = self._iterable_check(outputs)
                if outputs is None:
                    continue
            else:
                outputs = self.thread_func()
                outputs = self._iterable_check(outputs)

            for output in outputs:
                self.out_queue.put(output, block=True, timeout=None)

    def _iterable_check(self, outputs):
        if not isinstance(outputs, (tuple, list, np.ndarray)):
            outputs = (outputs,)
        return outputs


class InBatchKeys:
    list = 'list'
    vstack = 'vstack'
    hstack = 'hstack'


class PipelineError(Exception):
    pass

class DataBatch(object):
    images = None
    # Heatmap
    truth_maps = None
    trim_maps = None
    gt_boxes = None
    # RPN
    rpn_labels = None
    rpn_bbox_targets = None
    rpn_bbox_inside_weights = None
    rpn_bbox_outside_weights = None
    # Center Boxes
    wh_targets = None
    wh_labels = None
    # Meta data
    meta_images = None
    # Embedings
    phocs = None


class PipelineBase(object):
    """
    Multithreaded MetaImage loader
    Has 4 queues:
        (1) MetaImage loader - threads collect metaimages form data generators
        (2) Image processing queue - Loads images and boxes from MetaImage and resizes them
        (3) Input Format - trnasofrm images to format needed for model input
        (4) Batcher - put input format images
    """
    PERMUTATION_BATCH = 100
    DEBUG = False
    IMAGE_PROC_QUEUE_SIZE = 10

    def __init__(self, iterator, batch_size, fmap_x=None, fmap_y=None, target_y=None, target_x=None, logger=None,
                 **kwargs):

        self.fmap_height = fmap_y
        self.fmap_width = fmap_x
        self.target_h = target_y
        self.target_w = target_x
        self.batch_size = batch_size

        self._iterator = iterator
        self.meta_queue = Queue(maxsize=(self.PERMUTATION_BATCH + 1))
        self.image_proc_queue = Queue(maxsize=(self.IMAGE_PROC_QUEUE_SIZE * self.batch_size))
        self.input_format_queue = Queue(maxsize=(self.IMAGE_PROC_QUEUE_SIZE * self.batch_size))
        self.batch_queue = Queue(maxsize=self.batch_size)

        self.meta_producers = []
        self.image_proc_producers = []
        self.input_format_producers = []
        self.batch_producers = []

        self._augmentations = []
        self._extenders = []

        self._ready = False
        self._finished_data_generation = False

        self.logger = logger if logger is not None else print

    def run(self, num_producers):
        self.build_producers(num_producers=num_producers)
        self.start_producers()
        self._ready = True

    def build_producers(self, num_producers):
        # load metadata
        p_meta = Producer(get_data_func=self.get_metadata, out_queue=self.meta_queue)
        self.meta_producers.append(p_meta)
        for i in range(num_producers):
            # load image and properly reshape
            for i in range(2):
                p_image_proc = Producer(get_data_func=self.process_metadata_and_load_image, in_queue=self.meta_queue,
                                        out_queue=self.image_proc_queue)
                self.image_proc_producers.append(p_image_proc)

            # prepare data for neural net
            p_input_format = Producer(get_data_func=self.prepare_data_for_neural_net, in_queue=self.image_proc_queue,
                                      out_queue=self.input_format_queue)
            self.input_format_producers.append(p_input_format)

            batcher = Producer(get_data_func=self.batcher_function, in_queue=self.input_format_queue,
                               out_queue=self.batch_queue, pull_size=self.batch_size)
            self.batch_producers.append(batcher)
        pass

    def start_producers(self):

        def start_producer(producer_list, tag=None):
            if len(producer_list) > 0:
                for p in producer_list:
                    if tag is not None:
                        self.logger('Starting %s Producers' % tag)
                    else:
                        self.logger('Starting Meta Producers')
                    p.start()

        start_producer(self.meta_producers, tag='Meta')
        start_producer(self.image_proc_producers, tag='Image Processing')
        start_producer(self.input_format_producers, tag='Input Formatting')
        start_producer(self.batch_producers, tag='Batch')

        return

    def stop_producers(self):

        def stop_producer(producer_list):
            if len(producer_list) > 0:
                for p in producer_list:
                    p.should_stop = True

        self.logger('Stopping Producers')
        stop_producer(self.meta_producers)
        stop_producer(self.image_proc_producers)
        stop_producer(self.input_format_producers)
        stop_producer(self.batch_producers)

        return

    def get_metadata(self):
        """load image metadata"""
        generator = self._iterator

        metas = []
        for i in range(self.PERMUTATION_BATCH):
            try:
                meta_data = generator.next()
                if meta_data is None:
                    continue
                metas.append(meta_data)
            except StopIteration:
                self._finished_data_generation = True
                break
        return metas

    def add_augmentation(self, aug, **kwargs):
        if not issubclass(aug, AugmentationBase):
            self.logger('Unsupported augmenataion %s... Ignoring...' % aug.__name__)
        aug_inst = aug(target_width=self.target_w, target_height=self.target_h, **kwargs)
        self._augmentations.append(aug_inst)

    def process_metadata_and_load_image(self, meta_image):
        """
        this function receives metaimage class\
        class has path string, bboxes 2D np.array [x1,y1,x2,y2], getImage method and showImage method
        """
        if self.DEBUG:
            self.logger('Loading some images')

        image = meta_image.getImage()
        bboxes = meta_image.bboxes

        if self.target_h is not None and self.target_w is not None:
            # if no augmentations is added, add the resize
            if not self._augmentations:
                self.add_augmentation(Resize)

            # We support augmentations only if target size is specified
            for aug in self._augmentations:
                image, bboxes, meta_image = aug.apply(image, bboxes, meta_image)

        output = (image, bboxes, meta_image)

        return np.array([output])

    def add_extender(self, names, extender_func, in_batch='list', **kwargs):
        allowed = [InBatchKeys.list, InBatchKeys.vstack, InBatchKeys.hstack]
        assert in_batch in allowed, '%s unsupported. %s' % (in_batch, str(allowed))
        extender = partial(extender_func, names=names, **kwargs)
        self._extenders.append((extender, in_batch))

    def prepare_data_for_neural_net(self, image_and_meta):
        image = image_and_meta[0]
        gt_boxes = image_and_meta[1]
        meta_image = image_and_meta[2]

        output_dict = {'image': (image, InBatchKeys.vstack),
                       'gt_boxes': (gt_boxes, InBatchKeys.hstack),
                       'meta_image': (meta_image, InBatchKeys.list)
                       }
        for extender, in_batch in self._extenders:
            name, data = extender(image, gt_boxes, meta_image)
            if name is None or data is None:
                continue
            if not isinstance(name, (list, tuple)) and not isinstance(data, (list, tuple)):
                name = (name,)
                data = (data, )
            assert len(name) == len(data), 'Names must match data got %d names and %d data' % (len(name), len(data))
            for i in range(len(name)):
                output_dict.update({name[i]: (data[i], in_batch)})

        return output_dict

    def batcher_function(self, batch_slice):
        if not isinstance(batch_slice, (list, np.ndarray)):
            batch_slice = [batch_slice]

        batch_dict = {}
        in_batch_treatment = {}
        # batch_slice is an output_dict from prepare_data_for_neural_net
        for j, s in enumerate(batch_slice):
            for k, v in s.iteritems():
                data, in_batch = v
                in_batch_treatment[k] = in_batch
                dlist = batch_dict.get(k, [])
                if in_batch == InBatchKeys.vstack:
                    data = data[np.newaxis, :]
                if in_batch == InBatchKeys.hstack:
                    # Assumed we treat something like bboxes or phocs for hstack  -> (n, d) numpy arrays
                    # Assume we add a batch_id dim, (n, m) -> (n, m+1) where 0 dim is batch id
                    data = np.hstack((np.ones((data.shape[0], 1))*j, data))
                dlist.append(data)
                batch_dict[k] = dlist

        for k, v in in_batch_treatment.iteritems():
            raw_data = batch_dict[k]

            if in_batch_treatment[k] == InBatchKeys.list:
                # It's already a list. do nothing
                data = raw_data
            if in_batch_treatment[k] == InBatchKeys.vstack:
                data = np.vstack(raw_data)
            if in_batch_treatment[k] == InBatchKeys.hstack:
                # Not a mistake. We stacking it after added batch id dim in first loop above
                data = np.vstack(raw_data)

            batch_dict[k] = data

        return batch_dict

    def pull_data(self):
        if not self._ready:
            raise PipelineError('use run() before pulling data from pipe')

        data = self.batch_queue.get(block=True, timeout=None)
        return data

    @staticmethod
    def image_resize(image, boxes, target_y, target_x, debug=False):

        if float(image.shape[0]) / float(image.shape[1]) < target_y / target_x:
            f = float(target_x) / image.shape[1]
            dsize = (target_x, int(image.shape[0] * f))
        else:
            f = float(target_y) / image.shape[0]
            dsize = (int(image.shape[1] * f), target_y)

        image = cv2.resize(image, dsize=dsize)

        scaled_boxes = boxes * np.atleast_2d(np.array([f, f, f, f]))

        resized_image = cv2.copyMakeBorder(image,
                                           top=0,
                                           left=0,
                                           right=target_x - image.shape[1],
                                           bottom=target_y - image.shape[0],
                                           borderType=cv2.BORDER_REPLICATE)
        if debug:
            pass

        return resized_image, scaled_boxes

    def get_relative_points(self, fmap_w, fmap_h, as_batch=False):
        """
        Feature map relative points (centers of each feature pixel)
        """
        sh_x, sh_y = np.meshgrid(np.arange(fmap_w), np.arange(fmap_h))
        pts = np.vstack((sh_x.ravel(), sh_y.ravel())).transpose()
        cntr_pts = pts + np.array([0.5] * 2, np.float32)[np.newaxis, :]
        relative_pts = cntr_pts / np.array([fmap_w, fmap_h], np.float32)[np.newaxis, :]
        if as_batch:
            relative_pts = relative_pts[np.newaxis, :, :]
        return relative_pts


class FolderLoader(object):
    """ Loads images (resizes if needed) and their names from folder"""

    _supported_image_formats = ['jpg', '.jpg', 'png', 'tif']

    def __init__(self, folder, target_size=None):
        if target_size is not None:
            assert all([isinstance(target_size, (list, tuple)),
                        len(target_size) == 2]), \
                "target size must be list or tuple of the format (x,y)"
            self._target_size = target_size

        self._p = Path(folder)

    def _resize(self, image):
        """Resize the loaded image to a target size"""
        return image_resize(image=image, target_x=self._target_size[0], target_y=self._target_size[1])

    def _load(self, adress):
        """load image"""
        try:
            img = cv2.imread(adress)
        except:
            return None

        maybe_resized_image = self._resize(image=img) if self._target_size is not None else img

        return maybe_resized_image

    def generator(self):
        for f in self._p.glob('*.*'):
            if f.is_file() and f.suffix in self._supported_image_formats:
                img = self._load(adress=str(f))
                name = f.stem
                if img is not None:
                    yield img, name


def image_resize(image, boxes=None, target_y=None, target_x=None, debug=False):
    if target_y is None or target_x is None:
        return image, boxes

    if float(image.shape[0]) / float(image.shape[1]) < target_y / target_x:
        f = float(target_x) / image.shape[1]
        dsize = (target_x, int(image.shape[0] * f))
    else:
        f = float(target_y) / image.shape[0]
        dsize = (int(image.shape[1] * f), target_y)

    image = cv2.resize(image, dsize=dsize)

    resized_image = cv2.copyMakeBorder(image,
                                       top=0,
                                       left=0,
                                       right=target_x - image.shape[1],
                                       bottom=target_y - image.shape[0],
                                       borderType=cv2.BORDER_REPLICATE)
    if debug:
        pass

    if boxes is not None:
        scaled_boxes = boxes * np.atleast_2d(np.array([f, f, f, f]))
        return resized_image, scaled_boxes

    return resized_image


if __name__ == '__main__':
    from data.iamdb import IamDataset
    from data.data_extenders import phoc_embedding
    DATA_DIR = 'datasets/iclef'
    data = IamDataset(DATA_DIR)
    data.run()
    it = data.get_iterator(infinite=True)

    pipe = PipelineBase(it, batch_size=1,fmap_x=112, fmap_y=150, trim=0.2, target_x=900, target_y=1200)
    pipe.add_extender(('phocs', 'tf_gt_boxes'), phoc_embedding, in_batch='hstack')
    pipe.run(1)

    x = pipe.pull_data()