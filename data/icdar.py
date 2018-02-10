from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib2 import Path
import cv2
import numpy as np

from .metaimage import MetaImage


class IcdarDataset(object):

    _allowed_data_types = ['train', 'test']
    _cache_pattern = 'gt_boxes_%s.npy'
    _data_type = None
    _image_dir_pattern = 'images_%s'
    _image_suffix = 'tif'

    _ground_truth_pattern = 'gt_words_%s'
    _ground_truth_suffix = 'dat'

    _image_paths = None
    page_ids = None
    _ground_truth_paths = None

    _cached_gt_boxes = None
    _damaged_gt_boxes = None

    VALIDATION_IDS = []

    _ready = False

    idx_files = {'train': 'trainset.txt',
                 'test': 'testset.txt',
                 'val1': 'validationset1.txt'}

    def __init__(self, data_dir, test_set=False):
        super(IcdarDataset, self).__init__()

        self._data_dir = Path(data_dir)
        self._data_type = self._allowed_data_types[0] if not test_set else self._allowed_data_types[1]

        assert self._data_dir.exists(), 'Not a valid datadir'

    def run(self):
        self._build_file_database()
        self._load_or_cache()

        self.ready = True

    def _build_file_database(self):
        # Images
        p = self._image_paths = self._data_dir / (self._image_dir_pattern % self._data_type)
        self.page_ids = sorted(map(lambda x: x.stem, p.glob('*.%s' % self._image_suffix)))

        # Ground Truths
        self._ground_truth_paths = self._data_dir / (self._ground_truth_pattern % self._data_type)

    def _produce_data_dict(self, idx):
        boxes = self._cached_gt_boxes.get(idx, None)
        damaged = self._damaged_gt_boxes.get(idx, None)
        # not annotated for words
        words = []
        path = str(self._image_paths / ('%s.%s' % (idx, self._image_suffix)))
        return {'path': path, 'words': words, 'bboxes': boxes, 'damaged': damaged}

    def get_iterator(self, dataset='train', infinite=False, random=True):
        assert dataset in self.idx_files.keys(), 'Following datasets exists: %s' % str(self.idx_files.keys())
        allowed_idx = get_form_idx_from_file(str(Path(self._data_dir) / self.idx_files[dataset]))
        while True:
            maybe_random_idx = np.random.permutation(allowed_idx) if random else allowed_idx
            for idx in maybe_random_idx:
                data_dict = self._produce_data_dict(idx)
                yield MetaImage(data_dict)

            if not infinite:
                return

    def _load_or_cache(self):
        cache_path = self._data_dir / (self._cache_pattern % self._data_type)
        if not cache_path.exists():
            gt_dict = {}
            damaged_ids = {}
            print('Building Ground Truth Cache')
            all_images = self._image_paths.glob('*.%s' % self._image_suffix)
            i = 0
            for img in all_images:
                idx = img.stem
                shape = cv2.imread(str(img)).shape
                gt_file = self._ground_truth_paths / ('%s.%s' % (img.parts[-1], self._ground_truth_suffix))
                gt_map = load_map_from_file(str(gt_file), target_x=shape[0], target_y=shape[1])
                bboxes, damaged = bbox_from_map(gt_map)

                gt_dict[idx] = bboxes
                damaged_ids[idx] = damaged
                i += 1
                print ('%d) Done %s' % (i, img.stem))
            np.save(str(cache_path), np.array([gt_dict, damaged_ids]))
            pass

        gt_dict, damaged_ids = np.load(str(cache_path))
        assert isinstance(gt_dict, dict), 'Dictionary expected'
        assert isinstance(damaged_ids, dict), 'Dictionary expected'
        self._cached_gt_boxes = gt_dict
        self._damaged_gt_boxes = damaged_ids

    def _make_validation_set(self, size):
        assert size < 1.0, 'val cannot exceed 100%'
        l = len(self.page_ids)
        val_size = int(l*size)
        val_set = np.random.choice(np.array(self.page_ids), size=val_size, replace=False)
        return val_set


def load_map_from_file(filename, target_x, target_y, bytes_to_read=4):
    f = open(filename, 'rb')

    s = f.read()
    a = []
    for i in range(target_x*target_y):
        a.append([ord(k) for k in s[4*i:(i+1)*4]])
    b = np.array(a)
    b = b + np.ones_like(b)
    m = np.max(b, axis=0)
    if m[3] != 1 or m[2] != 1:
        print ('OMG')
    c = b[:, 0] + b[:, 1]*1000
    c = c.reshape(target_x, target_y)

    magic = np.unique(c).shape[0]
    cum_magic = np.load('magic.npy')[0] if Path('magic.npy').exists() else 0
    cum_magic += magic
    print ('so far %d' % cum_magic)
    np.save('magic.npy', np.array([cum_magic]))
    f.close()
    return c


def bbox_from_map(map):
    damaged = False
    # ignore the zero
    uns = np.unique(map)[1:]
    boxes = []
    for u in uns:
        dat = np.where(map == u)
        points = np.array(zip(dat[1], dat[0]))
        box = np.hstack([points.min(axis=0), points.max(axis=0)])
        box = np.squeeze(box)
        boxes.append(box)
    boxes = np.vstack(boxes)
    return boxes, damaged


def get_form_idx_from_file(file_name):
    with open(file_name, 'rb') as fp:
        lines = fp.readlines()
        idx = [v.strip() for v in lines]
    return idx


if __name__ == '__main__':
    """
    Test code to make sure cached bounding box data is complete 
    """
    ICDAR_DATADIR = 'datasets/icdar'

    total_train_boxes = 0
    for dset in ('train', 'val1'):
        d = IcdarDataset(data_dir=ICDAR_DATADIR, test_set=False)
        d.run()
        it = d.get_iterator(dataset=dset, infinite=False, random=False)
        for mi in it:
            total_train_boxes += mi.bboxes.shape[0]

    print ('Total Train Words: %d' % total_train_boxes)
    train_pass = total_train_boxes == 29423

    total_test_boxes = 0
    d = IcdarDataset(data_dir=ICDAR_DATADIR, test_set=True)
    d.run()
    it = d.get_iterator(dataset='test', infinite=False, random=False)
    for mi in it:
        total_test_boxes += mi.bboxes.shape[0]
    test_pass = total_test_boxes == 23525
    print ('Total Test Words: %d' % total_test_boxes)

    assert all((train_pass, test_pass))



