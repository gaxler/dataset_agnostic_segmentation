from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib2 import Path

import numpy as np
from .metaimage import MetaImage


class IclefDataset(object):
    BBOX_FILE = 'bboxs_train_for_query-by-example.txt/bboxs_train_for_query-by-example.txt'

    PAGES_PATTERN = 'pages_%s_jpg/pages_%s'

    VALIDATION_IDS = []

    idx_files = {'train': 'trainset.txt',
                 'test': 'testset.txt',
                 'val1': 'validationset1.txt',
                 'val2': 'validationset2.txt'}

    def __init__(self, data_dir, dataset='train'):
        self.dataset = dataset
        self.data_path = Path(data_dir) if not isinstance(data_dir, Path) else data_dir

        self.data_by_id = {}
        self.page_ids = None

        self.word_count = 0
        self.doc_count = 0

        self.ready = False

    def run(self, validation_ids=None, validation_size=0.0):
        self._load_word_file()
        # make validation
        if validation_size > 0. and validation_ids is None:
            self.VALIDATION_IDS = self._make_validation_set(validation_size)
        elif validation_ids is not None:
            assert isinstance(validation_ids, np.ndarray)
            self.VALIDATION_IDS = validation_ids

        self.ready = True

    def _get_pages_path(self):
        dataset = self.dataset
        p = self.data_path / str(self.PAGES_PATTERN % (dataset, dataset))
        return p

    def _get_image_filename_list(self):
        p = self._get_pages_path()
        return [f.stem for f in p.glob('*.jpg')]

    def _load_word_file(self):
        file = self.data_path / self.BBOX_FILE
        with file.open('r') as f:
            for l in f.readlines():
                page_name, data = self._proc_line(l)

                if self.data_by_id.get(page_name, None):
                    self.data_by_id[page_name].append(data)
                else:
                    self.data_by_id[page_name] = []
                    self.data_by_id[page_name].append(data)

        self.page_ids = self.data_by_id.keys()

    def _proc_line(self, line):

        if line[0] == "#":
            return None, None

        struct = line.split(' ')
        file_name = struct[0]
        l = struct[1]
        bbox = np.array(l.split('+')[0].split('x') + l.split('+')[1:]).astype(np.int)
        bbox = np.array([bbox[2], bbox[3], bbox[2] + bbox[0], bbox[3]+bbox[1]])

        word = struct[2]

        self.word_count += 1

        d = {'id': file_name,
             'num_id': self.word_count,
             'box': bbox,
             'text': word}

        return file_name, d

    def bboxes_from_doc(self, doc):
        all_boxes = [d['box'] for d in doc]
        return np.array(all_boxes)

    def image_path_from_doc(self, doc_id):
        path = self._get_pages_path() / str(doc_id +'.jpg')
        assert path.is_file(), 'Not a valid file'
        return path

    def _make_validation_set(self, size):
        assert size < 1.0, 'val cannot exceed 100%'
        l = len(self.page_ids)
        val_size = int(l*size)
        val_set = np.random.choice(np.array(self.page_ids), size=val_size, replace=False)
        return val_set

    def get_iterator(self, dataset='train', infinite=False, random=True):
        assert dataset in self.idx_files.keys(), 'Following datasets exists: %s' % str(self.idx_files.keys())
        allowed_idx = get_form_idx_from_file(str(Path(self.data_path) / self.idx_files[dataset]))
        while True:
            maybe_random_idx = np.random.permutation(allowed_idx) if random else allowed_idx
            for idx in maybe_random_idx:
                data_dict = self._produce_data_dict(idx)
                yield MetaImage(data_dict)

            if not infinite:
                return

    def get_generator(self, infinite=False, validation=False, random=True):
        while True:
            random_idx = np.random.permutation(self.page_ids) if random else self.page_ids
            for idx in random_idx:
                if idx in self.VALIDATION_IDS:
                    if validation:
                        data_dict = self._produce_data_dict(idx)
                        yield MetaImage(data_dict)
                    else:
                        continue
                else:
                    if validation:
                        continue
                    else:
                        data_dict = self._produce_data_dict(idx)
                        yield MetaImage(data_dict)
            if not infinite:
                return

    def _produce_data_dict(self, idx):
        doc = self.data_by_id[idx]
        data_dict = {'path': self.image_path_from_doc(idx),
                     'bboxes': self.bboxes_from_doc(doc),
                     'words': doc}
        return data_dict


def get_form_idx_from_file(file_name):
    with open(file_name, 'rb') as fp:
        lines = fp.readlines()
        idx = [v.strip() for v in lines]
    return idx