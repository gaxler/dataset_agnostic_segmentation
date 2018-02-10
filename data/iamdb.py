from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string

import numpy as np
from pathlib2 import Path

from lib.helpers import xywh_to_xyxy
from data.metaimage import MetaImage


class IamDataset(object):
    META_DATA_FOLDER = 'ascii'
    FORM_IMAGE_DIR = 'forms'

    WORD_FILE = 'words.txt'

    idx_files = {'train': 'trainset.txt',
                 'test': 'testset.txt',
                 'val1': 'validationset1.txt',
                 'val2': 'validationset2.txt'}

    VALIDATION_IDS = []

    def __init__(self, data_dir):
        # super(BaseDataset, self).__init__()

        self.data_dir = data_dir

        self.data_by_id = {}
        self.form_ids = None

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

    def _load_word_file(self):
        fpath = os.path.join(self.data_dir, self.META_DATA_FOLDER, self.WORD_FILE)
        file = open(fpath, 'r')
        for l in file.readlines():
            form_id, data = self._proc_line(l)
            if not form_id:
                continue

            if self.data_by_id.get(form_id, None):
                self.data_by_id[form_id].append(data)
            else:
                self.data_by_id[form_id] = []
                self.data_by_id[form_id].append(data)

        self.form_ids = self.data_by_id.keys()
        file.close()

    def _proc_line(self, line):

        if line[0] == "#":
            return None, None

        struct = line.split(' ')
        id = struct[0]
        form_id = self._proc_word_id_to_form_id(id)
        correct_segment = struct[1]
        gray_scale = struct[2]
        x = float(filter(lambda x: x in string.printable, struct[3]))
        y = float(filter(lambda x: x in string.printable, struct[4]))
        w = float(filter(lambda x: x in string.printable, struct[5]))
        h = float(filter(lambda x: x in string.printable, struct[6]))
        part_of_speech = filter(lambda x: x in string.printable, struct[7])
        word = filter(lambda x: x in string.printable, struct[8])

        self.word_count += 1

        d = {'id': id,
             'num_id': self.word_count,
             'box': xywh_to_xyxy([x, y, w, h]),
             'text': word,
             'part_of_speech': part_of_speech,
             'segment_ok': 1 if correct_segment == 'ok' else 0}
        return form_id, d

    def _proc_word_id_to_form_id(self, word_id):
        tmp = word_id.split('-')[:2]
        form_id = '%s-%s' % tuple(tmp)
        return form_id

    def bboxes_from_doc(self, doc):
        all_boxes = [d['box'] for d in doc]
        return np.array(all_boxes)

    def image_path_from_doc(self, doc_id):
        path = os.path.join(self.data_dir, self.FORM_IMAGE_DIR, '%s.png' % doc_id)
        return path

    def _make_validation_set(self, size):
        assert size < 1.0, 'val cannot exceed 100%'
        l = len(self.form_ids)
        val_size = int(l*size)
        val_set = np.random.choice(np.array(self.form_ids), size=val_size, replace=False)
        return val_set

    def get_iterator(self, dataset='train', infinite=False, random=True):
        assert dataset in self.idx_files.keys(), 'Following datasets exists: %s' % str(self.idx_files.keys())
        allowed_idx = get_form_idx_from_file(str(Path(self.data_dir) / self.idx_files[dataset]))
        while True:
            maybe_random_idx = np.random.permutation(allowed_idx) if random else allowed_idx
            for idx in maybe_random_idx:
                data_dict = self._produce_data_dict(idx)
                yield MetaImage(data_dict)

            if not infinite:
                return

    def get_generator(self, infinite=False, validation=False, random=True):
        while True:
            random_idx = np.random.permutation(self.form_ids) if random else self.form_ids
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

    def _produce_data_dict(self, idx, ):
        doc = self.data_by_id[idx]
        data_dict = {'path': self.image_path_from_doc(idx),
                     'bboxes': self.bboxes_from_doc(doc),
                     'words': doc}
        return data_dict

def get_form_idx_from_file(file_name):
    with open(file_name, 'rb') as fp:
        lines = fp.readlines()
        idx = [v.strip() for v in lines]

    form_idx = {}
    for widx in idx:
        form_id = '-'.join(widx.split('-')[:-1])
        form_idx[form_id] = 1
    return form_idx.keys()
