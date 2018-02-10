from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET

import numpy as np
# from .metaimage import MetaImage
from data.metaimage import MetaImage

from lib.show_images import debugShowBoxes


class BotanyDataset(object):
    WORD_PATTERN = 'xml/Botany_Train_%s_WL.xml'
    PAGES_PATTERN = 'Botany_Train_%s_PageImages'

    VALIDATION_IDS = []

    idx_files = {'I': 'trainset.txt',
                 'II': 'testset.txt',
                 'III': 'validationset1.txt',
                 'train': 'validationset1.txt'
                 }

    _data_map = {'train': 'III', 'val1': 'I', 'val2': 'II'}

    dataset_ids = idx_files.keys()

    def __init__(self, data_dir):
        self.data_path = Path(data_dir) if not isinstance(data_dir, Path) else data_dir

        self.data_by_id = {}
        self.page_ids = None

        self.word_count = 0
        self.doc_count = 0

        self.ready = True

    def run(self):
        self.ready = True

    def _get_pages_path(self, dataset):
        p = self.data_path / str(self.PAGES_PATTERN % dataset)
        return p

    def _get_image_filename_list(self, dataset):
        p = self._get_pages_path(dataset)
        return [f.stem for f in p.glob('*.jpg')]

    def _get_word_file(self, data_id):
        return self.data_path / (self.WORD_PATTERN % data_id)

    def _load_word_file(self, data_id):
        assert data_id in self.dataset_ids, 'Not valid dataset type [%s]' % str(self.dataset_ids)
        data_by_id = {}
        word_file = self._get_word_file(data_id)
        tree = ET.parse(str(word_file))
        root = tree.getroot()
        for child in root:
            if child.tag == 'spot':
                page_name, data = self._proc_line(child)

                if data_by_id.get(page_name, None):
                    data_by_id[page_name].append(data)
                else:
                    data_by_id[page_name] = []
                    data_by_id[page_name].append(data)

        # self.page_ids = self.data_by_id.keys()
        del root
        del tree
        return data_by_id

    def _proc_line(self, child):

        word = str(child.attrib['word']) if type(child.attrib['word']) == str else ''
        img = str(child.attrib['image'])
        x = int(child.attrib['x'])
        y = int(child.attrib['y'])
        w = int(child.attrib['w'])
        h = int(child.attrib['h'])

        bbox = np.array([x, y, x + w, y + h])

        self.word_count += 1

        d = {'id': img,
             'num_id': self.word_count,
             'box': bbox,
             'text': word}
        return img, d

    def bboxes_from_doc(self, doc):
        all_boxes = [d['box'] for d in doc]
        return np.array(all_boxes)

    def image_path_from_doc(self, doc_id, dataset):
        path = self._get_pages_path(dataset) / str(doc_id)
        assert path.is_file(), 'Not a valid file %s' % str(path)
        return path

    def get_iterator(self, dataset='I', infinite=False, random=True):
        dataset = self._data_map.get(dataset, dataset)
        assert dataset in self.idx_files.keys(), 'Following datasets exists: %s' % str(self.idx_files.keys())
        self.dataset = dataset
        data_by_id = self._load_word_file(dataset)
        allowed_idx = data_by_id.keys()
        while True:
            maybe_random_idx = np.random.permutation(allowed_idx) if random else allowed_idx
            for idx in maybe_random_idx:
                data_dict = self._produce_data_dict(idx, data_by_id[idx], dataset)
                yield MetaImage(data_dict)

            if not infinite:
                return

    def _produce_data_dict(self, idx, doc, dataset):
        data_dict = {'path': self.image_path_from_doc(idx, dataset),
                     'bboxes': self.bboxes_from_doc(doc),
                     'words': doc}
        return data_dict


def get_form_idx_from_file(file_name):
    with open(file_name, 'rb') as fp:
        lines = fp.readlines()
        idx = [v.strip() for v in lines]
    return idx


class BotanyTest(BotanyDataset):
    WORD_PATTERN = 'xml/Botany_Test_GT_SegFree_QbS.xml'
    PAGES_PATTERN = 'Botany_Test_PageImages'

    VALIDATION_IDS = []

    idx_files = {'I': 'trainset.txt',
                 'II': 'testset.txt',
                 'III': 'validationset1.txt',
                 'test': 'testset.txt'}

    dataset_ids = idx_files.keys()

    def _get_pages_path(self, dataset):
        """ This render dataset useless and only read test data"""
        p = self.data_path / str(self.PAGES_PATTERN)
        return p

    def _get_word_file(self, data_id):
        """ This render data_id useless and only read test data"""
        return self.data_path / self.WORD_PATTERN


if __name__ == '__main__':
    from pathlib2 import Path
    from data.augmentations import Resize
    DATA_DIR = 'datasets/botany_test/'
    for k in ['I', 'II', 'III']:
        data = BotanyTest(data_dir=DATA_DIR)
        it = data.get_iterator(dataset=k, infinite=False)
        words = 0
        pages = 0
        for mi in it:
            img = mi.getImage()
            res = Resize(target_height=1200, target_width=900)
            new_img, new_boxes = res.image_resize(img, mi.bboxes)
            debugShowBoxes(debugimg=new_img / 255**2, boxes=new_boxes, wait=50)
            words += mi.bboxes.shape[0]
            pages += 1
        print (k, pages, words)

