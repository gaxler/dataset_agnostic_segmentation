
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib2 import Path

import cv2

from lib.show_images import debugShowBoxes
from lib.helpers import clean_word


class MetaImage(object):
    path = None
    bboxes = None
    words = None
    DEBUG = False
    damaged = False

    def __init__(self, meta_data):
        self._meta_data = meta_data
        assert 'path' in meta_data.keys()
        assert 'bboxes' in meta_data.keys()
        assert 'words' in meta_data.keys()

        self.__dict__.update(meta_data)
        pass

    def filter_bad_words(self):
        words = [clean_word(w['text']) for w in self.words]
        pass

    def getImage(self):
        try:
            if isinstance(self.path, Path):
                img = cv2.imread(str(self.path))
            else:
                img = cv2.imread(self.path)
        except:
            return None

        return img

    def showImage(self, normalize=None, wait=100):
        img = self.getImage()
        if normalize:
            img = img / normalize
        boxes = self.bboxes
        debugShowBoxes(debugimg=img, boxes=boxes, wait=wait)
        pass

    def showCostumBoxes(self, boxes, normalize=None, wait=100):
        img = self.getImage()
        if normalize:
            img = img / normalize
        debugShowBoxes(debugimg=img, boxes=boxes, wait=wait)
        pass

    @property
    def doc_name(self):
        return Path(self.path).stem

    @property
    def doc_format(self):
        return Path(self.path).suffix

    @property
    def word_list(self):
        if isinstance(self.words, list) and len(self.words) > 0:
            words = [_check_str(clean_word(w['text'])) for w in self.words]
            return words

    def get_good_words_and_boxes_idx(self):
        if isinstance(self.words, list) and len(self.words) > 0:
            words = [clean_word(w['text']) for w in self.words if w.get('segment_ok', 1) == 1]
            boxes = [n for n, w in enumerate(self.words) if w.get('segment_ok', 1) == 1]
            assert len(words) == len(boxes), "Boxes IDs and words not same length!!!"
            return boxes, words

        return None, None


class MixUpMetaImage(MetaImage):
    pass



def _check_str(x):
    try:
        str(x)
        return str(x)
    except:
        return x.encode('ascii', 'ignore')
