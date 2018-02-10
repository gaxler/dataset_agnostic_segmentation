from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def mix_iterators(it_list, it_probs=None):
    if it_probs is None:
        it_probs = np.ones((len(it_list)))
    it_probs = np.array(it_probs)
    probs = it_probs / it_probs.sum()

    while len(it_list) > 0:
        the_one = np.random.choice(range(len(it_list)), size=1, p=probs)[0]
        try:
            mi = it_list[the_one].next()
        except StopIteration:
            it_list.pop(the_one)
            tmp = np.delete(probs, the_one).astype(np.float32)
            probs = tmp / tmp.sum()
            continue
        yield mi


class MixerDataset(object):

    ready = False

    def __init__(self, dset_list, dset_probs=None):
        if dset_probs is None:
            self._dset_probs = np.ones(shape=(len(dset_list)), dtype=np.float32) / len(dset_list)
        else:
            self._dset_probs = dset_probs

        self._dset_list = dset_list

    def run(self, val_paths, val_ratio=0.2):
        if val_paths is None:
            for db in self._dset_list:
                db.run()
        else:
            for db, p in zip(self._dset_list, val_paths):
                if p is None:
                    db.run()
                elif p.exists():
                    val_ids = np.load(str(p))
                    db.run(validation_ids=val_ids)
                else:
                    db.run(validation_size=val_ratio)
                    np.save(str(p), db.VALIDATION_IDS)

        self.ready = True
        pass

    def get_generator(self, infinite=False, validation=False, random=True):
        # TODO: adjust to support non-uniform distribution
        its = [db.get_generator(infinite=infinite, validation=validation, random=random)
               for db in self._dset_list]
        probs = np.ones(shape=len(its), dtype=np.float32) / len(its)
        while len(its) > 0:
            the_one = np.random.choice(range(len(its)), size=1, p=probs)[0]
            try:
                mi = its[the_one].next()
            except StopIteration:
                its.pop(the_one)
                tmp = np.delete(probs, the_one).astype(np.float32)
                probs = tmp / tmp.sum()
                continue
            yield mi

