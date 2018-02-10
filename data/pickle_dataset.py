from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pklRick

from pathlib2 import Path


def pickle_iterator(it, output_dest):

    output_path = Path(output_dest)
    c = 0
    for mi in it:
        fname = output_path / ('%s.pkl' % Path(mi.path).stem)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        with fname.open('wb') as fnp:
            pklRick.dump(mi, fnp)
        c += 1
    print ('Saved %d pickles' % c)
    return output_dest


def get_pkl_iterator(pkl_path, infinite=False, suffix='pkl'):
    pkl_path = Path(pkl_path)
    c = 0
    while True:
        for pk in pkl_path.glob('*.%s' % suffix):
            mi = pklRick.load(pk.open('rb'))
            c += 1
            yield mi

        assert c > 0, "Iterator from %s has zero meta images" % str(pkl_path)
        if not infinite:
            break


if __name__ == '__main__':
    import data
    DATA_DIR = 'datasets/iamdb'
    PKL_DSET_PATH = 'datasets/iamdb/pkl'
    iamdb = data.IamDataset(data_dir=DATA_DIR)
    iamdb.run()

    it = iamdb.get_iterator()
    dest = pickle_iterator(it, PKL_DSET_PATH)
    it = get_pkl_iterator(dest)
    print ('/')