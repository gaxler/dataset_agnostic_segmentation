from pathlib2 import Path

from data import IamDataset, IclefDataset, IcdarDataset, BotanyDataset, BotanyTest
from data.mixer import mix_iterators
from data.pickle_dataset import get_pkl_iterator


def get_dataset_loader(args):
    return get_dataset_loader_by_name(args.dataset, args.data_type, args=args)


def get_dataset_loader_by_name(dataset, data_type, args=None):

    if data_type[0] == '#':
        data_type = data_type.split('#')[1:]

    dprob = None
    if args is not None and args.data_type_prob is not None and args.data_type_prob[0] == '#':
        dprob = map(float, args.data_type_prob.split('#')[1:])
        assert len(dprob) == len(data_type), 'Probability length %d while %d data types passed. Must be equal' % (len(dprob), len(data_type))

    if dataset == 'iamdb':
        iamdb = IamDataset(data_dir='datasets/iamdb')
        iamdb.run()
        if isinstance(data_type, list):
            dlist = [iamdb.get_iterator(dataset=dtyp, infinite=True) for dtyp in data_type]
            it = mix_iterators(dlist, it_probs=dprob)
        else:
            it = iamdb.get_iterator(dataset=data_type, infinite=True)

    elif dataset == 'iclef':
        iclef = IclefDataset(data_dir='datasets/iclef')
        iclef.run()
        if isinstance(data_type, list):
            dlist = [iclef.get_iterator(dataset=dtyp, infinite=True) for dtyp in data_type]
            it = mix_iterators(dlist, it_probs=dprob)
        else:
            it = iclef.get_iterator(dataset=data_type, infinite=True)

    elif dataset == 'botany':
        if data_type == 'test':
            botany = BotanyTest(data_dir='datasets/botany_test')
            it = botany.get_iterator(infinite=True)
        else:
            botany = BotanyDataset(data_dir='datasets/botany')
            if isinstance(data_type, list):
                dlist = [botany.get_iterator(dataset=dtyp, infinite=True) for dtyp in data_type]
                it = mix_iterators(dlist, it_probs=dprob)
            else:
                it = botany.get_iterator(dataset=data_type, infinite=True)

    elif dataset == 'icdar':
        test_set = data_type == 'test'
        icdar = IcdarDataset(data_dir='datasets/icdar', test_set=test_set)
        icdar.run()
        if isinstance(data_type, list):
            dlist = [icdar.get_iterator(dataset=dtyp, infinite=True) for dtyp in data_type]
            it = mix_iterators(dlist, it_probs=dprob)
        else:
            it = icdar.get_iterator(dataset=data_type, infinite=True)

    elif dataset == 'from-pkl-folder':
        print ('NOTICE: Expect path to folder containing *.pkl files as data_type')
        p = Path(data_type)
        assert p.exists(), '%s is not a valid pkl folder'
        it = get_pkl_iterator(p, infinite=True)

    else:
        raise ValueError('Unknown dataset %s' % dataset)
    return it


def log_params(logger, args, params=None):
    logger('')
    logger('####### %s ########' % args.name)
    logger('')
    logger('######## args ########')
    sorted_args = sorted([arg for arg in vars(args)])
    for arg in sorted_args:
        logger('%s: %s' % (str(arg), getattr(args, arg)))

    if params is not None:
        logger('######## PARAMS ########')

        for par in dir(params):
            if not par.startswith('__'):
                logger('%s: %s' % (str(par), getattr(params, par)))

    return


def write_params_to_args(params, args, override=False):
    """ Join params into args and return args"""
    assert isinstance(params, dict), 'params must be a dict'
    for par in params.keys():
        if not par.startswith('__'):
            if not hasattr(args, par) or override:
                print ('adding %s: %s' % (par, str(params.get(par))))
                setattr(args, par, params.get(par))
    return args
