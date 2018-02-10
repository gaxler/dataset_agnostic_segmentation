from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from pathlib2 import Path
import cPickle as pklRick

import numpy as np
from numba import jit
from stop_words import get_stop_words

from lib.phocs import phoc_letters_and_digits

from lib.helpers import Timer, Logger
from settings.input_stream import get_dataset_loader_by_name


def calculate_results_folder(folder, threshold=0.9):
    """ """
    C = 0
    M = 0
    N = 0
    pages = 0
    for page in Path(folder).glob('**/*.json'):
        page_dict = json.load(page.open('rb'))
        correct, preds, gts = calculate_results_page(page_dict, threshold=threshold)
        C += correct
        M += preds
        N += gts
        pages += 1
    if N == 0 or M == 0:
        print ('No files %s' % folder)
        return
    DR = float(C) / N
    RA = float(C) / M
    F = 2*DR*RA / (DR + RA)
    print ('Total Words in folder: %d (in %d pages)' % (N, pages))
    print ('predicted Words: %d' % M)
    print ('Correct Words: %d' % C)
    print ('DR: %5.4f RA: %5.4f F: %5.4f' % (DR, RA, F))
    pass


def calculate_results_page(page_dict, threshold=0.9):
    """ """
    # Number of model predictions in a page
    preds = page_dict['predictions']
    # Number of gt words in a page
    gts = page_dict['gt_boxes']
    correct = 0
    for k in page_dict.keys():
        # Run over all gt and marke prediction as correct if it meets IoU criteria
        if 'word_' in k:
            word = page_dict[k]
            correct += (word['cover'] > threshold)*1
    return correct, preds, gts


def query_page_folder_phoc(queries, folder, threshold, dont_load=False, use_gt_phoc=False, filter_small=False):
    """
    Evaluate folder of predictions given queries and threshold
    """

    # Dicts of query words
    all_dists = {}
    all_matches = {}
    all_relevants = {}

    cache_name = 'queries_%d.pkl' % (threshold*100)
    c = 0
    load_time = Timer()
    qtime = Timer()
    all_phocs, _ = phoc_letters_and_digits(queries)
    if (Path(folder) / cache_name).exists() and not dont_load:
        all_dists, all_matches = pklRick.load((Path(folder) / cache_name).open('rb'))
    else:
        print ('No chache found caching to %s' % cache_name)
        # Run over all predicted pages
        for page in Path(folder).glob('**/*.json'):
            load_time.tic()
            try:
                page_dict = json.load(page.open('rb'))
            except ValueError:
                print ('Somethin worng in %s lets see if we can go on' % page.stem)
            c += 1
            qtime.tic()
            # Run all queries per page
            for p, query in enumerate(queries):
                dists, matches, word_idx, words_in_page = query_page_phoc(query, page_dict, threshold=threshold,
                                                                          phoc=all_phocs[p, :], use_gt_phoc=use_gt_phoc, filter_small=filter_small)
                tmp_dist = all_dists.get(query, [])
                tmp_dist.extend(dists)
                all_dists[query] = tmp_dist

                tmp_match = all_matches.get(query, [])
                tmp_match.extend(matches)
                all_matches[query] = tmp_match

                tmp_match = all_relevants.get(query, [])
                tmp_match.append(words_in_page)
                all_relevants[query] = tmp_match

    # Cache mAP base data for fast reproduction of evaluation
    pklRick.dump((all_dists, all_matches), (Path(folder) / cache_name).open('wb'))
    mAP = 0
    recall = 0
    accuracy = 0
    n = 0
    for query in queries:
        # Per query evaluation
        AP, rec, acc = _map_and_recall(all_dists[query], all_matches[query], all_relevants[query])
        if AP is None or rec is None or acc is None:
            continue
        # Running means
        mAP = (1 / float(n+1))*AP + (float(n) / (n+1))*mAP
        recall = (1 / float(n+1))*rec + (float(n) / (n+1))*recall
        accuracy = (1 / float(n+1))*acc + (float(n) / (n+1))*accuracy
        n += 1
    return mAP, recall, accuracy


@jit
def _map_and_recall(query_dists, query_matches, all_relevants, eps=10 ** -7):
    """ """
    query_dists = np.array(query_dists)
    query_matches = np.array(query_matches)
    total_query_matching_words_in_page = np.array(all_relevants).sum()
    sorted_matches = query_matches[np.argsort(query_dists)]

    if total_query_matching_words_in_page < 1:
        return None, None, None

    # This makes it a bit faster with LLVM
    cum_mean = 0
    mAp_numer = 0
    total_matches = 0
    for j in range(sorted_matches.shape[0]):
        cum_mean = sorted_matches[j]*(1 / float(j + 1)) + (float(j) / (j+1))*cum_mean
        total_matches += sorted_matches[j]
        mAp_numer += cum_mean*sorted_matches[j]

    mAP = mAp_numer / (total_query_matching_words_in_page + eps)

    recall = query_matches.sum() / (float(total_query_matching_words_in_page) + eps)
    accuracy = query_matches.mean()

    return mAP, recall, accuracy


def query_page_phoc(query, page_dict, threshold, phoc=None, use_gt_phoc=False, filter_small=False, small_size=8):
    """
    Give a page, retrieve a list of words and evaluate average precision of a list given query word

    return:
        dists: list of distances of page words to query word
        matches: a list of {0, 1} indicating whether the word is matched by the algorithm
        retrived_word_idx: list of word idx
        matching_words_in_page: number of words in a page that match the query
    """
    if phoc is None:
        q_phoc, _ = phoc_letters_and_digits([query])
        q_phoc = q_phoc[0, :]
    else:
        q_phoc = phoc
    dists = []
    matches = []
    matching_words_in_page = 0
    retrived_word_idx = []

    for k in page_dict.keys():
        # Ground truth words that were assigned predictions by the model
        if 'word_' in k:
            word = page_dict[k]
            if use_gt_phoc:
                # Debug option to see how good is the segmentation assuming perfect PHOC
                word_phoc = word.get('gt_phoc')
            else:
                # If we have no prediction word this word return a zero PHOC
                word_phoc = word.get('pre_phoc', np.zeros(q_phoc.shape[0]))

            word_box = word['gt']
            if filter_small:
                if (word_box[2] - word_box[0] < small_size) or (word_box[3] - word_box[1] < small_size):
                    continue
            iou = word.get('cover', 0.0)
            dist = _dist_calc(q_phoc, word_phoc)
            dists.append(dist)

            is_this_the_q_word = (word['text'] == query)
            if not k.startswith('word_red'):
                # 'word_red' are redundant predictions generated by the model and are not part of the ground truth.
                matching_words_in_page += 1*is_this_the_q_word

            # A word considered a match if it's the query word and IoU is above threshold
            match = (is_this_the_q_word & (iou >= threshold))*1
            matches.append(match)
            retrived_word_idx.append(k)

    return dists, matches, retrived_word_idx, matching_words_in_page


@jit
def _dist_calc(q_phoc, w_phoc):
    """ """
    s = 0
    qsq = 10**-8
    psq = 10**-8
    for q, p in zip(q_phoc, w_phoc):
        s += q*p
        qsq += q**2
        psq += p**2
    q_norm = qsq**(0.5)
    p_norm = psq**(0.5)
    dist = 1 - s / (q_norm * p_norm)
    return dist


def phoc_spottig(it, folder, ignore_stop_words=False, iou_thresh=0.5, dont_load=False, use_gt_phoc=False, filter_small=False, logger=None, max_word_num=None):
    """
    Evaluate mAP for phoc based word spotting using an evaluation folder (contating *.json) produced by main.py script.
    assumes folder contains json files with {'word_%d': {'gt': gt_box, 'text': word annotation, 'cover': IoU of predicted word with GT box,
                                                          'gt_phoc': PHOC for text, 'pre_phoc': predicted PHOC}}

    Performs two tasks:
        (1) Prepare query words
        (2) Call query function on query words
    """
    # Create all query words - based on test set ground truth
    if logger is None:
        logger = print
    qwords = []
    for page in it:
        # Some of the datasest (e.g. IAMDB) have words with bad annotations, those are ignored by the eval protocol
        words = page.get_good_words_and_boxes_idx()[1]
        qwords.extend(words)
    qwords = set(qwords)

    logger('Query Words %d' % len(qwords))
    if ignore_stop_words:
        # If there are stop words to be removed...
        qwords = qwords - set(get_stop_words('en'))
        logger('Without stop words %d' % len(qwords))

    if max_word_num is not None:
        # Subsample all queries to partially evaluate
        if len(qwords) > max_word_num:
            idx = np.random.choice(range(len(qwords)), max_word_num, replace=False)
            qwords = set(np.array(list(qwords))[idx])
            logger('Sampled %d words' % len(qwords))

    qtimer = Timer()
    qtimer.tic()
    mAP, recall, accuracy = query_page_folder_phoc(qwords, folder, threshold=iou_thresh, dont_load=dont_load, use_gt_phoc=use_gt_phoc, filter_small=filter_small)
    logger('Finished after %d secs mAP %4.2f Recall %4.2f Accuracy %4.2f' % (qtimer.toc(), mAP*100, recall*100, accuracy*100))
    return


def mAP_stats(args):
    """
    Evaluation helper for mAP on word spotting task.
    The datasets we used are IAMDB and BOTANY.
    You can add additional datasets by changing eval_attrib below
    """

    if args.use_test_data:
        it = get_dataset_loader_by_name(args.dataset, 'test')
    else:
        it = get_dataset_loader_by_name(args.dataset, args.data_type)

    # Evaluation tuples [(boolean: ignore stop words?, boolean: filter small words (less than 8x8)]
    eval_attrib = {'botany': (False, False), 'iamdb': (True, True)}

    p = Path(args.experiment_dir)
    stop_words, filter_small = eval_attrib[args.dataset]
    logger = Logger(p)
    for iou in [0.25, 0.5]:
        logger('--- %s @ %3.2f ---' % (args.dataset, iou))
        phoc_spottig(it, str(p), ignore_stop_words=stop_words,
                     iou_thresh=iou, dont_load=True, use_gt_phoc=False, filter_small=filter_small, logger=logger)


def segment_eval_struct_output(args):
    folder = Path(args.eval_dir)
    for ds in ['icdar', 'iamdb', 'iclef']:
        ev_dir = folder.glob('*%s_*/**/eval' % ds)
        for ev in ev_dir:
            source = ev.parent.parent.stem.split('_')[0]
            target = ev.parent.parent.stem.split('_')[1]
            th = 0.9 if target == 'icdar' else 0.6
            print ('[%s -> %s %2.2f] %s' % (source, target, th, str(ev)))
            calculate_results_folder(str(ev), th)


def segment(args):
    """
    """
    folder = Path(args.eval_dir)
    print('%s [threshold %2.2f ]' % (str(folder), args.iou))
    calculate_results_folder(str(folder), threshold=args.iou)
