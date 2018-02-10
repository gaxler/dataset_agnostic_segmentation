from string import ascii_lowercase

from phoc import build_phoc
from lib.helpers import clean_word

letters_and_digits_unigrams = map(str, [c for c in ascii_lowercase] + range(10))


def phoc_letters_and_digits(word_list, unigram_levels_list=None):
    """
    Construct PHOC, follows the construction of PHOCNet

    Sudholt, Sebastian, and Gernot A. Fink. "PHOCNet: A deep convolutional neural network for word spotting in handwritten documents."
    In Frontiers in Handwriting Recognition (ICFHR), 2016 15th International Conference on, pp. 277-282. IEEE, 2016.
    """

    if unigram_levels_list is None:
        unigram_levels_list = range(1, 6)

    word_list = [clean_word(w) for w in word_list]

    phocs = build_phoc(word_list, phoc_unigrams=letters_and_digits_unigrams, unigram_levels=unigram_levels_list)
    phoc_dim = len(letters_and_digits_unigrams) * sum(unigram_levels_list)

    return phocs, phoc_dim