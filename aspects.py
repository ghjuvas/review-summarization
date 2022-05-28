import collections
import math
from typing import Tuple
from gensim import models
from gensim.corpora.dictionary import Dictionary

# default_path = 'apple-macbook-air-13.txt'
# with open(default_path, 'r', encoding='utf-8') as corpus_file:
#     default_corpus = corpus_file.read().split('\n\n')

# default_mode = 'TF-IDF + Discontinious collocations + Reference corpus Positive/Negative'


def frequencies(lemmas, threshold) -> dict:
    '''
    Get frequent lemmas or n-grams (seen more than some defined times)
    '''

    items = []
    for i in lemmas:
        items += i

    freq_dict = collections.Counter(items)
    freq_dict = dict(filter(lambda item: item[1] > threshold, freq_dict.items()))

    return freq_dict


# TF-IDF block
def tf(lemmas) -> dict:
    '''
    Calculate TF from lemmas.
    Get collections.Counter with frequencies
    of N-grams for each text.
    '''
    freq_dict = collections.Counter(lemmas)
    for i in freq_dict:
        freq_dict[i] = freq_dict[i]/float(len(lemmas))
    return freq_dict


def tf_idf(rev_lemmas, threshold) -> list:
    '''
    Calculate TF-IDF for each review in corpus.
    Get list of dicts with N-gram as key and tf-idf as value.
    '''
    full_tf_idf = []

    for rev in rev_lemmas:
        rev_tf_idf = {}

        # tf
        freq_dict = tf(rev)

        for item in freq_dict:
            # idf - logarithm for number of reviews
            # divided by number of texts/documents with target word.
            docs_target = sum([1.0 for r in rev_lemmas if item in r])
            res_idf = math.log10(len(rev_lemmas) / docs_target)
            rev_tf_idf[item] = freq_dict[item] * res_idf

        # the greater the value,
        # the more important the element

        top_tf_idf = dict(filter(lambda item: item[1] > threshold, rev_tf_idf.items()))
        full_tf_idf.append(top_tf_idf)

    return full_tf_idf


# weirdness block
def tf_w(reviews) -> Tuple[dict, int]:

    lemmas = []  # all lemmas
    for rev in reviews:
        lemmas += rev

    freq_dict = tf(lemmas)
    # print(freq_dict)

    return freq_dict, len(lemmas)


def weirdness(sphere_t: list, sphere_r: list) -> dict:
    '''
    Calculate weirdness value
    for each N-gram that occurs in both corpuses.
    '''

    tf_t, w_t = tf_w(sphere_t)
    tf_r, w_r = tf_w(sphere_r)

    # we can use unique lemmas
    # because we check their presence in collections
    full_lemmas = set(tf_t + tf_r)

    w_dict = {}  # weirdness for each item
    # t_list = []  # not used lemmas from target
    # r_list = []  # not used lemmas from reference
    for lem in full_lemmas:
        if lem in tf_t and lem in tf_r:
            weird = (tf_t[lem] / float(w_t)) / (tf_r[lem] / float(w_r))
            w_dict[lem] = weird
        # TODO do user need other lists?
        # else:
        #     if lem in tf_t:
        #         t_list.append(lem)
        #     elif lem in tf_r:
        #         r_list.append(lem)

    w_dict = dict(filter(lambda item: item[1] > 1, w_dict.items()))

    return w_dict


# chi-square
def chisq_obs(sphere_t: list, sphere_r: list) -> dict:
    '''
    Calculate value of chi-square test
    for each N-gram in both corpuses.
    '''

    len1 = float(len(sphere_t))  # num of documents from sphere 1
    len2 = float(len(sphere_r))  # num of documents from sphere 2

    reviews = sphere_t + sphere_r

    rev_N = len(reviews)  # num of documents in both corpuses

    lemmas = []  # all lemmas
    for rev in reviews:
        lemmas += rev

    chisq_obs = {}
    # we can use unique lemmas
    # because we just check their presence in collections
    for lemma in set(lemmas):

        tsd1 = sum([1.0 for rev in sphere_t if lemma in rev])  # refer/contain
        tsn1 = sum([1.0 for rev in sphere_r if lemma in rev])  # do not refer/contain

        tsd0 = len1 - tsd1  # refer/do not contain
        tsn0 = len2 - tsn1  # do not refer/do not contain

        value = (rev_N * (tsd1 * tsn0 - tsd0 * tsn1)) / (len1 * len2 * (tsd1 + tsn1) * (tsd0 + tsn0))

        chisq_obs[lemma] = value

    chisq = dict(filter(lambda item: item[1] > 1, chisq_obs.items()))

    return chisq


def tm_lda(lemmas: list, bigrams: list, trigrams: list):
    '''
    Get topics from documents with gensim implementation
    of Latent Dirichlet Allocation (LDA).
    '''

    new_corpus = []
    for doc_lem, doc_bigr, doc_trigr in zip(lemmas, bigrams, trigrams):
        new_doc = doc_lem + doc_bigr + doc_trigr
        new_corpus.append(new_doc)

    # print(new_corpus[0])
    dictionary = Dictionary(new_corpus)
    emb_corpus = [dictionary.doc2bow(text) for text in new_corpus]
    # print(len(new_corpus))
    # print(len(emb_corpus))

    # training parameters
    num_topics = 7
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None

    temp = dictionary[0]
    id2word = dictionary.id2token

    print('Training LDA model...')

    model = models.LdaModel(
        corpus=emb_corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    # top = model.top_topics(emb_corpus)
    print('Got topics...')
    # pprint(top)

    show = model.show_topics(formatted=False)
    # topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in show]
    topics_words = []
    
    for tp in show:
        d = {}
        for wd in tp[1]:
            d[wd[0]] = float(wd[1])
        topics_words.append(d)

    # print(topics_words)

    # words_list = [words for _, words in topics_words]

    return topics_words


def chisq_or_weird(sentiment: str, func: str,
                   other, **kwargs):

    if func == 'Chi-square_test':
        f = chisq_obs
    if func == 'Weirdness':
        f = weirdness

    if sentiment == 'Reference_corpus_Positive/Negative':  # without sentiment

        pos_neg = other
        # print(type(pos_neg))
        # print(len(pos_neg))
        # print(pos_neg)

        pos_lemmas = f(pos_neg[0], pos_neg[3])
        neg_lemmas = f(pos_neg[3], pos_neg[0])
        pos_bigrams = f(pos_neg[1], pos_neg[4])
        neg_bigrams = f(pos_neg[4], pos_neg[1])
        pos_trigrams = f(pos_neg[2], pos_neg[5])
        neg_trigrams = f(pos_neg[5], pos_neg[2])

        # print(pos_bigrams)
        # print(neg_bigrams)

        return pos_lemmas, pos_bigrams, pos_trigrams, neg_lemmas, neg_bigrams, neg_trigrams

    if sentiment == 'Reference_corpus_Camera_reviews':  # sentiment

        cam = other

        target_lemmas = f(kwargs['lemmas'], cam[0])
        target_bigrams = f(kwargs['bigrams'], cam[0])
        target_trigrams = f(kwargs['trigrams'], cam[0])

        return target_lemmas, target_bigrams, target_trigrams


def get_lda_polar(lemmas: list,
                  bigrams: list,
                  trigrams: list):
    '''
    Get words from LDA processing.
    '''

    topics = tm_lda(lemmas, bigrams, trigrams)

    words = []
    for v in topics:
        words.append(dict(filter(lambda x: x[0] not in ['не', 'ни', 'это'], v.items())))

    # print(words)

    return words


def extract_aspects(lemmas: list, bigrams: list, trigrams: list,  # standard
                    aspects: str, collocations: str, sentiment: str,  # modes
                    other):  # reference corpus or distribution by pos-neg
    '''
    Extract aspects by selected mode.
    '''

    print('Find aspects...')

    if aspects == 'TF-IDF':  # sentiment

        # FIXME optimization: zip for n-grams and thresholds
        if collocations == 'N-grams' or collocations == 'Patterns':
            threshold1 = 0.01
            threshold2 = 0.02
            threshold3 = 0.02
        if collocations == 'Discontinious_collocations':
            threshold1 = 0.01
            threshold2 = 0.002
            threshold3 = 0.001

        tf_idf_lemmas = tf_idf(lemmas, threshold1)
        tf_idf_bigrams = tf_idf(bigrams, threshold2)
        tf_idf_trigrams = tf_idf(trigrams, threshold3)

        # # if all values of bigrams or trigrams are similar
        # # this distribution is unrelevant
        # tf_idf_bigrams = list(filter(lambda i: len(set(i.values())) != 1, tf_idf_bigrams))
        # # print(len(tf_idf_bigrams))
        # tf_idf_trigrams = list(filter(lambda i: len(set(i.values())) != 1, tf_idf_trigrams))
        # # print(len(tf_idf_trigrams))

        # # get sets of N-grams
        # sets_ngrams = []
        # for tf_idf_list in [tf_idf_lemmas, tf_idf_bigrams, tf_idf_trigrams]:
        #     final_list = []
        #     for rev in tf_idf_list:
        #         final_list += list(rev)
        #     sets_ngrams.append(set(final_list))

        # # print('Итоговые леммы', sets_ngrams[0])
        # # print('Итоговые биграммы', sets_ngrams[1])
        # # print('Итоговые триграммы', sets_ngrams[2])

        # set_lemmas = sets_ngrams[0]
        # set_bigrams = sets_ngrams[1]
        # set_trigrams = sets_ngrams[2]

        # TODO filter bigrams by lemmas, trigrams by bigrams?

        # print('Леммы по tf-idf', tf_idf_lemmas)
        # print('Биграммы по tf-idf', tf_idf_bigrams)
        # print('Триграммы по tf-idf', tf_idf_trigrams)

        return tf_idf_lemmas, tf_idf_bigrams, tf_idf_trigrams

    if aspects == 'Frequencies':  # sentiment

        freq_lemmas = frequencies(lemmas, 3)
        # print(freq_lemmas)
        freq_bigrams = frequencies(bigrams, 1)
        # print(freq_bigrams)
        freq_trigrams = frequencies(trigrams, 1)
        # print(freq_trigrams)

        # print('Леммы с высокой частотностью появления', freq_lemmas)
        # print('Биграммы с высокой частотностью появления', freq_bigrams)
        # print('Триграммы с высокой частотностью появления', freq_trigrams)

        return freq_lemmas, freq_bigrams, freq_trigrams

    if aspects in ['Chi-square_test', 'Weirdness']:  # 50 / 50

        res = chisq_or_weird(
            sentiment, aspects,
            other,
            lemmas=lemmas, bigrams=bigrams, trigrams=trigrams)

        return res

    if aspects == 'LDA':  # without sentiment

        pos_neg = [part for part in other]

        pos_topics = get_lda_polar(*pos_neg[:3])
        neg_topics = get_lda_polar(*pos_neg[3:])

        return pos_topics, neg_topics

    if aspects == 'Statistic_collocations':

        new_lemmas = []
        for rev in lemmas:
            for lem in rev:
                new_lemmas.append(lem)
        c = collections.Counter(new_lemmas)
        lemmas = [lem for lem, _ in c.most_common(60)]

        return lemmas, bigrams, trigrams

    return 'Something went wrong!'
