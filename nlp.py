from string import punctuation
import re
from typing import Tuple
import jamspell
import stanza
import nltk
from nltk import collocations

from nltk.corpus import stopwords
rus_sw = list(stopwords.words('russian'))
# can improve sentiment valuation
for i in ['не', 'нет', 'ни']:
    rus_sw.remove(i)
rus_sw += ['достоинство', 'недостаток', 'плюс', 'минус', 'комментарий', 'отзыв']

# patterns for sentiment parts extraction
# see pos_neg_corpus and split_sent_parts for more
# order of review parts
str_plus = 'Плюсы'  # first type
str_minus = 'Минусы'
str_review = 'Отзыв'
str_adv = 'Достоинства'  # second type
str_disadv = 'Недостатки'
str_comm = 'Комментарий'

# parsing 'Достоинства Очень стильный дизайн...'
pattern_plus = '\s*' + str_plus + '\s+'
pattern_minus = '\s*' + str_minus + '\s+'
pattern_review = '\s*' + str_review + '\s+'
pattern_adv = '\s*' + str_adv + '\s+'
pattern_disadv = '\s*' + str_disadv + '\s+'
pattern_comm = '\s*' + str_comm + '\s+'

# parsing 'Достоинства: Очень стильный дизайн...'
pattern_plus_comma = '\s*' + str_plus + ':' + '\s+'
pattern_minus_comma = '\s*' + str_minus + ':' + '\s+'
pattern_review_comma = '\s*' + str_review + ':' + '\s+'
pattern_adv_comma = '\s*' + str_adv + ':' + '\s+'
pattern_disadv_comma = '\s*' + str_disadv + ':' + '\s+'
pattern_comm_comma = '\s*' + str_comm + ':' + '\s+'

# print(str_plus)
# print(pattern_plus)
# print(pattern_plus_comma)


def camera_reviews(sp):

    # reference camera corpus
    with open('camera.txt', 'r', encoding='utf-8-sig') as camera_file:
        camera = camera_file.read()

    # a little processing for better split
    camera = camera.replace('\nФотоаппарат', 'Фотоаппарат')
    camera = camera.replace(
        '(оптика) Недостатки:',
        '(оптика)\nНедостатки:'
        )

    cam_reviews = re.split('\s*\n+', camera)
    print('Camera reviews length:', len(cam_reviews))

    print('Got reference camera corpus...')

    if sp:
        cam_reviews = spellchecking(cam_reviews)
        print('Camera reviews were corrected!')

    return cam_reviews


def spellchecking(reviews):
    '''
    Spell correction with jamspell.
    '''

    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('ru_small.bin')

    reviews = [corrector.FixFragment(rev) for rev in reviews]
    print('Number of revs after spellcorrection', len(reviews))

    # print(reviews[0])

    return reviews


def split_sent_parts(rev: str, p_plus: str, p_minus: str,
                     p_comm: str, match_comm: bool):
    '''
    Split review texts by sentiment parts.
    '''

    rev_wo_plus = re.sub(p_plus, '', rev)
    plus, minus = re.split(p_minus, rev_wo_plus)

    if match_comm:
        minus, _ = re.split(p_comm, minus)

    return plus, minus


def pos_neg_corpus(reviews: list) -> Tuple[list, list]:
    '''
    Get two corpuses from original review corpus
    by polarity of opinion.
    '''

    # print('Всего отзывов', len(reviews))

    # texts by pluses/minuses
    pos = []
    neg = []

    count_undesirable = 0

    for rev in reviews:
        # find relevant parts by regex
        # parts must start with 'Достоинства', 'Плюсы', etc

        match_plus = re.search(pattern_plus, rev)
        match_minus = re.search(pattern_minus, rev)
        match_review = re.search(pattern_review, rev)
        match_adv = re.search(pattern_adv, rev)
        match_disadv = re.search(pattern_disadv, rev)
        match_comm = re.search(pattern_comm, rev)

        match_plus_comma = re.search(pattern_plus_comma, rev)
        match_adv_comma = re.search(pattern_adv_comma, rev)
        match_minus_comma = re.search(pattern_minus_comma, rev)
        match_review_comma = re.search(pattern_review_comma, rev)
        match_disadv_comma = re.search(pattern_disadv_comma, rev)
        match_comm_comma = re.search(pattern_comm_comma, rev)

        # FIXME optimization: duplicated code for parts append
        # first type
        if match_plus and match_minus:
            plus, minus = split_sent_parts(rev,
                                           pattern_plus,
                                           pattern_minus,
                                           pattern_review,
                                           match_review)

            pos.append(plus)
            neg.append(minus)

        elif match_plus_comma and match_minus_comma:
            plus, minus = split_sent_parts(rev,
                                           pattern_plus_comma,
                                           pattern_minus_comma,
                                           pattern_review_comma,
                                           match_review_comma)

            pos.append(plus)
            neg.append(minus)

        # second type
        elif match_adv and match_disadv:
            plus, minus = split_sent_parts(rev,
                                           pattern_adv,
                                           pattern_disadv,
                                           pattern_comm,
                                           match_comm)

            pos.append(plus)
            neg.append(minus)

        elif match_adv_comma and match_disadv_comma:
            plus, minus = split_sent_parts(rev,
                                           pattern_adv_comma,
                                           pattern_disadv_comma,
                                           pattern_comm_comma,
                                           match_comm_comma)

            pos.append(plus)
            neg.append(minus)

        else:
            count_undesirable += 1
            # print(rev)

    print(f'From all {len(reviews)} reviews {count_undesirable} texts can not be separated')

    return pos, neg


def pos_neg(reviews, collocations, sw):

    print('Achieving positive and negative corpuses')
    pos, neg = pos_neg_corpus(reviews)

    pos_lemmas = preprocessing(pos, collocations, sw)
    neg_lemmas = preprocessing(neg, collocations, sw)

    if collocations == 'Discontinious_collocations':
        pos_bigrams, pos_trigrams = get_discont_collocations(pos_lemmas)
        neg_bigrams, neg_trigrams = get_discont_collocations(neg_lemmas)
    if collocations == 'Statistic_collocations':
        pos_bigrams, pos_trigrams = get_stat_collocations(pos_lemmas)
        neg_bigrams, neg_trigrams = get_stat_collocations(neg_lemmas)
    if collocations == 'N-grams':
        pos_bigrams, pos_trigrams = get_ngrams(pos_lemmas)
        neg_bigrams, neg_trigrams = get_ngrams(neg_lemmas)
    if collocations == 'Patterns':
        pos_unigrams, pos_bigrams, pos_trigrams = get_colls_from_patterns(pos_lemmas, sw)
        neg_unigrams, neg_bigrams, neg_trigrams = get_colls_from_patterns(neg_lemmas, sw)

        return pos_unigrams, pos_bigrams, pos_trigrams, neg_unigrams, neg_bigrams, neg_trigrams

    # print('Биграммы позитивные', pos_bigrams)
    # print('Триграммы негативные', neg_trigrams)

    print('End of preprocessing.')

    # print(pos_lemmas)

    return pos_lemmas, pos_bigrams, pos_trigrams, neg_lemmas, neg_bigrams, neg_trigrams


def get_reviews(corpus: str, sp: bool):
    '''
    Open file with corpus and get list of reviews.
    Texts must be separated by two blank lines.
    Processing of reference camera corpus is independent.
    '''

    # TODO user decide about both corpuses?

    with open(corpus, 'r', encoding='utf-8') as corpus_file:
        reviews = corpus_file.read().split('\n\n')
        reviews = [rev for rev in reviews if rev != '']
        reviews = list(set(reviews))
        if len(reviews) in [0, 1]:
            raise ValueError('Not enough reviews to analyze!')
        print('Target corpus length:', len(reviews))

    # spelling correction if required
    if sp:
        reviews = spellchecking(reviews)
        print('Reviews were corrected!')

        return reviews

    print('Got corpus...')

    return reviews


def stanza_nlp(reviews: list) -> list:
    '''
    Get parsing of texts with stanza.
    Get list with docs for each text.
    '''
    print('Parse morphosyntax...')

    nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma')

    # list of parsing for each review
    docs = []  # save parsing to other tasks
    for rev in reviews:
        # print(rev)
        doc = nlp(rev)
        docs.append(doc)

    return docs


def stanza_get_lemma(docs: list, sw: bool) -> list:
    '''
    Get lemmas from texts with stanza parsing.
    Get list with lemmas for each text.
    '''
    # lemmas for each review in corpus
    lemmas = []
    info = []

    for doc in docs:
        rev_lemmas = []
        rev_info = []
        for sent in doc.sentences:
            for word, token in zip(sent.words, sent.tokens):
                # FIXME optimization: duplicated code
                if sw:
                    if word.lemma not in rus_sw:
                        if word.lemma not in punctuation and word.lemma != '–':
                            rev_info.append((
                                word.lemma.lower(), word.text,
                                token.start_char, token.end_char
                                ))
                            rev_lemmas.append(word.lemma.lower())
                else:
                    if word.lemma not in punctuation and word.lemma != '–':
                        rev_info.append((
                            word.lemma.lower(), word.text,
                            token.start_char, token.end_char
                            ))
                        rev_lemmas.append(word.lemma.lower())
        # if rev_lemmas not in lemmas:
        lemmas.append(rev_lemmas)
        info.append(rev_info)

    print('Got lemmas and info from corpus...')
    print('Length of lemma corpus', len(lemmas))

    return lemmas


def stanza_get_lemma_morph(docs, sw):

    rev_lemmas_pos = []

    for doc in docs:  # каждый отзыв

        lemmas_pos = []  # собираем лемму, тег и морфологию
        for sent in doc.sentences:
            for word in sent.words:
                # FIXME duplicated code
                if sw:
                    if word.lemma not in punctuation and word.lemma not in rus_sw:
                        # check pos tags
                        if word.feats:  # if got tags
                            final_feats = {}
                            if 'Number' in word.feats:
                                # print('Found number', word.feats)
                                number_match = re.search(
                                    'Number=(\w+)', word.feats)
                                final_feats['Number'] = number_match.group(1)

                            if 'Case' in word.feats:
                                # print('Found case', word.feats)
                                case_match = re.search(
                                    'Case=(\w+)', word.feats)
                                final_feats['Case'] = case_match.group(1)
                        else:  # default if nothing
                            final_feats = '_'
                        # print(final_feats)
                        # append
                        lemmas_pos.append((
                            word.lemma.lower(),
                            word.upos, final_feats))
                else:
                    if word.lemma not in punctuation:
                        # check pos tags
                        if word.feats:  # if got tags
                            final_feats = {}
                            if 'Number' in word.feats:
                                # print('Found number', word.feats)
                                number_match = re.search(
                                    'Number=(\w+)', word.feats)
                                final_feats['Number'] = number_match.group(1)

                            if 'Case' in word.feats:
                                # print('Found case', word.feats)
                                case_match = re.search(
                                    'Case=(\w+)', word.feats)
                                final_feats['Case'] = case_match.group(1)
                        else:  # default if nothing
                            final_feats = '_'
                        # print(final_feats)
                        # append
                        lemmas_pos.append((
                            word.lemma.lower(),
                            word.upos, final_feats))

        rev_lemmas_pos.append(lemmas_pos)

    # print(rev_lemmas_pos[0])

    return rev_lemmas_pos


def get_discont_collocations(reviews):
    '''
    Get discontinious collocations with nltk.collocations.
    '''

    discont_bigrams = []
    discont_trigrams = []

    bigram_measures = collocations.BigramAssocMeasures()
    trigram_measures = collocations.TrigramAssocMeasures()

    for rev in reviews:
        # леммы для каждого отзыва
        # print(rev)
        finder_2_win = collocations.BigramCollocationFinder.from_words(rev, window_size=3)
        finder_3_win = collocations.TrigramCollocationFinder.from_words(rev, window_size=3)

        # min_score 0 to choose all
        found2 = list(finder_2_win.above_score(bigram_measures.likelihood_ratio, 0))
        found3 = list(finder_3_win.above_score(trigram_measures.likelihood_ratio, 0))
        found2 = [' '.join(f) for f in found2]
        found3 = [' '.join(f) for f in found3]
        discont_bigrams.append(found2)
        discont_trigrams.append(found3)

    # print('Discont bigrams', discont_bigrams)
    # print('Discont trgirams', discont_trigrams)

    return discont_bigrams, discont_trigrams


def get_stat_collocations(reviews) -> Tuple[list, list]:
    '''
    Get statistic collocations with nltk.collocations.
    Use likelihood ratio to search for collocations.
    '''
    # full lemmas
    # this is not a best solution
    # because now we can take bigrams between documents
    lemmas = []
    for ll in reviews:
        lemmas += ll

    # we can take fourgrams
    # but ngrams have just 2 options
    bigram_measures = collocations.BigramAssocMeasures()
    trigram_measures = collocations.TrigramAssocMeasures()

    # here it is available to choose from_documents
    # but this method do not have window_size argument
    finder_2_win = collocations.BigramCollocationFinder.from_words(lemmas, window_size=5)
    finder_3_win = collocations.TrigramCollocationFinder.from_words(lemmas, window_size=5)

    # nbest 55 on bigrams and 60 on trigrams
    # on the mac reviews
    coll2 = finder_2_win.nbest(bigram_measures.likelihood_ratio, 55)
    coll3 = finder_3_win.nbest(trigram_measures.likelihood_ratio, 60)
    coll2 = [' '.join(f) for f in coll2]
    coll3 = [' '.join(f) for f in coll3]
    # print(type(coll2[0]))
    # print(type(coll3[0]))

    return coll2, coll3


def get_ngrams(reviews):
    '''
    Get continious N-grams
    (unigrams, bigrams, trigrams) from lemmas.
    '''

    bigrams = []
    trigrams = []
    # separated documents
    for rev in reviews:
        bigrams.append([f'{word_1} {word_2}' for word_1, word_2 in list(nltk.bigrams(rev))])
        trigrams.append([f'{word_1} {word_2} {word_3}' for word_1, word_2, word_3 in list(nltk.trigrams(rev))])

    return bigrams, trigrams


def check_pos_tags(first_word, second_word):
    '''
    Check POS-tags compatibility.
    '''

    if first_word[1] == 'NOUN':
        if type(second_word[2]) == dict:
            if 'Case' in second_word[2]:
                if second_word[1] == 'NOUN' and second_word[2]['Case'] != 'Gen':
                    # print('Not gen')
                    return False
            if 'Number' in second_word[2] and 'Number' in first_word[2]:
                if second_word[1] == 'ADJ' and second_word[2]['Number'] != first_word[2]['Number']:
                    # print('NOUN, but ADJ not agreed')
                    return False

    elif first_word[1] == 'ADJ':
        # print(second_word[2])
        if 'Number' in second_word[2] and 'Number' in first_word[2]:
            if second_word[1] == 'NOUN' and second_word[2]['Number'] != first_word[2]['Number']:
                # print('NOUN, but ADJ not agreed')
                return False

    return True


def search_from_patterns(t_id, rev, bigram,
                         bigrams_templates,
                         trigrams_templates):
    '''
    Compose and find n-grams from patterns.
    '''

    if t_id + 1 <= len(rev) - 1:
        if bigram:
            first_word = rev[t_id]
            second_word = rev[t_id + 1]
            template = first_word[1] + '+' + second_word[1]
            if template in bigrams_templates:
                # print(template)
                # print('Found bigram in patterns')
                res = check_pos_tags(first_word, second_word)
                if res:
                    # print('Bigram was checked!')
                    gram = first_word[0] + ' ' + second_word[0]
                    return gram

    if t_id + 2 <= len(rev) - 1:
        if bigram is False:
            first_word = rev[t_id]
            second_word = rev[t_id + 1]
            third_word = rev[t_id + 2]
            template = first_word[1] + '+' + second_word[1] + '+' + third_word[1]
            if template in trigrams_templates:
                # print(template)
                # print('Found trigram in patterns')
                res_1 = check_pos_tags(first_word, second_word)
                res_2 = check_pos_tags(second_word, third_word)
                if res_1 and res_2:
                    # print('Trigram was checked!')
                    gram = first_word[0] + ' ' + second_word[0] + ' ' + third_word[0]
                    return gram


def get_colls_from_patterns(reviews, sw):

    pos_list = ['NOUN', 'ADJ', 'ADP']  # not unigrams just check function
    uni_list = ['NOUN', 'ADJ', 'ADV']
    bigrams_templates = [
        'ADJ+NOUN', 'NOUN+ADJ',
        'NOUN+NOUN', 'ADV+VERB', 'VERB+ADV']
    trigrams_templates = [
        'ADJ+ADJ+NOUN', 'ADJ+NOUN+NOUN',
        'ADJ+ADP+NOUN', 'NOUN+ADP+NOUN',
        'ADV+VERB+NOUN', 'NOUN+VERB+ADV',
        'NOUN+ADV+VERB']

    if sw:
        pos_list.remove('ADP')
        trigrams_templates.remove('ADJ+ADP+NOUN')
        trigrams_templates.remove('NOUN+ADP+NOUN')

    final_unigrams = []
    final_bigrams = []
    final_trigrams = []

    for rev in reviews:
        unigrams = []
        bigrams = []
        trigrams = []
        for t_id, lemma_pos in enumerate(rev):

            pos_to_check = lemma_pos[1]
            if pos_to_check in uni_list:
                unigrams.append(lemma_pos[0])
            if pos_to_check in pos_list:

                # bigram search
                bigram = search_from_patterns(t_id, rev, True,
                                              bigrams_templates,
                                              trigrams_templates)
                if bigram:
                    # print('Found bigram', chunk)
                    bigrams.append(bigram)

                # trigram search (independent from bigram!)
                trigram = search_from_patterns(t_id, rev, False,
                                               bigrams_templates,
                                               trigrams_templates)
                if trigram:
                    # print('Found trigram', chunk)
                    trigrams.append(trigram)

        final_unigrams.append(unigrams)
        final_bigrams.append(bigrams)
        final_trigrams.append(trigrams)

    # print(final_unigrams)

    return final_unigrams, final_bigrams, final_trigrams


def processing(corpus, collocations, sentiment, sp, sw):
    '''
    Main function to preprocessing reviews.
    Aggregate all of the processing functions
    like morphosyntax parsing and collocation search.
    '''

    # corpus - filename
    # mode - make needed collocations and process reference corpus
    # sw - remove stopwords (default true)
    # sp - spellchecking (default true)

    reviews = get_reviews(corpus, sp)

    # preprocessing
    lemmas = preprocessing(reviews, collocations, sw)

    # collocations
    if collocations == 'Discontinious_collocations':
        bigrams, trigrams = get_discont_collocations(lemmas)
        print('Discontinious_collocations')
    if collocations == 'Statistic_collocations':
        bigrams, trigrams = get_stat_collocations(lemmas)
        print('Statistic_collocations')
    if collocations == 'N-grams':
        bigrams, trigrams = get_ngrams(lemmas)
        print('N-grams')
    if collocations == 'Patterns':
        lemmas, bigrams, trigrams = get_colls_from_patterns(lemmas, sw)
        print('Patterns')

    # FIXME optimization: duplicated code

    # sentiment
    if sentiment == 'Reference_corpus_Positive/Negative':
        pos_lemmas, pos_bigrams, pos_trigrams, neg_lemmas, neg_bigrams, neg_trigrams = pos_neg(reviews, collocations, sw)

        # print('Оригинальные позитивные негативные', pos_neg_ngrams)

        return lemmas, bigrams, trigrams, pos_lemmas, pos_bigrams, pos_trigrams, neg_lemmas, neg_bigrams, neg_trigrams

    if sentiment == 'Reference_corpus_Camera_reviews':
        cam_reviews = camera_reviews(sp)
        cam_lemmas = preprocessing(cam_reviews, collocations, sw)

        if collocations == 'Discontinious_collocations':
            cam_bigrams, cam_trigrams = get_discont_collocations(cam_lemmas)
        if collocations == 'N-grams':
            cam_bigrams, cam_trigrams = get_ngrams(cam_lemmas)
        if collocations == 'Patterns':
            cam_bigrams, cam_trigrams = get_colls_from_patterns(cam_lemmas, sw)

        return lemmas, bigrams, trigrams, cam_lemmas, cam_bigrams, cam_trigrams

    # print('Биграммы', bigrams)
    # print('Триграммы', trigrams)

    print('End of preprocessing.')

    return lemmas, bigrams, trigrams


def preprocessing(reviews, collocations, sw):
    '''
    Tokenization, lemmatization,
    cleaning for stopwords if required.
    '''
    print('Preprocessing...')

    # tokenization and morphosyntax parsing
    docs = stanza_nlp(reviews)
    # print('First docs', docs[0])

    # lemmatization
    if collocations == 'Patterns':
        lemmas_morph = stanza_get_lemma_morph(docs, sw)

        return lemmas_morph

    else:
        lemmas = stanza_get_lemma(docs, sw)
        # print('Lemmas after preprocessing', lemmas)
        # print(type(lemmas))
        # print('First lemmas', lemmas[0])

        return lemmas
