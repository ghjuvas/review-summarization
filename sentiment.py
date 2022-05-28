from collections import Counter

# from nlp import processing
# from aspects import extract_aspects


def check_polar_freq(selected, pos_corpus, neg_corpus):
    '''
    Check frequency of ngram in two polar corpuses.
    '''
    # pos_ngrams = []
    # neg_ngrams = []

    # weirdness, chi-square, lda have pos|neg option

    # frequencies, camera reviews
    if isinstance(selected, dict):
        # print('DICTIONARY')
        pos_dict = {}
        neg_dict = {}
        for ngram in selected:
            v_pos = pos_corpus.get(ngram)
            v_neg = neg_corpus.get(ngram)
            if v_pos > 0 or v_neg > 0:
                if v_pos >= v_neg:
                    pos_dict[ngram] = selected[ngram]
                else:
                    neg_dict[ngram] = selected[ngram]

        return pos_dict, neg_dict

    # tf-idf, lda
    if isinstance(selected, list):
        if isinstance(selected[0], dict):
            # print('TF-IDF or LDA')
            pos_list = []
            neg_list = []
            for rev in selected:
                # print(rev)
                pos_dict = {}
                neg_dict = {}
                for ngram in rev:  # даже если дикт
                    v_pos = pos_corpus.get(ngram)
                    v_neg = neg_corpus.get(ngram)
                    if v_pos > 0 or v_neg > 0:
                        if v_pos >= v_neg:
                            pos_dict[ngram] = rev[ngram]
                        else:
                            neg_dict[ngram] = rev[ngram]
                pos_list.append(pos_dict)
                neg_list.append(neg_dict)

            return pos_list, neg_list

        else:
            # print('LIST')
            pos_list = []
            neg_list = []
            for ngram in selected:
                v_pos = pos_corpus.get(ngram)
                v_neg = neg_corpus.get(ngram)
                if v_pos > 0 or v_neg > 0:
                    if v_pos >= v_neg:
                        pos_list.append(ngram)
                    else:
                        neg_list.append(ngram)

            return pos_list, neg_list

    return 'Something went wrong!'


def from_corpus(ngrams, selected):
    '''
    Get ngram sentiment from two polar corpuses.
    '''

    # print('Позитивные негативные проверяем', ngrams[0])
    # get counter of all original ngrams in corpuses
    # print(len(ngrams))
    counters = []
    for idx, corp in enumerate(ngrams):
        # print(corp[0])

        # all n-grams for polarity
        # собираем все позитивные леммы
        new_corpus = []
        for rev in corp:
            new_corpus += rev
        # n-gram frequency in parts
        # print(new_corpus)
        c = Counter(new_corpus)
        # считаем их частотности
        # print(c)

        if isinstance(selected[0][0], dict):
            # приходит
            # [{}, {}]
            small_c = {}
            if idx in [0, 3]:
                for dict_lem in selected[0]:
                    # словарь лемм
                    for lem in dict_lem:
                        # сами леммы
                        small_c[lem] = c.get(lem, 0)
                        # если не видим c в позитивных
                        # записываем как 0
            if idx in [1, 4]:
                for dict_lem in selected[1]:
                    for lem in dict_lem:
                        small_c[lem] = c.get(lem, 0)
            if idx in [2, 5]:
                for dict_lem in selected[2]:
                    for lem in dict_lem:
                        small_c[lem] = c.get(lem, 0)
            # print(small_c)
        else:
            # stat colls
            small_c = {}
            if idx in [0, 3]:
                for lem in selected[0]:
                    small_c[lem] = c.get(lem, 0)
            if idx in [1, 4]:
                for lem in selected[1]:
                    small_c[lem] = c.get(lem, 0)
            if idx in [2, 5]:
                for lem in selected[2]:
                    small_c[lem] = c.get(lem, 0)

        counters.append(small_c)

    # print('Pos lemmas', counters[0])
    # print('Pos bigrams', counters[1])
    # print('Neg lemmas', counters[3])

    # print(type(selected))
    # print(len(selected))
    # print(selected[0])
    pos_lemmas, neg_lemmas = check_polar_freq(selected[0],
                                              counters[0],
                                              counters[3])
    pos_bigrams, neg_bigrams = check_polar_freq(selected[1],
                                                counters[1],
                                                counters[4])
    pos_trigrams, neg_trigrams = check_polar_freq(selected[2],
                                                  counters[2],
                                                  counters[5])

    # print('Леммы, которые чаще появились в позитивных частях', pos_lemmas)
    # print('Леммы, которые чаще появились в негативных частях', neg_lemmas)
    # print('Биграммы, которые чаще появились в позитивных частях', pos_bigrams)
    # print('Биграммы, которые чаще появились в негативных частях', neg_bigrams)
    # print('Триграммы, которые чаще появились в позитивных частях', pos_trigrams)
    # print('Триграммы, которые чаще появились в негативных частях', neg_trigrams)

    return pos_lemmas, pos_bigrams, pos_trigrams, neg_lemmas, neg_bigrams, neg_trigrams


def check_2lemmas_sent(value_1, value_2):
    '''
    Check sentiment of two lemmas.
    '''

    sent_1 = value_1[2]
    sent_2 = value_2[2]
    sent_list = [sent_1, sent_2]
    # print(sent_list)
    if sent_1 == sent_2:
        # print('Тональности равны')
        sent = value_1[2]
    else:
        if 'positive' in sent_list:
            sent = 'positive'
        elif 'negative' in sent_list and 'neutral' in sent_list:
            sent = 'negative'
        elif 'negative' in sent_list:
            sent = 'negative'
        elif 'neutral' in sent_list:
            sent = 'neutral'

    return sent


def from_lexicon(lemmas, bigrams, trigrams):
    '''
    Get ngram sentiment from sentiment lexicon (provided by RuSentiLex-2017).
    '''

    # sentiment lexicon file
    with open('rusentilex_2017.txt', 'r', encoding='utf-8') as sl_file:
        sl = sl_file.read()

    sentilex = {}
    sl = sl.split('\n')[19:-1]
    for row in sl:
        splited = row.split(', ')
        sentilex[splited[2]] = splited[:2] + splited[3:]
    #   sentilex = sl_file.read().split('\n')[19:]

        # text pos lemma sentiment type (disambiguation)

    if isinstance(lemmas, dict):
        # lemma - just check sentiment
        pos_lemmas = {}
        neg_lemmas = {}
        for lemma in lemmas:
            value = sentilex.get(lemma, False)
            if value:
                sent = value[2]
            else:
                sent = 'neutral'

            if sent == 'positive':
                pos_lemmas[lemma] = lemmas[lemma]
            if sent == 'negative':
                neg_lemmas[lemma] = lemmas[lemma]

        # print(pos_lemmas)
        # print(neg_lemmas)

        # bigram - check sentiment and lemma entries sentiment
        pos_bigrams = {}
        neg_bigrams = {}
        for bigram in bigrams:
            value = sentilex.get(bigram, False)
            if value:
                # print('У целого выражения есть тональность!')
                sent = value[2]
                # print(sent)
            else:
                # print('У целого выражения тональности нет:')
                lemma_1, lemma_2 = bigram.split()
                value_1 = sentilex.get(lemma_1, False)
                value_2 = sentilex.get(lemma_2, False)
                if value_1 and value_2:
                    # print('Есть тональность у обоих составляющих')
                    # print(bigram)
                    sent = check_2lemmas_sent(value_1, value_2)
                elif value_1:
                    # print('Тональность только у первого')
                    sent = value_1[2]
                elif value_2:
                    # print('Тональность только у второго')
                    sent = value_2[2]
                else:
                    # print('Вообще ничего не нашли')
                    sent = 'neutral'

            if sent == 'positive':
                pos_bigrams[bigram] = bigrams[bigram]
            if sent == 'negative':
                neg_bigrams[bigram] = bigrams[bigram]

        # print(pos_bigrams)
        # print(neg_bigrams)

        # trigram - check sentiment and lemma entries sentiment
        pos_trigrams = {}
        neg_trigrams = {}
        for trigram in trigrams:
            value = sentilex.get(trigram, False)
            if value:
                # print('У целого выражения есть тональность!')
                sent = value[2]
                # print(trigram)
            else:
                # print('У целого выражения тональности нет:')
                lemma_1, lemma_2, lemma_3 = trigram.split()
                value_1 = sentilex.get(lemma_1, False)
                value_2 = sentilex.get(lemma_2, False)
                value_3 = sentilex.get(lemma_3, False)
                if value_1 and value_2 and value_3:
                    # print('Есть тональность у всех составляющих')
                    # print(trigram)
                    sent_1 = value_1[2]
                    sent_2 = value_2[2]
                    sent_3 = value_3[2]
                    sent_list = [sent_1, sent_2, sent_3]
                    # print(sent_list)
                    if sent_1 == sent_2 == sent_3:
                        # print('Тональности равны')
                        sent = value_1[2]
                    else:
                        c = Counter(sent_list)
                        pos = c.get('positive', False)
                        neg = c.get('negative', False)
                        neut = c.get('neutral', False)
                        if pos and neg and neut:
                            sent = 'positive'
                        else:
                            if pos:
                                if pos > 1:
                                    sent = 'positive'
                            if neg:
                                if neg > 1:
                                    sent = 'negative'
                            if neut:
                                if neut > 1:
                                    sent = 'neutral'

                elif value_1 and value_2:
                    sent = check_2lemmas_sent(value_1, value_2)
                elif value_2 and value_3:
                    sent = check_2lemmas_sent(value_2, value_3)
                elif value_1 and value_3:
                    sent = check_2lemmas_sent(value_1, value_3)

                elif value_1:
                    # print('Тональность только у первого')
                    sent = value_1[2]
                elif value_2:
                    # print('Тональность только у второго')
                    sent = value_2[2]
                elif value_3:
                    # print('Тональность только у третьего')
                    sent = value_3[2]

                else:
                    # print('Вообще ничего не нашли')
                    sent = 'neutral'

            if sent == 'positive':
                pos_trigrams[trigram] = trigrams[trigram]
            if sent == 'negative':
                neg_trigrams[trigram] = trigrams[trigram]

        # print(pos_trigrams)
        # print(neg_trigrams)

    if isinstance(lemmas, list):
        if isinstance(lemmas[0], dict):
            # lemma - just check sentiment
            pos_lemmas = []
            neg_lemmas = []
            for rev in lemmas:
                pos_rev = {}
                neg_rev = {}
                for lemma in rev:
                    value = sentilex.get(lemma, False)
                    if value:
                        sent = value[2]
                    else:
                        sent = 'neutral'

                    if sent == 'positive':
                        pos_rev[lemma] = rev[lemma]
                    if sent == 'negative':
                        neg_rev[lemma] = rev[lemma]

                pos_lemmas.append(pos_rev)
                neg_lemmas.append(neg_rev)

            # print(pos_lemmas)
            # print(neg_lemmas)

            # bigram - check sentiment and lemma entries sentiment
            pos_bigrams = []
            neg_bigrams = []
            for rev in bigrams:
                pos_rev = {}
                neg_rev = {}
                for bigram in rev:
                    value = sentilex.get(bigram, False)
                    if value:
                        # print('У целого выражения есть тональность!')
                        sent = value[2]
                        # print(sent)
                    else:
                        # print('У целого выражения тональности нет:')
                        lemma_1, lemma_2 = bigram.split()
                        value_1 = sentilex.get(lemma_1, False)
                        value_2 = sentilex.get(lemma_2, False)
                        if value_1 and value_2:
                            # print('Есть тональность у обоих составляющих')
                            # print(bigram)
                            sent = check_2lemmas_sent(value_1, value_2)
                        elif value_1:
                            # print('Тональность только у первого')
                            sent = value_1[2]
                        elif value_2:
                            # print('Тональность только у второго')
                            sent = value_2[2]
                        else:
                            # print('Вообще ничего не нашли')
                            sent = 'neutral'

                    if sent == 'positive':
                        pos_rev[bigram] = rev[bigram]
                    if sent == 'negative':
                        neg_rev[bigram] = rev[bigram]

                pos_bigrams.append(pos_rev)
                neg_bigrams.append(neg_rev)

                # print(pos_bigrams)
                # print(neg_bigrams)

            # trigram - check sentiment and lemma entries sentiment
            pos_trigrams = []
            neg_trigrams = []
            for rev in trigrams:
                pos_rev = {}
                neg_rev = {}
                for trigram in rev:
                    value = sentilex.get(trigram, False)
                    if value:
                        # print('У целого выражения есть тональность!')
                        sent = value[2]
                        # print(trigram)
                    else:
                        # print('У целого выражения тональности нет:')
                        lemma_1, lemma_2, lemma_3 = trigram.split()
                        value_1 = sentilex.get(lemma_1, False)
                        value_2 = sentilex.get(lemma_2, False)
                        value_3 = sentilex.get(lemma_3, False)
                        if value_1 and value_2 and value_3:
                            # print('Есть тональность у всех составляющих')
                            # print(trigram)
                            sent_1 = value_1[2]
                            sent_2 = value_2[2]
                            sent_3 = value_3[2]
                            sent_list = [sent_1, sent_2, sent_3]
                            # print(sent_list)
                            if sent_1 == sent_2 == sent_3:
                                # print('Тональности равны')
                                sent = value_1[2]
                            else:
                                c = Counter(sent_list)
                                pos = c.get('positive', False)
                                neg = c.get('negative', False)
                                neut = c.get('neutral', False)
                                if pos and neg and neut:
                                    sent = 'positive'
                                else:
                                    if pos:
                                        if pos > 1:
                                            sent = 'positive'
                                    if neg:
                                        if neg > 1:
                                            sent = 'negative'
                                    if neut:
                                        if neut > 1:
                                            sent = 'neutral'

                        elif value_1 and value_2:
                            sent = check_2lemmas_sent(value_1, value_2)
                        elif value_2 and value_3:
                            sent = check_2lemmas_sent(value_2, value_3)
                        elif value_1 and value_3:
                            sent = check_2lemmas_sent(value_1, value_3)

                        elif value_1:
                            # print('Тональность только у первого')
                            sent = value_1[2]
                        elif value_2:
                            # print('Тональность только у второго')
                            sent = value_2[2]
                        elif value_3:
                            # print('Тональность только у третьего')
                            sent = value_3[2]

                        else:
                            # print('Вообще ничего не нашли')
                            sent = 'neutral'

                    if sent == 'positive':
                        pos_rev[trigram] = rev[trigram]
                    if sent == 'negative':
                        neg_rev[trigram] = rev[trigram]

                pos_trigrams.append(pos_rev)
                neg_trigrams.append(neg_rev)

            # print(pos_trigrams)
            # print(neg_trigrams)

        else:
            # lemma - just check sentiment
            pos_lemmas = []
            neg_lemmas = []
            for lemma in lemmas:
                value = sentilex.get(lemma, False)
                if value:
                    sent = value[2]
                else:
                    sent = 'neutral'

                if sent == 'positive':
                    pos_lemmas.append(lemma)
                if sent == 'negative':
                    neg_lemmas.append(lemma)

            # print(pos_lemmas)
            # print(neg_lemmas)

            # bigram - check sentiment and lemma entries sentiment
            pos_bigrams = []
            neg_bigrams = []
            for bigram in bigrams:
                value = sentilex.get(bigram, False)
                if value:
                    # print('У целого выражения есть тональность!')
                    sent = value[2]
                    # print(sent)
                else:
                    # print('У целого выражения тональности нет:')
                    lemma_1, lemma_2 = bigram.split()
                    value_1 = sentilex.get(lemma_1, False)
                    value_2 = sentilex.get(lemma_2, False)
                    if value_1 and value_2:
                        # print('Есть тональность у обоих составляющих')
                        # print(bigram)
                        sent = check_2lemmas_sent(value_1, value_2)
                    elif value_1:
                        # print('Тональность только у первого')
                        sent = value_1[2]
                    elif value_2:
                        # print('Тональность только у второго')
                        sent = value_2[2]
                    else:
                        # print('Вообще ничего не нашли')
                        sent = 'neutral'

                if sent == 'positive':
                    pos_bigrams.append(bigram)
                if sent == 'negative':
                    neg_bigrams.append(bigram)

            # print(pos_bigrams)
            # print(neg_bigrams)

            # trigram - check sentiment and lemma entries sentiment
            pos_trigrams = []
            neg_trigrams = []
            for trigram in trigrams:
                value = sentilex.get(trigram, False)
                if value:
                    # print('У целого выражения есть тональность!')
                    sent = value[2]
                    # print(trigram)
                else:
                    # print('У целого выражения тональности нет:')
                    lemma_1, lemma_2, lemma_3 = trigram.split()
                    value_1 = sentilex.get(lemma_1, False)
                    value_2 = sentilex.get(lemma_2, False)
                    value_3 = sentilex.get(lemma_3, False)
                    if value_1 and value_2 and value_3:
                        # print('Есть тональность у всех составляющих')
                        # print(trigram)
                        sent_1 = value_1[2]
                        sent_2 = value_2[2]
                        sent_3 = value_3[2]
                        sent_list = [sent_1, sent_2, sent_3]
                        # print(sent_list)
                        if sent_1 == sent_2 == sent_3:
                            # print('Тональности равны')
                            sent = value_1[2]
                        else:
                            c = Counter(sent_list)
                            pos = c.get('positive', False)
                            neg = c.get('negative', False)
                            neut = c.get('neutral', False)
                            if pos and neg and neut:
                                sent = 'positive'
                            else:
                                if pos:
                                    if pos > 1:
                                        sent = 'positive'
                                if neg:
                                    if neg > 1:
                                        sent = 'negative'
                                if neut:
                                    if neut > 1:
                                        sent = 'neutral'

                    elif value_1 and value_2:
                        sent = check_2lemmas_sent(value_1, value_2)
                    elif value_2 and value_3:
                        sent = check_2lemmas_sent(value_2, value_3)
                    elif value_1 and value_3:
                        sent = check_2lemmas_sent(value_1, value_3)

                    elif value_1:
                        # print('Тональность только у первого')
                        sent = value_1[2]
                    elif value_2:
                        # print('Тональность только у второго')
                        sent = value_2[2]
                    elif value_3:
                        # print('Тональность только у третьего')
                        sent = value_3[2]

                    else:
                        # print('Вообще ничего не нашли')
                        sent = 'neutral'

                if sent == 'positive':
                    pos_trigrams.append(trigram)
                if sent == 'negative':
                    neg_trigrams.append(trigram)

            # print(pos_trigrams)
            # print(neg_trigrams)

    return pos_lemmas, pos_bigrams, pos_trigrams, neg_lemmas, neg_bigrams, neg_trigrams


def identify_sentiment(selected,  # selected by metric
                       sentiment, aspects,  # modes
                       pos_neg_ngrams=False):  # original
    '''
    Identify sentiment of aspects by selected mode.
    '''

    print('Identifying sentiment...')

    # chi-square test, weirdness, lda with positive/negative distribution
    # give us sentiment already
    if sentiment == 'Reference_corpus_Positive/Negative':

        if aspects in ['Chi-square_test', 'Weirdness', 'LDA']:

            return selected

        else:
            # print('Selected', len(selected))
            # print(selected[0])
            # print('9 словарей', len(selected[0]))
            sentiments = from_corpus(pos_neg_ngrams, selected)

            return sentiments

    if sentiment == 'Sentiment_lexicon' or 'Reference_corpus_Camera_reviews':

        # print(len(selected))
        sentiments = from_lexicon(*selected)

        return sentiments

    return 'Something went wrong!'


# processed = processing(
#     'apple-macbook-air-13.txt',
#     'Discontinious_collocations',
#     'Reference_corpus_Positive/Negative',
#     sp=True,
#     sw=True)
# # lemmas, bigrams, trigrams - просто списки
# extracted = extract_aspects(
#     *processed[:3],
#     'TF-IDF',
#     'Discontinious_collocations',
#     'Reference_corpus_Positive/Negative',
#     processed[3:])
# # print(len(processed))
# # print('Extracted', extracted)
# # print('Pos neg', processed[3:])
# identify_sentiment(
#     extracted,
#     'Reference_corpus_Positive/Negative',
#     'TF-IDF',
#     pos_neg_ngrams=processed[3:])
