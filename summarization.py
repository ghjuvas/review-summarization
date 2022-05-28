# from nlp import processing
# from aspects import extract_aspects
# from sentiment import identify_sentiment


def summarize(sent, number):
    '''
    Summarize reviews by aspects.
    '''

    # extract_aspects returns different types of data
    # depending on modes
    # to parse data correctly we need to prove data types
    if isinstance(sent[0], dict):  # Weirdness, Chi-square, Frequencies

        list_important_pos = []
        for s in sent[:3]:
            sorted_keys_pos = sorted(s, key=s.get, reverse=True)
            important_pos = sorted_keys_pos[:number]
            list_important_pos += important_pos

        list_important_neg = []
        for s in sent[3:]:
            sorted_keys_neg = sorted(s, key=s.get, reverse=True)
            important_neg = sorted_keys_neg[:number]
            list_important_neg += important_neg

    if isinstance(sent[0], list):  # TF-IDF, LDA, Statistic collocations
        if isinstance(sent[0][0], dict):
            if len(sent) > 2:
                list_important_pos = []
                for s in sent[:3]:
                    for rev in s:
                        sorted_keys = sorted(rev, key=rev.get, reverse=True)
                        important = sorted_keys[:number]
                        list_important_pos += important

                list_important_neg = []
                for s in sent[3:]:
                    for rev in s:
                        sorted_keys = sorted(rev, key=rev.get, reverse=True)
                        important = sorted_keys[:number]
                        list_important_neg += important

            else:
                list_important_pos = []
                for rev in sent[0]:
                    sorted_keys = sorted(rev, key=rev.get, reverse=True)
                    important = sorted_keys[:number]
                    list_important_pos += important

                list_important_neg = []
                for rev in sent[1]:
                    sorted_keys = sorted(rev, key=rev.get, reverse=True)
                    important = sorted_keys[:number]
                    list_important_neg += important

        else:  # Statistic collocations (pre-defined ranking)

            list_important_pos = []
            for s in sent[:3]:
                list_important_pos += s[:number]

            list_important_neg = []
            for s in sent[3:]:
                list_important_neg += s[:number]

    print('Достоинства:', set(list_important_pos))
    print('Недостатки:', set(list_important_neg))

    return 'End!'


# processed = processing(
#     'apple-macbook-air-13.txt',
#     'Statistic_collocations',
#     'Reference_corpus_Positive/Negative',
#     sp=True,
#     sw=True)
# extracted = extract_aspects(
#     *processed[:3],
#     'Statistic_collocations',
#     'Statistic_collocations',
#     'Reference_corpus_Positive/Negative',
#     processed[3:])
# print('Extracted', extracted)
# print(len(extracted))
# # print('Pos neg', processed[3:])
# senti = identify_sentiment(
#     extracted,
#     'Reference_corpus_Positive/Negative',
#     'Statistic_collocations',
#     pos_neg_ngrams=processed[3:])
# summarize(senti, 10)
