'''
Code for aspect-based review summarization course paper.

First run:
    pip3 install -r requirements.txt

If there is a problem with jamspell installation,
run this:
    sudo apt update -y
    sudo apt install -y swig3.0
    pip3 install jamspell

Please download sentiment lexicon RuSentiLex 2017:
    http://www.labinform.ru/pub/rusentilex/rusentilex_2017.txt
'''

import argparse
import os
from nlp import processing
from aspects import extract_aspects
from sentiment import identify_sentiment
from summarization import summarize
# from evaluation import evaluate

try:
    import jamspell
except (ImportError, ModuleNotFoundError):
    print('Please, install libraries from requirements file!')

modes = [
    'aspects + collocations + sentiment or all',
    '------------------------------------------',
    'TF-IDF+Discontinious_collocations+Reference_corpus_Positive/Negative',
    'TF-IDF+N-grams+Reference_corpus_Positive/Negative',
    'TF-IDF+Patterns+Reference_corpus_Positive/Negative',
    'TF-IDF+Discontinious_collocations_Sentiment_lexicon',
    'TF-IDF+N-grams+Sentiment_lexicon',
    'TF-IDF+Patterns+Sentiment_lexicon',
    'Frequencies+Patterns+Reference_corpus_Positive/Negative',
    'Frequencies+Patterns+Sentiment_lexicon',
    'Chi-square_test+Discontinious_collocations+Reference_corpus_Positive/Negative',
    'Chi-square_test+Discontinious_collocations+Reference_corpus_Camera_reviews',
    'Chi-square_test+N-grams+Reference_corpus_Positive/Negative',
    'Chi-square_test+N-grams+Reference_corpus_Camera_reviews',
    'Weirdness+Discontinious_collocations+Reference_corpus_Positive/Negative',
    'Weirdness+Discontinious_collocations+Reference_corpus_Camera reviews',
    'Weirdness+N-grams+Reference_corpus_Positive/Negative',
    'Weirdness+N-grams+Reference_corpus_Camera_reviews',
    'LDA+Discontinious_collocations+Reference_corpus_Positive/Negative',
    'LDA+N-grams+Reference_corpus_Positive/Negative',
    'Statistic_collocations+Statistic_collocations+Sentiment_lexicon',
    'Statistic_collocations+Statistic_collocations+Reference_corpus_Positive/Negative'
]


def main():
    '''
    Parse arguments and run chosen mode for summarization.
    '''

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Aspect-based review summarization by various modes'
        )

    parser.add_argument(
        '-c', '--corpus', type=str,
        help='path to review corpus'
        )
    # как лучше сделать mode?
    # просто прописать все возможные варианты?
    parser.add_argument(
        '-m', '--mode',
        help='mode of analysis, see list to choose'
        )
    parser.add_argument(
        '-sw', '--stopwords', type=bool,
        default=True, help='remove stopwords'
        )
    parser.add_argument(
        '-sp', '--spellchecking', type=bool,
        default=True, help='spelling correction'
        )
    parser.add_argument(
        '-n', '--number', type=int,
        default=5, help='number of aspects in summarization'
        )
    parser.add_argument(
        '-t', '--test', type=bool,
        default=False,
        help='run code with test data and baseline mode tf-idf on n-grams by reference corpus on positive and negative opinions'
        )
    parser.add_argument(
        '-lm', '--list',
        action='store_true',
        help='print the list of available modes to choose'
        )

    args = parser.parse_args()

    # print list of available modes separated
    if args.list:
        [print(i) for i in modes]

    # run evaluation code on the native corpuses
    # elif args.test:
    #     return evaluate()

    # run other program code
    else:
        # handle extension errors
        if not os.path.exists(args.corpus):
            raise ValueError('Path is not valid!')
        if os.path.splitext(args.corpus)[1] not in ['.txt', '']:
            raise ValueError(
                '''Program do not support this file extension!
                Please, use .txt or nothing'''
                )

        '''
        Pipeline:
        1. Tokenization - stanza_nlp
        2. Lemmatization - stanza_get_lemma
        3. Collocations (by selection):
            get_n_grams, get_discont_colls, patterns
        4. Search aspect solution (by selection):
            tf-idf, tf, tf-idf, weirdness, chi-square, lda, stat colls
        5. Search sentiment solution (by selection):
            camera reviews, positive|negative, sentiment lexicon
        6. Get summary:
            from ranking
        '''

        mode_a, mode_c, mode_s = args.mode.split('+')

        processed = processing(args.corpus,
                               mode_c,
                               mode_s,
                               args.spellchecking,
                               args.stopwords)
        # processing returns different number of variables
        # it depends on mode
        if len(processed) > 3:
            aspects = extract_aspects(*processed[:3],
                                      mode_a,
                                      mode_c,
                                      mode_s,
                                      processed[3:])
        else:
            aspects = extract_aspects(*processed[:3],
                                      mode_a,
                                      mode_c,
                                      mode_s)
        sent = identify_sentiment(aspects,
                                  mode_s, mode_a,
                                  processed[3:])
        summarize(sent, args.number)

    #     if args.aspects:
    #         print(extract_aspects(reviews, args.mode, args.stopwords))
    #     elif args.sentiment:
    #         extracted_aspects = extract_aspects(reviews, args.mode)
    #         print(identify_sentiment(extracted_aspects))
    #     else:
    #         extracted_aspects = extract_aspects(reviews, args.mode)
    #         subdivided_aspects = identify_sentiment(extracted_aspects)
    #         summary = summarize(subdivided_aspects)


if __name__ == '__main__':
    main()
