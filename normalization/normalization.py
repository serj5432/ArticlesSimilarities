import re
import os
import string


import nltk
import pymorphy2
import stop_words
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from .contractions import CONTRACTION_MAP




wnl = WordNetLemmatizer()
morph = pymorphy2.MorphAnalyzer()
stopword_list = nltk.corpus.stopwords.words('english') +\
                stop_words.get_stop_words('russian') +\
                pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'names.csv'))['name'].tolist()


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def sent_tokenize(document):

    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences



def morph_word_ru(word):


    if (word[0] <= 'z' and word[0] >= 'a') or (word[0] <= 'Я' and word[0] >= 'А') or word.isupper():
        return word

    if len(word) <= 1 or word[0].isdigit() or word[0] < 'а' or word[0] > 'я':
        return None

    form = morph.parse(word)[0]
    POS = form.tag.POS
    if str(form.tag) == 'LATN':
        return None

    normal_form = form.normal_form
    return normal_form


def morph_word_en(word):


    if len(word) <= 1:
        return None
    word = wnl.lemmatize(word, 'v')
    word, pos = pos_tag([word])[0]
    if pos is not None and pos[0] in set(['V', 'N', 'J']):
        return word
    return None



def expand_contractions(text, contraction_mapping):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text



def normalize_text(text, language='en', use_abbr=False,  min_sent_len=0):


    re_abbr = re.compile( r'\b(?:[A-ZА-Я][a-za-я]*){2,}', flags=re.UNICODE)
    if language == 'en':
        morph_word = morph_word_en
    elif language == 'ru':
        morph_word = morph_word_ru
    else:
        raise ValueError('Invalind language')

    if language == 'en':
        text = expand_contractions(text, CONTRACTION_MAP)


    sents_upper = sent_tokenize(text)
    normalized_text = []
    sents = []
    if use_abbr:
        for sent in sents_upper:
            new_sent = []
            for w in sent.split(' '):
                if re_abbr.findall(w):
                    new_sent.append(w)
                else:
                    new_sent.append(w.lower())
            sents.append(' '.join(new_sent))
    else:
        sents = [sent.lower() for sent in sents_upper]

    for sent in sents:
        if len(sent) < min_sent_len:
            continue
        sent = remove_special_characters(sent)
        sent_tokens = tokenize_text(sent)
        filtered_sent_tokens = [token for token in sent_tokens if token not in stopword_list]
        for token in filtered_sent_tokens:
            morphed_token = morph_word(token)
            if morphed_token is None or morphed_token in stopword_list:
                continue
            normalized_text.append(morphed_token)
        normalized_text.append('EOS')
    return sents_upper, ' '.join(normalized_text)


