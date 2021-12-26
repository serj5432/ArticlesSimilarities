import numpy as np
import gensim.downloader as api
from normalization import normalization



class TextVectorizer:

    def __init__(self, language='ru', min_sent_len=20):
        self.language = language
        self.min_sent_len = min_sent_len


        if self.language == 'ru':
            self.valid_poses = ['ADJ', 'NOUN', 'VERB']
            self.model = api.load("word2vec-ruscorpora-300")
        else:
            self.model = api.load("glove-wiki-gigaword-300")

        self.vocab = self.model.index_to_key


    def vectorize(self, text):

        try:
            _, text_norm = normalization.normalize_text(text,
                                                        language=self.language,
                                                        min_sent_len=self.min_sent_len
                                                        )
            text_norm = text_norm.replace('ё', 'е')
        except KeyError:
            response = 'Invalid text format', None
            return response

        vecs = []

        for w in text_norm.split():

            if w == 'EOS':
                continue
            if self.language == 'ru':
                for pos in self.valid_poses:
                    w_pos = '{}_{}'.format(w, pos)
                    if w_pos in self.vocab:
                        vecs.append(self.model[w_pos])
                        break
            else:
                if w in self.vocab:
                    vecs.append(self.model[w])

        if len(vecs) > 0:
            vecs = np.array(vecs)
            v_mean = vecs.mean(axis=0)
            return text_norm, v_mean
        else:
            response = 'No valid words in text', None
            return response
