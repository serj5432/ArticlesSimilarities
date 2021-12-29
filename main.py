import os
import codecs
import json


import yaml
import numpy as np
from scipy import spatial

from utils import url_to_text
from vectorizer import TextVectorizer




if __name__ == '__main__':

    with codecs.open('config.yml', encoding='utf-8') as f:
        d_config = yaml.load(f, Loader=yaml.FullLoader)

    n_closest = int(d_config['n_closest'])


    text_vectorizer = TextVectorizer(language=d_config['language'],
                                     min_sent_len=d_config['min_sent_len'])

    with codecs.open(d_config['url_list'], 'r', encoding='utf-8') as f:
        urls = [line.rstrip() for line in f]

    output = []
    for url in urls:
        d_out = {}
        text = url_to_text(url)
        response, v = text_vectorizer.vectorize(text)
        if v is None:
            continue
        d_out['title'] = url
        d_out['v'] = v.tolist()
        output.append(d_out)

    vs = [item['v'] for item in output]
    vs = np.array(vs)
    dists = spatial.distance.cdist(vs, vs)

    for n_item, item in enumerate(output):
        pos_closest = np.argsort(dists[n_item])[1:]
        titles_closest = [output[pos]['title'] for pos in pos_closest]
        item['titles_closest'] = titles_closest[:n_closest]
        #item['titles_closest_ws'] = [1 - dists[n_item][pos] for pos in pos_closest]
        del item['v']


    with open('result.txt', 'w', encoding='utf8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



