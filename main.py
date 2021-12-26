import os
import codecs


import yaml, json
import numpy as np
from scipy import spatial

from utils import html_to_text
from vectorizer import TextVectorizer




if __name__ == '__main__':

    with codecs.open('config.yml', encoding='utf-8') as f:
        d_config = yaml.load(f, Loader=yaml.FullLoader)

    text_vectorizer = TextVectorizer(language=d_config['language'],
                                     min_sent_len=d_config['min_sent_len'])


    output = []
    for f_name in os.listdir(d_config['data_path']):
        d_out = {}
        f_name_full = os.path.join(d_config['data_path'], f_name)
        text = html_to_text(f_html=f_name_full)
        response, v = text_vectorizer.vectorize(text)
        if v is None:
            continue
        #vs.append(v)
        d_out['title'] = f_name
        d_out['v'] = v.tolist()
        output.append(d_out)

    vs = [item['v'] for item in output]
    vs = np.array(vs)
    dists = spatial.distance.cdist(vs, vs)

    for n_item, item in enumerate(output):
        pos_closest = np.argsort(dists[n_item])[1:]
        titles_closest = [output[pos]['title'] for pos in pos_closest]
        item['titles_closest'] = titles_closest
        item['titles_closest_ws'] = [1 - dists[n_item][pos] for pos in pos_closest]
        del item['v']



    with open(os.path.join(d_config['data_path'], 'result.txt'), 'w', encoding='utf8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



