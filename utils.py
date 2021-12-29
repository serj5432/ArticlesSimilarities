import codecs
import requests

from bs4 import BeautifulSoup


def url_to_text(url):
    text = ''

    try:
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    except:
        return text

    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        'dl'
        # there may be more elements you don't want, such as "style", etc.
    ]
    for t in soup.find_all(text=True):
        if t.parent.name not in blacklist:
            text += '{} '.format(t)
    return text
