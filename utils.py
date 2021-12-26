from bs4 import BeautifulSoup
import codecs

def html_to_text(f_html):

    with codecs.open(f_html, encoding='utf-8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    text = soup.find_all(text=True)
    res = ''
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
    for t in text:
        if t.parent.name not in blacklist:
            res += '{} '.format(t)
    return res

