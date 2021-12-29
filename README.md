# ArticlesSimilarities

Finds similar artcles among webpages using pretrained models for Russian and English. 
Check https://github.com/RaRe-Technologies/gensim-data

## Installation

Steps:

1. Please, use python 3.9 or higher
2. Insall dependencies using requirements.txt
3. Download NLTK resources via Python cmd:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

4. config.yml:
		url_list - path to url list
		language - either 'ru' or 'en'
		min_sent_len - minimum sentence len to be processed 
		n_closest - n closest articles to current article
		
5. run main.py


