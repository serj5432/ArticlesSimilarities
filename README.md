# ArticlesSimilarities

Finds simlar artcles among webpages using pretrained models for Russian and English. 
Check https://github.com/RaRe-Technologies/gensim-data

## Installation

Steps:

1. Please, use python 3.9 ot higher
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
		data_path - path to html files
		language - either 'ru' or 'en'
		min_sent_len - minimum sentence len to be processed 
		
5. run main.py


