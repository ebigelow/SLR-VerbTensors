"""
http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim

"""

from gensim.corpora.wikicorpus import WikiCorpus
from tqdm import tqdm

fname = 'enwiki-latest-pages-articles.xml.bz2'
out_file = 'wiki.en.txt'

i = 0
with open(out_file, 'w') as f:
    print 'loading . . .'
    wiki = WikiCorpus(fname, lemmatize=False, dictionary={})
    print 'loaded corpus!!'
    for text in tqdm(wiki.get_texts()):
        f.write('.'.join(text) + '\n')