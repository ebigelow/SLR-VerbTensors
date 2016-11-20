
import nltk
import gensim.models as gm


def isplit(iterable, splitters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]


class DataClean:

    def __init__(self, TODO):
        self.trip2id = TODO
        self.word2id = TODO

        self.id2info = {}
        self.id2lines = {}

    def prepare_data(self, fname, corenlp_jar='./data/stanford-corenlp/stanford-corenlp-3.4.1.jar',
                     candc_bin='./data/candc/bin/', save_dir='/Volumes/ebigelow-sg/Other/wiki-pages/'):
        cur_id = None
        self.lem  = nltk.WordNetLemmatizer()
        self.toke = nltk.tokenize.StanfordTokenizer(path_to_jar=corenlp_jar)

        with open(fname, 'r') as f:
            for line in f:

                if line[:5] == '<doc ':

                    _, id_, url, title, _ = line[:5].split('id','url','title',' >')
                    # TODO: fix regex here
                    id_ = re.find('id="[0-9]^"')
                    url = re.find('url="[*]^"')
                    title = re.find('title="[*]^"')
                    
                    self.id2lines[id_] = []
                    self.id2info[id_] = (title, url)
                    cur_id = id_

                elif line[:6] == '</doc>':
                    self.id2lines[id_] = []

                    self.handle_doc(cur_doc, cur_id)
                    cur_doc = []
                else:
                    cur_doc.append(line)   

        return id2info         


    # def lemmatize_sent(self, sent): 
    #     return ' '.join([self.lem.lemmatize(w) for w in sent])


    def handle_doc(self, doc_lines, cur_n):
        """
        doc_lines : all lines for a document

        Returns
        -------
        - Lemmatized document w/ triplet tags. One sentence per line,
          one number at end of sentence to indicate which triplet is in it
        - Dictionary of { triplet -> id }
        - Possibly -- substitute words for #'s and keep passing through '

        TODO
        ----
        - use doc id for saving 
        - save everything to a big compressed file, with 1 sentence per line, and keep
          a big dict mapping which line goes to which line in which wiki document
        > use MORPHA instead of wnl
        > use this for basic stuff like n-grams
        > then parse this all at once
        - then load parsed lines / originals 1 by 1 and collect input sentences + labels for gensim


        """

        # Lines roughtly correspond to paragraphs
        tokenized_lines = self.toke.tokenize_sentences(doc_lines)
        # List of list of sentence strings
        doc_sents = [' '.join(isplit(l, ['.'])) for l in tokenized_lines]
        # List of list of sentence strings
        #lemma_sents = [[self.lemmatize_sent(sent) for sent in doc] for doc in doc_sents]

        with open(self.save_dir + doc_id + '.txt', 'w') as f:
            f.writelines(doc_sents)

        return (cur_n, len(doc_sents)








"""


# Adapted some code from: 
#  https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/tokenizer/ptbtokenizer.py
#
# Do the PTB Tokenization and remove punctuations.
#

# import os
# import sys
# import subprocess
# import tempfile
# import itertools




# path to the stanford corenlp jar
corenlp_jar = './stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
punctuations = ['\'\'', '\'', '``', '`', '-LRB-', '-RRB-', '-LCB-', '-RCB-', \
                '?', '!', ',', ':', '-', '--', '...', ';']  # '.', 






# def PTB_tokenize(line):
#     cmd = ['java', '-cp', corenlp_jar, 'edu.stanford.nlp.process.PTBTokenizer', 
#            '-preserveLines', '-lowerCase']

#     # Save sentences to temporary file
#     path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
#     tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
#     tmp_file.write(sentences)
#     tmp_file.close()

#     # Tokenize sentence
#     cmd.append(os.path.basename(tmp_file.name))
#     p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, stdout=subprocess.PIPE)
#     token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]

#     # Remove temp file
#     os.remove(tmp_file.name)

#     lines = token_lines.split('\n')
#     lines = [' '.join([w for w in line.rstrip().split(' ') 
#                          if w not in punctuations])  for line in token_lines.split('\n')]
#     return lines










"""