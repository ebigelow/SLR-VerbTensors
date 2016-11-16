


import nltk
wnl = nltk.stem.WordNetLemmatizer()



def prepare_data(self, fname, save_dir='/Volumes/ebigelow-sg/Other/wiki-pages/'):
    id2info = {}
    cur_id = None

    with open(fname, 'r') as f:
        for line in f:

            if line[:5] == '<doc ':
                # TODO: fix regex here
                id_ = re.find('id="[0-9]^"')
                url = re.find('url="[*]^"')
                title = re.find('title="[*]^"')
                
                id2info[id_] = (title, url)
                cur_id = id_

            elif line[:6] == '</doc>':

                tokenized  = PTB_tokenize(cur_text)
                lemmatized = lemmatize

                save_wikipage(id_, save_dir)
                cur_text = []

            else:
                cur_text.append(lines)   

    return id2info         


def save_wikipage(id_, save_dir):
    """ TODO: save to individual file. """

    return

def lemmatize(TODO):
    TODO
    return


def tokenize(line):
    st = nltk.tokenize.StanfordTokenizer(path_to_jar='./stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar')
    st.tokenize('this is a test.')
    # returns:  [u'this', u'is', u'a', u'test', u'.']








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