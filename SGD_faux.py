import numpy as np
import gensim.models as gm
import pandas as pd
import itertools
from collections import defaultdict
# from scipy.spatial.distance import cosine
from tqdm import tqdm, trange



class Verb:

    def __init__(self, rank=50, svec=200, nvec=300, init_noise=0.1,
                 test_t=None, test_s=None, test_o=None):
        self.P = init_noise * np.random.rand(rank, svec)
        self.Q = init_noise * np.random.rand(rank, nvec)
        self.R = init_noise * np.random.rand(rank, nvec)
        self.test_data = (test_t, test_s, test_o)

    def V(self, s, o):
        P,Q,R = (self.P, self.Q, self.R)
        Qs_Ro = Q.dot(s) * R.dot(o)
        return P.T.dot(Qs_Ro)

    def test_loss():
        t,s,o = self.test_data
        if t is None:
            print 'error: no test data'; return 0.0
        mse = ((t - self.V(s, o)) ** 2).sum()  /  t.shape[1]
        return mse

    def SGD_cp(self, sentences, subjects, objects, 
               epochs=100, batch_size=4, learning_rate=1.0):

        Mv = sentences.shape[0]
        lr = learning_rate / Mv
        batches = Mv / batch_size

        for e in trange(epochs):
            for i in range(batches):

                P,Q,R = (self.P, self.Q, self.R)
                t = sentences[i : i + batch_size].T
                s =  subjects[i : i + batch_size].T
                o =   objects[i : i + batch_size].T

                Qs = Q.dot(s)
                Ro = R.dot(o)
                Qs_Ro = Qs * Ro

                if e % 3 == 0:
                    dL_dP = Qs_Ro.dot(Qs_Ro.T).dot(P)  -  t.dot(Qs_Ro.T).T
                    self.P -= lr * dL_dP

                elif e % 3 == 1:
                    dL_dQ = ( Ro * (P.dot(P.T).dot(Qs_Ro)  -  P.dot(t)) ).dot(s.T)
                    self.Q -= lr * dL_dQ 

                elif e % 3 == 2:
                    dL_dR = ( Qs * (P.dot(P.T).dot(Qs_Ro)  -  P.dot(t)) ).dot(o.T)
                    self.R -= lr * dL_dR





if __name__ == '__main__':

    """
    
    pre-trained w2v, d2v
    --------------------
    - randomly sample some of the other subjects and objects in our list
    - do this to fill faux triplets (e.g. "boy kick " + ~"tree")  AND
      to fill subjs and objs (e.g. "kick" + ~"producer", ~"table")
    > should we sample *verbs* not in the list??? (no... why?)

    wikipedia
    ---------
    - sample frequent n-gram completions --
        maybe find these by querying google API, or using colala-corpus???
    - sample words not in list -- this seems important, for use in various things




    """

    # Load up the testing data
    gs_data = pd.read_csv('data/eval/GS2011data.txt', delimiter=' ')
    ks_data = pd.read_csv('data/eval/KS2014.txt',     delimiter=' ')

    # Collect sets of unique triplets and pairs (verb-subject, verb-object)
    print 'collecting triplets . . .'
    gs_SVO = zip(gs_data['verb'].tolist(), gs_data['landmark'].tolist(),
                 gs_data['subject'].tolist(), gs_data['object'].tolist())
    V_SO = defaultdict(lambda: set())
    V_S  = defaultdict(lambda: set())
    V_O  = defaultdict(lambda: set())

    for v,t,s,o in gs_SVO:
        V_SO[v].update({(s, o)})
        V_SO[t].update({(s, o)})  # TODO: should this be (s,o) -- are we fitting test data in this way?
        V_S[v].update({s})
        V_O[v].update({o})

    # Genetrate some extra datas
    gs_nouns = set(gs_data['subject'].tolist() + gs_data['object'].tolist())
    noun_prods = [i for i in itertools.product(gs_nouns, gs_nouns)]

    # Number of unique triplets for each verb  (250 given for GS)
    n_trips = 500

    # Sample random triplets so we have `n_trips` for each verb
    for v in tqdm(V_SO):
        n_cur = len(V_SO[v])
        n_samples = n_trips - n_cur

        candidates = [(s,o) for s,o in noun_prods if (s,o) not in V_SO[v]]
        sample_idxs = np.random.choice(range(len(candidates)), n_samples, replace=False)
        samples = [candidates[i] for i in sample_idxs]
        V_SO[v].update(set(samples))

    # Setup dictionaries mapping triplets to vectors
    print 'building faux vectors . . .'
    d2v = gm.Doc2Vec.load('data/doc2vec_wiki/doc2vec.bin')
    w2v = gm.Word2Vec.load('data/word2vec_wiki/word2vec.bin')

    faux_data = defaultdict(lambda: list())
    for v in tqdm(V_SO):
        for s,o in V_SO[v]:
            svo_vec = d2v[(s,v,o)].sum(axis=0)         # TODO is this right . . ?
            s_vec = w2v[s]
            o_vec = w2v[o]
            faux_data[v].append((svo_vec, s_vec, o_vec))
        faux_data[v] = [np.vstack(x) for x in zip(*faux_data[v])]   # stack vectors into matrices

    # Save Gensim data to numpy dict
    faux_ = {v:faux_data[v] for v in faux_data}
    np.save(faux_, 'data/faux_vectors.npy')
    V_SO_ = {v:V_SO[v] for v in V_SO}
    np.save(V_SO_, 'data/faux_triplets.npy')

    # faux_data = np.load('data/faux_vectors.npy').item()
    # V_SO = np.load('data/faux_triplets.npy').item()


    # Parameters
    rank = 50
    svec = len(svo_vec)
    nvec = len(s_vec)

    # Build and train model for each verb
    verbs = {}
    tq = tqdm(V_SO, desc='\nbegin training . . .\n\n', leave=True)

    for v in tq:
        tq.set_description('Training: "' + v + '"')
        verbs[v] = Verb(rank=rank, svec=svec, nvec=nvec, init_noise=0.1)
        verbs[v].SGD_cp(*faux_data[v])

    # Predict similarity for GS test data, using learned verb representations
    test_pairs = []
    for row in gs_data.iterrows():

        pid, v, s, o, t, score, hilo = row[1]

        verb_vec = verbs[v].V(s,o)
        test_vec = verbs[t].V(s,o)

        predict = cosine(verb_vec)     # TODO: test different distances
        test_pairs.append((predict, score))

    # Compute spearman R for full data
    rho, pvalue = scipy.stats.spearmanr(*zip(*test_pairs))
    print 'rho: {}\npvalue: {}'.format(rho, pvalue)




