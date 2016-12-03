import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from tqdm import tqdm, trange


class Verb:

    def __init__(self, test_data=None, stop_t=0.01, rank=50, svec=100, nvec=100, init_noise=0.1):
        self.test_data, self.stop_t = (test_data, stop_t)
        self.rank, self.svec, self.nvec = (rank, svec, nvec)
        self.init_weights(init_noise)

    def init_weights(self, init_noise):
        self.P = init_noise * np.random.rand(self.rank, self.svec)
        self.Q = init_noise * np.random.rand(self.rank, self.svec)
        self.R = init_noise * np.random.rand(self.rank, self.svec)

    def V(self, s, o):
        P,Q,R = (self.P, self.Q, self.R)
        Qs_Ro = Q.dot(s) * R.dot(o)
        return P.T.dot(Qs_Ro)

    def L(self, sentences, subjects, objects):
        Mv = sentences.shape[0]
        sq_diffs = [sum( (self.V(s,o) - t)**2 ) for s,o,t in zip(subjects, objects, sentences)]
        return sum(sq_diffs) / Mv

    def update_min(self, cur_loss):
        if not hasattr(self, 'min_loss') or (cur_loss < self.min_loss):
            self.min_loss = cur_loss
            self.min_params = {'P': self.P.copy(), 'Q': self.Q.copy(), 'R': self.R.copy()}

    def stop_early(self):
        if self.test_data in (None, 0):
            return False
        else:
            cur_loss = self.L(*self.test_data)
            if not hasattr(self, 'prev_loss'):
                self.prev_loss = cur_loss

            self.update_min(cur_loss)

            d = cur_loss - self.prev_loss
            self.prev_loss = cur_loss
            return (d > self.stop_t)


    def SGD(self, sentences, subjects, objects, 
            epochs=100, batch_size=4, learning_rate=1.0):
        """
        Arguments
        ---------
        sentences :  (Mv x s) matrix
        subjects  :  (Mv x n) matrix
        objects   :  (Mv x n) matrix

        Algorithm
        ---------
        alternate updating P, Q, R

        """

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

            if self.stop_early():
                # print 'stopped at : {}'.format(e)
                return

            # if e % (epochs / 8) == 0:
            #     L = self.L(sentences, subjects, objects)
            #     print 'epoch: {}   |   L: {}'.format(e, L)

    def ADA_delta(self, sentences, subjects, objects, n_trials=10,
                  epochs=100, batch_size=100, learning_rate=1.0,
                  rho=0.95, eps=1e-6):
        """
        https://arxiv.org/pdf/1212.5701v1.pdf

        pseudocode
        ----------
        rho: decay rate
        eps: (noise?) constant

        initialize E[g^2]_0 = 0, initialize E[dx^2]_0 = 0

        for t = 1..T:
            g_t = gradient
            E[g^2]_t = rho * E[g^2]_{t-1} + (1-r) g_t^2
            dx_t = - g_t * RMS(dx)_{t-1} / RMS(g_t) 
                 = - g_t * sqrt(E[dx^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps)
            E[dx^2]_t = rho * E[dx^2]_{t-1} + (1-rho) dx_t^2
            x_{t+1} = x_t + dx_t

        where RMS(y)_t := sqrt(E[y^2]_t + eps)

        in python
        ---------

        E_g2_prev  = 0.0
        E_dx2_prev = 0.0

        for t in trange(T):
            g = COMPUTE_GRADIENT
            E_g2 = rho * E_g2_prev  +  (1-rho) * g**2
            dx = -g * np.sqrt(E_dx2_prev + eps) / np.sqrt(E_g2 + eps)
            E_dx2 = rho * E_dx2_prev  +  (1-rho) * dx**2
            x += dx
            E_g2_prev, E_dx2_prev = (E_g2, E_dx2)

        """
        Mv = sentences.shape[0]
        lr = learning_rate / Mv
        batches = Mv / batch_size

        E_g2_prev  = [0.0, 0.0, 0.0]
        E_dx2_prev = [0.0, 0.0, 0.0]

        # for tr in trange(n_trials):

        for e in trange(epochs):

            shuffled_batches = sorted(range(batches), key=lambda x: np.random.rand())
            for i in shuffled_batches:

                P,Q,R = (self.P, self.Q, self.R)
                t = sentences[i : i + batch_size].T
                s =  subjects[i : i + batch_size].T
                o =   objects[i : i + batch_size].T

                Qs = Q.dot(s)
                Ro = R.dot(o)
                Qs_Ro = Qs * Ro

                if e % 3 == 0:
                    g = Qs_Ro.dot(Qs_Ro.T).dot(P)  -  t.dot(Qs_Ro.T).T
                    E_g2  = rho * E_g2_prev[0]  +  (1-rho) * g**2
                    dL_dP = -g * np.sqrt(E_dx2_prev[0] + eps) / np.sqrt(E_g2 + eps)
                    E_dx2 = rho * E_dx2_prev[0]  +  (1-rho) * dL_dP**2

                    self.P += lr * dL_dP
                    E_g2_prev[0] = E_g2
                    E_dx2_prev[0] = E_dx2

                elif e % 3 == 1:
                    g = ( Ro * (P.dot(P.T).dot(Qs_Ro)  -  P.dot(t)) ).dot(s.T)
                    E_g2  = rho * E_g2_prev[1]  +  (1-rho) * g**2
                    dL_dQ = -g * np.sqrt(E_dx2_prev[1] + eps) / np.sqrt(E_g2 + eps)
                    E_dx2 = rho * E_dx2_prev[1]  +  (1-rho) * dL_dQ**2

                    self.Q += lr * dL_dQ
                    E_g2_prev[1] = E_g2
                    E_dx2_prev[1] = E_dx2

                elif e % 3 == 2:
                    g = ( Qs * (P.dot(P.T).dot(Qs_Ro)  -  P.dot(t)) ).dot(o.T)
                    E_g2  = rho * E_g2_prev[2]  +  (1-rho) * g**2
                    dL_dR = -g * np.sqrt(E_dx2_prev[2] + eps) / np.sqrt(E_g2 + eps)
                    E_dx2 = rho * E_dx2_prev[2]  +  (1-rho) * dL_dR**2

                    self.R += lr * dL_dR
                    E_g2_prev[2] = E_g2
                    E_dx2_prev[2] = E_dx2

            if self.stop_early():
                # print 'stopped at : {}'.format(e)
                break

        self.P = self.min_params['P']
        self.Q = self.min_params['Q']
        self.R = self.min_params['R']
        del self.min_params


def save_verbs(verbs, fname):
    d = { w: { 'r':v.rank, 's':v.svec, 'n':v.nvec,
               'P': v.P,   'Q': v.Q,   'R': v.R  }  for w, v in verbs.items()}
    np.save(fname, d)
    #print 'Saved verbs to: ' + fname


def load_verbs(fname):
    # TODO: update
    d = np.load(fname).item()
    verbs = {}

    for w, v in d.items():
        verb = Verb(rank=v['r'], svec=v['s'], nvec=v['n'])
        verb.P = v['P']
        verb.Q = v['Q']
        verb.R = v['R']
        verbs[w] = verb

    print 'Loaded verbs from ' + fname
    return verbs



def test_verbs(verbs, w2v_nn, test_data, dset='GS', verbal=False):

    # Predict similarity for GS test data, using learned verb representations
    test_pairs = []
    if verbal: print '\n\nTesting on '+dset+' data . . .'

    for row in test_data.iterrows():

        if dset == 'GS':
            pid, v, s, o, t, gt_score, hilo = row[1]
            
            svo1_vec = verbs[v].V(w2v_nn[s], w2v_nn[o])     # "verb"
            svo2_vec = verbs[t].V(w2v_nn[s], w2v_nn[o])     # "landmark" (target)

        elif dset == 'KS':
            pid, s1,v1,o1, s2,v2,o2, gt_score = row[1]
            
            svo1_vec = verbs[v1].V(w2v_nn[s1], w2v_nn[o1])
            svo2_vec = verbs[v2].V(w2v_nn[s2], w2v_nn[o2])

        similarity = 1 - cosine(svo1_vec, svo2_vec)     # TODO: test different distances
        test_pairs.append((similarity, gt_score))

    # Compute spearman R for full data
    rho_, pvalue = spearmanr(*zip(*test_pairs))
    if verbal: print '\trho: {}\n\tpvalue: {}'.format(rho_, pvalue)
    return rho_, pvalue 


def train_verbs(w2v_nn, w2v_svo, test_data=None, stop_t=0.1, svec=100, nvec=100, rank=50, 
                batch_size=20, epochs=500, n_trials=5, learning_rate=1e-1, init_noise=0.1,
                optimizer='ADAD', rho=0.95, eps=1e-6):
    # Build and train model for each verb
    print '\n'
    verbs = {}

    tq = tqdm(w2v_svo.items(), desc='', leave=True)
    for v, s_o in tq:      # TODO change:   in w2v_svo.items():
        tq.set_description('Training: "' + v + '"')

        verbs[v] = Verb(test_data=test_data[v], stop_t=stop_t, rank=rank, svec=svec, nvec=nvec, init_noise=init_noise)
        sentences, subjects, objects = format_data(w2v_nn, s_o)

        if optimizer == 'SGD':
            verbs[v].SGD(sentences, subjects, objects, learning_rate=learning_rate,
                         batch_size=batch_size, epochs=epochs, n_trials=n_trials)
        elif optimizer == 'ADAD':
            verbs[v].ADA_delta(sentences, subjects, objects, learning_rate=learning_rate, n_trials=n_trials,
                               epochs=epochs, batch_size=batch_size, rho=rho, eps=eps)

    return verbs

def format_data(w2v_nn, s_o):
    sentences = np.vstack(s_o.values())
    subj_keys, obj_keys = zip(*s_o.keys())
    subjects = np.vstack([w2v_nn[sk] for sk in subj_keys])
    objects  = np.vstack([w2v_nn[ok] for ok in obj_keys])
    return sentences, subjects, objects


def save_meta(fname):
    # TODO: concise but assumes global vars
    d = {     \
        'optimizer':     optimizer,
        'cg':            cg,
        'ck':            ck,
        'batch_size':    batch_size,
        'rank':          rank,
        'epochs':        epochs,
        'n_trials':      n_trials,
        'learning_rate': learning_rate,
        'init_noise':    init_noise,
        'rho':           rho,
        'eps':           eps,
        'n_stop':        n_stop,
        'stop_t':        stop_t
    }
    np.save(fname, d)


def load_test_data(cg=0, ck=-1, gs_file='data/eval/GS2011data.txt', ks_file='data/eval/KS2014.txt'):

    gs_data = pd.read_csv(gs_file, delimiter=' ')
    ks_data = pd.read_csv(ks_file, delimiter=' ')

    gs_v1 = list(set(gs_data['verb']))[-cg:]
    gs_v2 = set(gs_data[gs_data['verb'].isin(gs_v1)]['landmark'])
    ks_v1 = list(set(ks_data['verb1']))[:ck]
    ks_v2 = set(ks_data[ks_data['verb1'].isin(ks_v1)]['verb2'])

    gs_data = gs_data[gs_data['verb'].isin(gs_v1)   &  gs_data['landmark'].isin(gs_v2)]
    ks_data = ks_data[ks_data['verb1'].isin(ks_v1)  &  ks_data['verb2'].isin(ks_v2)]
    test_vs =  set.union(set(gs_v1), gs_v2, set(ks_v1), ks_v2)

    return gs_data, ks_data, test_vs


def load_word2vec(test_vs, nn_file='data/w2v/w2v-nouns.npy', svo_file='data/w2v/w2v-svo-triplets.npy'):

    w2v_nn  = np.load(nn_file).item()
    w2v_svo = np.load(svo_file).item()

    for v, s_o in w2v_svo.items():
        for s,o in w2v_svo[v].keys():
            if s not in w2v_nn or o not in w2v_nn:
                del w2v_svo[v][(s, o)]
                # print 'Removed: ({}, {}, {})'.format(v,s,o)
        if v not in test_vs:
            del w2v_svo[v]

    return w2v_nn, w2v_svo


def split_test(w2v_svo_full, n_stop=0.1):
    w2v_svo = {}
    w2v_svo_test = {}
    for v, s_o in w2v_svo_full.items():
        full_keys = s_o.keys()
        N = len(full_keys)
        n_test = int(N * n_stop)
        test_keys = [full_keys[i] for i in np.random.choice(range(N), n_test, replace=False)]
        w2v_svo_test[v] = { k:s_o[k] for k in test_keys }
        w2v_svo[v]      = { k:s_o[k] for k in full_keys if k not in test_keys }

    return w2v_svo, w2v_svo_test



if __name__ == '__main__':



    # ------------------------------------------------------------------------
    # Parameters

    save_file     = 'data/verbs_test3'
    train         = True

    rank          = 5

    batch_size    = 20
    epochs        = 500
    n_trials      = 5

    learning_rate = 1.0
    init_noise    = 0.1

    optimizer     = 'ADAD'  # | 'SGD'
    rho           = 0.9
    eps           = 1e-6

    cg            = 0   # set to 0 for full data
    ck            = -1     # set to -1 for full data  (minus 1 point)

    n_stop        = 0.1
    stop_t        = 0

    # ------------------------------------------------------------------------
    # Load & filter test data

    gs_file = 'data/eval/GS2011data.txt'
    ks_file = 'data/eval/KS2014.txt'
    gs_data, ks_data, test_vs = load_test_data(cg, ck, gs_file=gs_file, ks_file=ks_file)

    # ------------------------------------------------------------------------
    # Load & filter word/triplet vectors

    nn_file  = 'data/w2v/w2v-nouns.npy'
    svo_file = 'data/w2v/w2v-svo-triplets.npy'
    w2v_nn, w2v_svo_full = load_word2vec(test_vs, nn_file=nn_file, svo_file=svo_file)
    w2v_svo, w2v_svo_test = split_test(w2v_svo_full, n_stop=n_stop)
    test_data = {k:format_data(w2v_nn, s_o) for k,s_o in w2v_svo_test.items()}

    # ------------------------------------------------------------------------
    # Train / load verb parameters

    if train:
        best_acc = 0.0
        for k in trange(n_trials):
            verbs = []
            verbs = train_verbs(w2v_nn, w2v_svo, test_data=test_data, stop_t=stop_t,
                                svec=100, nvec=100, rank=rank, 
                                batch_size=batch_size, epochs=epochs, n_trials=n_trials,
                                learning_rate=learning_rate, init_noise=init_noise,
                                optimizer=optimizer, rho=rho, eps=eps)

            test_verbs(verbs, w2v_nn, gs_data, dset='GS')
            test_verbs(verbs, w2v_nn, ks_data, dset='KS')

            # cur_acc = np.mean([test_verbs(verbs, w2v_nn, gs_data, dset='GS')[0],
                               # test_verbs(verbs, w2v_nn, ks_data, dset='KS')[0]])
            cur_acc = test_verbs(verbs, w2v_nn, ks_data, dset='KS')[0]
            if cur_acc > best_acc:
                save_verbs(verbs, save_file + '.npy')
                save_meta(save_file + '_meta.npy')



    else:
        verbs = load_verbs(save_file + '.npy')

    test_verbs(verbs, w2v_nn, gs_data, dset='GS', verbal=True)
    test_verbs(verbs, w2v_nn, ks_data, dset='KS', verbal=True)




