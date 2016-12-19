import numpy as np
import pandas as pd
import inspect
import os

from verb import Verb





def make_path(path):
    try: 
        os.mkdir(path)
    except OSError:
        if not os.path.isdir(path):  raise

def par2tuple(P):
    return tuple((k,v) for k,v in P.items() if k not in IGNORE)



# ----------------------------------------------------------------------------------------------------
# Saving / loading tools



def save_verb(verb, fname):
    d = { 'stop_t': v.stop_t, 
          'r':v.rank, 's':v.svec, 'n':v.nvec,
          'P': v.P,   'Q': v.Q,   'R': v.R}
    np.save(fname, d)

def load_verb(fname):
    d = np.load(fname).item()
    v = parameterize(Verb, d)
    v.P, v.Q, v.R = (d['P'], d['Q'], d['R'])
    return v

def save_verbs(verbs, fname):
    d = { w: { 'stop_t': v.stop_t, 
               'r':v.rank, 's':v.svec, 'n':v.nvec,
               'P': v.P,   'Q': v.Q,   'R': v.R   }  for w, v in verbs.items()}
    np.save(fname, d)

def load_verbs(fname):
    d = np.load(fname).item()
    verbs = {}

    for w, v in d.items():
        v['init_restarts'] = 0
        verb = parameterize(Verb, v)
        verb.P = v['P']
        verb.Q = v['Q']
        verb.R = v['R']
        verbs[w] = verb

    return verbs





# ----------------------------------------------------------------------------------------------------





IGNORE = ['w2v_nn', 'w2v_svo', 'w2v_svo_full', 'test_data', 'gs_data', 'ks_data', 'objects', 'subjects', 'sentences']




def save_meta(params, fname):
    d = {k:v for k,v in params.items() if k not in IGNORE}
    np.save(fname, d)



def parameterize(f, params):
    """
    TODO: describe

    """
    if inspect.isclass(f):
        func_args = f.__init__.__code__.co_varnames[1:]
    else:
        func_args = f.__code__.co_varnames

    P = {k:v for k,v in params.items() if k in func_args}
    return f(**P)






def ablate_data(w2v_svo, data_ratio):
    for v, s_o in w2v_svo.items():
        full_keys = s_o.keys()
        N = len(full_keys)
        keep = int(N * data_ratio)

        keep_keys  = [full_keys[i] for i in np.random.choice(range(N), keep, replace=False)]
        w2v_svo[v] = {k:s_o[k] for k in keep_keys}

    return w2v_svo





def test_to_params(params):
    """ 
    Randomly select 10% of triplets as test data; we do this for each trial.

    """
    P = params.copy()

    w2v_svo, w2v_svo_test = split_test(P['w2v_svo_full'], n_stop=P['n_stop'])
    w2v_svo = ablate_data(w2v_svo, P['data_ratio'])

    P['w2v_svo']   = w2v_svo
    P['test_data'] = {k:format_data(P['w2v_nn'], s_o) for k,s_o in w2v_svo_test.items()}

    del P['w2v_svo_full']
    return P






# ----------------------------------------------------------------------------------------------------




def format_data(w2v_nn, s_o):
    sentences = np.vstack(s_o.values())
    subj_keys, obj_keys = zip(*s_o.keys())
    subjects = np.vstack([w2v_nn[sk] for sk in subj_keys])
    objects  = np.vstack([w2v_nn[ok] for ok in obj_keys])
    return sentences, subjects, objects


def load_test_data(cg=0, ck=0, gs_file='data/eval/GS2011data.txt', ks_file='data/eval/KS2014.txt'):

    gs_data = pd.read_csv(gs_file, delimiter=' ')
    ks_data = pd.read_csv(ks_file, delimiter=' ')

    gs_v1 = list(set(gs_data['verb']))[-cg:]
    gs_v2 = set(gs_data[gs_data['verb'].isin(gs_v1)]['landmark'])
    ks_v1 = list(set(ks_data['verb1']))[-ck:]
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

