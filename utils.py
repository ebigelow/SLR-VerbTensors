import numpy as np
import pandas as pd
import inspect



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






def test_to_params(params):
    """ 
    Randomly select 10% of triplets as test data; we do this for each trial.

    """
    P = params.copy()
    w2v_svo_full, n_stop = (P['w2v_svo_full'], P['n_stop'])
    w2v_svo, w2v_svo_test = split_test(w2v_svo_full, n_stop=n_stop)
    test_data = {k:format_data(P['w2v_nn'], s_o) for k,s_o in w2v_svo_test.items()}

    P['w2v_svo']   = w2v_svo
    P['test_data'] = test_data
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

