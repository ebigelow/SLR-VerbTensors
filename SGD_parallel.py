import itertools

from utils import *
from SGD_new import *
from verb import *

# from SimpleMPI.MPI_map import MPI_map

import time

def train_trials_grid(params, grid_params, parallel=False):
    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    rows = []

    iter_ = enumerate(itertools.product(*it_params))
    loop1 = tqdm(iter_) if params['verbose'] else iter_

    for i, grid_iter in loop1:
        params_iter = dict(params.items() + list(grid_iter))

        best_acc_gs = 0.0
        best_acc_ks = 0.0

        loop2 = trange if params['verbose'] else range

        t1 = time.time()
        for k in loop2(params['n_trials']):
            P = test_to_params(params_iter)
            # verbs = train_verbs(P)
            verbs = train_verbs_parallel(P) if parallel else train_verbs(P)

            curr_acc_gs = test_verbs(verbs, P['w2v_nn'], P['gs_data'], dset='GS', verbose=params['verbose'])[0]
            if curr_acc_gs > best_acc_gs:
                save_verbs(verbs, '{}-{}_GS.npy'.format(P['save_file'], i))
                best_acc_gs = curr_acc_gs

            curr_acc_ks = test_verbs(verbs, P['w2v_nn'], P['ks_data'], dset='KS', verbose=params['verbose'])[0]
            if curr_acc_ks > best_acc_ks:
                save_verbs(verbs, '{}-{}_KS.npy'.format(P['save_file'], i))
                best_acc_gs = curr_acc_ks

        elapsed = time.time() - t1
        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), 
                          ('id', i) ]  +  [(k,v) for k,v in P.items() if k not in IGNORE]  ))
        pd.DataFrame(rows).to_csv(params['grid_file'])
        print '~~~~~ Grid iteration: {}  time: {}    best GS: {}   best KS: {}'.format(i, elapsed, best_acc_gs, best_acc_ks)


def verb_fun(P):
    return train_verbs(test_to_params(P))

def train_trials_grid_parallel(params, grid_params):
    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    rows = []

    iter_ = enumerate(itertools.product(*it_params))
    loop1 = tqdm(iter_) if params['verbose'] else iter_

    for i, grid_iter in loop1:
        P = dict(params.items() + list(grid_iter))

        best_acc_gs = 0.0
        best_acc_ks = 0.0


        t1 = time.time()        
        parfor = MPI_map(verb_fun, [P]*P['n_trials'], progress_bar=False)
        t2 = time.time()

        for verbs in parfor:

            curr_acc_gs = test_verbs(verbs, P['w2v_nn'], P['gs_data'], dset='GS', verbose=P['verbose'])[0]
            if curr_acc_gs > best_acc_gs:
                save_verbs(verbs, '{}-{}_GS.npy'.format(P['save_file'], i))
                best_acc_gs = curr_acc_gs

            curr_acc_ks = test_verbs(verbs, P['w2v_nn'], P['ks_data'], dset='KS', verbose=P['verbose'])[0]
            if curr_acc_ks > best_acc_ks:
                save_verbs(verbs, '{}-{}_KS.npy'.format(P['save_file'], i))
                best_acc_ks = curr_acc_ks

        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), 
                          ('id', i) ]  +  [(k,v) for k,v in P.items() if k not in IGNORE]  ))
        pd.DataFrame(rows).to_csv(params['grid_file'])
        print '~~~~~ Grid iteration: {}  time: {}    best GS: {}   best KS: {}'.format(i, t2-t1, best_acc_gs, best_acc_ks)





def verb_parfun(arguments):
    """
    Build and train model for each verb

    """
    v, s_o, params = arguments

    P = params.copy()
    verb = parameterize(Verb, P)

    P['sentences'],  P['subjects'], P['objects'] = format_data(w2v_nn, s_o)
    if P['optimizer'] == 'SGD':
        parameterize(verb.SGD, P)
    elif P['optimizer'] == 'ADAD':
        parameterize(verb.ADA_delta, P)

    return (v, verb)


def train_verbs_parallel(params):
    """
    Split test data before, to minimize RAM overhead

    """
    map_inputs = []
    for v, s_o in params['w2v_svo'].items():
        P = params.copy()

        nouns = [n for pair in s_o for n in pair]
        P['w2v_nn'] = {n:vec for n, vec in P['w2v_nn'].items() if n in nouns}
        P['test_data'] = params['test_data'][v]
        del P['w2v_svo']; del P['ks_data']; del P['gs_data']

        map_inputs.append((v, s_o, P))

    #return []
    parfor = MPI_map(verb_parfun, map_inputs, progress_bar=False)
    #print '\n\n== len: {}\n\n'.format([len(abc) for abc in parfor])
    #parfor = map(verb_parfun, map_inputs)

    verbs = {v:verb for v,verb in parfor}
    return verbs




if __name__ == '__main__':



    # ------------------------------------------------------------------------
    # Grid-search parameters

    grid_params = {
        'rank':     [20, 25], #[1, 5, 10, 20, 30, 40, 50, 100],    
        'rho':      [0.9, 0.95]#, 0.85, 0.99],
        # 'eps':      [1e-5, 1e-6, 1e-7],
        # 'stop_t':   [0, 0.003, 0.01, 0.03],
        # 'lamb':     [0.1, 0.2, ...]       # Regularization parameter, when we have that...
    }



    # ------------------------------------------------------------------------
    # Parameters

    params = {   
        'save_file'     : 'data/test-4/grid',
        'grid_file'     : 'data/test-4/grid_accuracy.csv',
        'verbose'       : False,
        'rank'          : 20,
        'batch_size'    : 20,
        'epochs'        : 500,
        'n_trials'      : 120,        # TODO change back to higher number  (also cg, ck)
        'learning_rate' : 1.0,
        'init_noise'    : 0.1,
        'optimizer'     : 'ADAD',  # | 'SGD',
        'rho'           : 0.9,
        'eps'           : 1e-6,
        'cg'            : 0,        # TODO set to 0 for full data,
        'ck'            : 0,        # TODO set to -1 for full data  (minus 1 point),
        'n_stop'        : 0.1,
        'stop_t'        : 0,
    }



    # ------------------------------------------------------------------------
    # Load & filter test data

    gs_file = 'data/eval/GS2011data.txt'
    ks_file = 'data/eval/KS2014.txt'
    gs_data, ks_data, test_vs = load_test_data(params['cg'], params['ck'], 
                                               gs_file=gs_file, ks_file=ks_file)

    # ------------------------------------------------------------------------
    # Load & filter word/triplet vectors

    nn_file  = 'data/w2v/w2v-nouns.npy'
    svo_file = 'data/w2v/w2v-svo-triplets.npy'
    w2v_nn, w2v_svo_full = load_word2vec(test_vs, nn_file=nn_file, svo_file=svo_file)

    params.update({
        'w2v_nn':        w2v_nn,
        'w2v_svo_full':  w2v_svo_full,
        'gs_data':       gs_data,
        'ks_data':       ks_data,
    })

    # ------------------------------------------------------------------------
    # Train / load verb parameters

    # train_trials(params, parallel=True)
    train_trials_grid_parallel(params, grid_params)

    # verbs = load_verbs(params['save_file'] + '.npy')
    # test_verbs(verbs, w2v_nn, gs_data, dset='GS', verbal=True)
    # test_verbs(verbs, w2v_nn, ks_data, dset='KS', verbal=True)








# 
