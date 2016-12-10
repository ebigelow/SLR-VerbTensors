"""


O(G T V E D)

for grid_params:
| for n_trials:
| | for v in verbs:
| | | for epochs:
| | | | for d in data[v]:
| | | | | train = trainer( v, d )
| | | | | train( w )



"""
from utils import *
from SGD_new import *
from verb import *
from SimpleMPI.MPI_map import MPI_map
import itertools, sys, time


def test_row(row, w2v_nn, dset):
    if dset == 'GS':
        pid, v, s, o, t, gt_score, hilo = row[1]
        
        svo1_vec = verbs[v].V(w2v_nn[s], w2v_nn[o])     # "verb"
        svo2_vec = verbs[t].V(w2v_nn[s], w2v_nn[o])     # "landmark" (target)

    elif dset == 'KS':
        pid, s1,v1,o1, s2,v2,o2, gt_score = row[1]
        
        svo1_vec = verbs[v1].V(w2v_nn[s1], w2v_nn[o1])
        svo2_vec = verbs[v2].V(w2v_nn[s2], w2v_nn[o2])

    similarity = 1 - cosine(svo1_vec, svo2_vec)     # TODO: test different distances
    return (similarity, gt_score)


def test_verbs_parallel(verbs, w2v_nn, test_data, dset='GS', verbose=False):

    # Predict similarity for GS test data, using learned verb representations
    test_pairs = []
    if verbose: print '\n\nTesting on '+dset+' data . . .'

    test_fun = lambda row: test_row(row, w2v_nn, dset)
    test_pairs = MPI_map(test_fun, test_data.iterrows(), progress_bar=False)

    # Compute spearman R for full data
    rho_, pvalue = spearmanr(*zip(*test_pairs))
    if verbose: print '\trho: {}\n\tpvalue: {}'.format(rho_, pvalue)
    return rho_, pvalue 


def save_acc(i, verbs, P, best_acc, test_data, dset='GS', parallel=False):
    test = test_verbs_parallel if parallel else test_verbs

    curr_acc = test_verbs(verbs, P['w2v_nn'], test_data, 
                          dset=dset, verbose=P['verbose'])[0]
    if curr_acc > best_acc:
        save_verbs(verbs, '{}/grid-{}_{}.npy'.format(P['out_dir'], i, dset))
        return curr_acc
    else:
        return best_acc


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
            verbs = train_verbs_parallel(P) if parallel else train_verbs(P)

            best_acc_gs = save_acc(i, verbs, P, best_acc_gs, P['gs_data'], dset='GS')
            best_acc_ks = save_acc(i, verbs, P, best_acc_ks, P['ks_data'], dset='KS')

        elapsed = time.time() - t1
        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), 
                          ('id', i) ]  +  [(k,v) for k,v in P.items() if k not in IGNORE]  ))
        pd.DataFrame(rows).to_csv(params['out_dir'] + 'grid_accuracy.csv')
        print '~~~~~ Grid iteration: {}  time: {}    best GS: {}   best KS: {}\n\t{}'.format(i, elapsed, best_acc_gs, best_acc_ks, list(grid_iter))



# ----------------------------------------------------------------------------------------------------



def verb_fun(P):
    return train_verbs(test_to_params(P)), tuple((k,v) for k,v in P.items() if k not in IGNORE)

def train_trials_grid_parallel(params, grid_params):
    """
    Train all verbs, parallelizing trials for each set of parameters.

    """
    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    rows = []

    iter_ = list(enumerate(itertools.product(*it_params)))
    loop1 = tqdm(iter_) if params['verbose'] else iter_

    for i, grid_iter in loop1:
        P = dict(params.items() + list(grid_iter))

        best_acc_gs = 0.0
        best_acc_ks = 0.0

        t1 = time.time()        
        parfor = MPI_map(verb_fun, [P]*P['n_trials'], progress_bar=False)
        t2 = time.time()

        for verbs in parfor:
            best_acc_gs = save_acc(i, verbs, P, best_acc_gs, P['gs_data'], dset='GS', parallel=True)
            best_acc_ks = save_acc(i, verbs, P, best_acc_ks, P['ks_data'], dset='KS', parallel=True)

        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), 
                          ('id', i) ]  +  [(k,v) for k,v in P.items() if k not in IGNORE]  ))
        pd.DataFrame(rows).to_csv(params['out_dir'] + '/grid_accuracy.csv')
        print '~~~~~ Grid iteration: {}/{}  time: {}    best GS: {}   best KS: {}\n\t{}'.format(i, len(iter_)+1, t2-t1, best_acc_gs, best_acc_ks, list(grid_iter))



# ----------------------------------------------------------------------------------------------------



def train_trials_grid_parallel2(params, grid_params):
    """
    Parallelize across trials AND all grid parameter settings.

    I believe this is the fastest, by a decent margin.

    """
    n_trials = params['n_trials']

    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    iter_ = list(itertools.product(*it_params))

    # ----------------------------------------------------------------------------------------
    # Run experiment

    map_P = [i for grid in iter_ for i in [dict(params.items() + list(grid))] * n_trials]
    t1 = time.time()
    parfor = MPI_map(verb_fun, map_P, progress_bar=False)
    t2 = time.time()
    print 'MPI done. Time: {}'.format(t2-t1)

    # ----------------------------------------------------------------------------------------
    # Save best-scoring parameters, record accuracy for each trial

    from collections import defaultdict
    verb_bins = defaultdict(lambda: list())
    for result, par in parfor:
        verb_bins[par].append(result)

    rows = []
    for i, (par, trial_verbs) in enumerate(verb_bins.items()):
        best_acc_gs = 0.0
        best_acc_ks = 0.0

        for j, verbs in enumerate(trial_verbs):
            best_acc_gs = save_acc(i, verbs, params, best_acc_gs, params['gs_data'], dset='GS')
            best_acc_ks = save_acc(i, verbs, params, best_acc_ks, params['ks_data'], dset='KS')

        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), ('id', i)] + list(par)))
        print '~~~~~  best GS: {}   best KS: {}\n\t{}'.format(best_acc_gs, best_acc_ks, par)
    pd.DataFrame(rows).to_csv(params['out_dir'] + '/grid_accuracy.csv')



# ----------------------------------------------------------------------------------------------------



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
    Train 1 parameter grid item at a time, parallelize across verbs.

    

    """
    map_inputs = []
    for v, s_o in params['w2v_svo'].items():
        P = params.copy()

        # Split data to include only what each MPI process needs
        nouns = [n for pair in s_o for n in pair]
        P['w2v_nn'] = {n:vec for n, vec in P['w2v_nn'].items() if n in nouns}
        P['test_data'] = params['test_data'][v]
        del P['w2v_svo']; del P['ks_data']; del P['gs_data']

        map_inputs.append((v, s_o, P))

    # Train verbs in parallel
    parfor = MPI_map(verb_parfun, map_inputs, progress_bar=False)

    verbs = {v:verb for v,verb in parfor}
    return verbs




if __name__ == '__main__':



    # ------------------------------------------------------------------------
    # Grid-search parameters

    grid_params = {
        #'rank':     [1, 5, 10, 20, 30, 40, 50],    
        #'rho':      [0.9, 0.95, 0.99],
        #'init_restarts': [1, 1000],
        #'stop_t':   [0, 0.01, 0.03],
        #'learning_rate': [1.0, 2.0, 3.0],
        'batch_size': [1,5,10,20],
        # 'eps':      [1e-5, 1e-6, 1e-7],
        # 'lamb':     [0.1, 0.2, ...]       # Regularization parameter, when we have that...
    }



    # ------------------------------------------------------------------------
    # Parameters

    params = {   
        'out_dir'       : 'data/out/run-{}',
        'verbose'       : False,
        'rank'          : 20,
        'batch_size'    : 20,
        'epochs'        : 500,
        'n_trials'      : 200,
        'learning_rate' : 2.0,
        'init_noise'    : 0.1,
        'init_restarts' : 1,
        'optimizer'     : 'ADAD',  # | 'SGD',
        'rho'           : 0.95,
        'eps'           : 1e-6,
        'cg'            : 0,
        'ck'            : 0,
        'n_stop'        : 0.1,
        'stop_t'        : 0,
    }

    dir_n = sys.argv[1]
    params['out_dir'] = params['out_dir'].format(dir_n)
    make_path(params['out_dir'])


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

    #train_trials(params, parallel=True)
    # train_trials_grid_parallel(params, grid_params)
    train_trials_grid_parallel2(params, grid_params)


