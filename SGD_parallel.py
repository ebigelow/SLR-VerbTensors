"""


O(G T V E D)

for grid_params:
| for n_trials:
| | for v in verbs:
| | | for epochs:
| | | | for d in data[v]:
| | | | | train = trainer( v, d )
| | | | | train( w )



train_trials_grid_parallel
--------------------------

for grid_params:
| for n_trials:
| | for v in verbs:
| | * for epochs:
| | * | for d in data[v]:
| | * | | train = trainer( v, d )
| | * | | train( w )




train_trials_grid_parallel2
---------------------------

for grid_params:
| parfor n_trials:
| * for v in verbs:
| * | for epochs:
| * | | for d in data[v]:
| * | | | train = trainer( v, d )
| * | | | train( w )


train_trials_grid_parallel3
---------------------------

parfor grid_params, n_trials:
* for v in verbs:
* | for epochs:
* | | for d in data[v]:
* | | | train = trainer( v, d )
* | | | train( w )


"""
from utils import *
from SGD_new import *
from verb import Verb
from SimpleMPI.MPI_map import MPI_map
import itertools, os, sys, time
from itertools import product as combinations
from collections import defaultdict


def save_acc_par(i, verbs, P, test_data, dset='GS'):
    curr_acc = test_verbs(verbs, P['w2v_nn'], test_data, 
                          dset=dset, verbose=P['verbose'])[0]
    
    save_verbs(verbs, '{}/grid-{}_{}.npy'.format(P['out_dir'], i, dset))
    return verbs, curr_acc


def save_acc(i, verbs, P, best_acc, test_data, dset='GS'):
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

    iter_ = enumerate(combinations(*it_params))
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
        print '\n~~~~~ Grid iteration: {}  time: {}    best GS: {}   best KS: {}\n\t{}'.format(i, elapsed, best_acc_gs, best_acc_ks, list(grid_iter))



# ----------------------------------------------------------------------------------------------------



def L_combined(verb, train_data, test_ratio):
    L_test  = verb.L(*verb.test_data)  *  test_ratio
    L_train = verb.L(*train_data)      *  (1 - test_ratio)


def get_best_verbs(trained_verbs, all_data): #, test_ratio):
    all_verbs = defaultdict(lambda: list())
    for verbs in trained_verbs:
        for v, verb in verbs.items():
            all_verbs[v].append(verb)

    for v, verbs in all_verbs.items():
        ## L = lambda verb: L_combined(verb, train_data, test_ratio)
        L = lambda verb: verb.L(*all_data)
        all_verbs[v] = sorted(verbs, key=L)

    return {v:verbs[0] for v,verbs in all_verbs.items()}






def verb_fun(P):
    return train_verbs(test_to_params(P)), par2tuple(P)

def train_trials_grid_parallel(params, grid_params):
    """
    Train all verbs, parallelizing trials for each set of parameters.

    """
    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    rows = []

    iter_ = list(enumerate(combinations(*it_params)))
    loop1 = tqdm(iter_) if params['verbose'] else iter_

    for i, grid_iter in loop1:
        P = dict(params.items() + list(grid_iter))

        best_acc_gs = 0.0
        best_acc_ks = 0.0

        t1 = time.time()        
        parfor = MPI_map(verb_fun, [P]*P['n_trials'], progress_bar=False)
        t2 = time.time()

        for verbs in parfor:
            best_acc_gs = save_acc(i, verbs, P, best_acc_gs, P['gs_data'], dset='GS')
            best_acc_ks = save_acc(i, verbs, P, best_acc_ks, P['ks_data'], dset='KS')

        # -------------------------------------------
        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), 
                          ('id', i) ]  +  [(k,v) for k,v in P.items() if k not in IGNORE]  ))
        pd.DataFrame(rows).to_csv(params['out_dir'] + '/grid_accuracy.csv')
        print '\n~~~~~ Grid iteration: {}/{}  time: {}    best GS: {}   best KS: {}\n\t{}'.format(i, len(iter_)+1, t2-t1, best_acc_gs, best_acc_ks, list(grid_iter))




# ----------------------------------------------------------------------------------------------------



def train_trials_grid_parallel2(params, grid_params):
    """
    Parallelize across trials AND all grid parameter settings.

    I believe this is the fastest, by a decent margin.

    """
    n_trials = params['n_trials']

    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    iter_ = list(combinations(*it_params))

    # ----------------------------------------------------------------------------------------
    # Run experiment

    map_P = [i for grid in iter_ for i in [dict(params.items() + list(grid))] * n_trials]
    t1 = time.time()
    parfor = MPI_map(verb_fun, map_P, progress_bar=False)
    t2 = time.time()
    print 'Training done. Time: {}\n'.format(t2-t1)

    # ----------------------------------------------------------------------------------------
    # Save best-scoring parameters, record accuracy for each trial

    from collections import defaultdict
    verb_bins = defaultdict(lambda: list())
    for result, par in parfor:
        verb_bins[par].append(result)

    rows = []
    for i, (par, trial_verbs) in enumerate(verb_bins.items()):
        ##best_acc_gs = 0.0
        ##best_acc_ks = 0.0
        ##
        ##for j, verbs in enumerate(trial_verbs):
        ##    best_acc_gs = save_acc(i, verbs, params, best_acc_gs, params['gs_data'], dset='GS')
        ##    best_acc_ks = save_acc(i, verbs, params, best_acc_ks, params['ks_data'], dset='KS')


        # ----------------------------------------------------------------------------------------
        # print '\n\n\n~~~~ best individual verbs ~~~~'
        
        all_data = format_data(params['w2v_nn'], params['w2v_svo_full'])
        best_verbs = get_best_verbs(parfor, all_data)
        
        best_acc_gs = save_acc(best_verbs, P['w2v_nn'], P['gs_data'], dset='GS', verbose=P['verbose'])[0]
        best_acc_ks = save_acc(best_verbs, P['w2v_nn'], P['ks_data'], dset='KS', verbose=P['verbose'])[0]
        
        # ----------------------------------------------------------------------------------------

        # Append row with metadata and accuracy
        rows.append(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), ('id', i)] + list(par)))
        print '\n~~~~~  best GS: {}   best KS: {}\n\t{}'.format(best_acc_gs, best_acc_ks, par)
    pd.DataFrame(rows).to_csv(params['out_dir'] + '/grid_accuracy.csv')


# ----------------------------------------------------------------------------------------------------
# parallel2 + parallel testing






def train_trials_grid_parallel3(params, grid_params):
    """
    Parallelize across trials AND all grid parameter settings.

    I believe this is the fastest, by a decent margin.

    """
    n_trials = params['n_trials']

    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    iter_ = list(combinations(*it_params))

    # ----------------------------------------------------------------------------------------
    # Run experiment

    map_P = [i for grid in iter_ for i in [dict(params.items() + list(grid))] * n_trials]
    t1 = time.time()
    parfor = MPI_map(verb_fun, map_P, progress_bar=False)
    t2 = time.time()
    print 'Training done. Time: {}\n'.format(t2-t1)

    # ----------------------------------------------------------------------------------------
    # Save best-scoring parameters, record accuracy for each trial

    from collections import defaultdict
    verb_bins = defaultdict(lambda: list())
    for result, par in parfor:
        verb_bins[par].append(result)

    rows = []
    for i, (par, trial_verbs) in enumerate(verb_bins.items()):

        gs_inputs = [[i] + [verbs] + [params, params['gs_data'], 'GS'] for verbs in trial_verbs]
        ks_inputs = [[i] + [verbs] + [params, params['ks_data'], 'KS'] for verbs in trial_verbs]
        #save_acc_par(i, verbs, P, test_data, dset='GS')

        t1 = time.time()
        gs_verbs, gs_acc = sorted( MPI_map(save_acc_par, gs_inputs, progress_bar=False), key=lambda x: -x[1])[0]
        ks_verbs, ks_acc = sorted( MPI_map(save_acc_par, ks_inputs, progress_bar=False), key=lambda x: -x[1])[0]
        t2 = time.time()

        save_acc(i, gs_verbs, params, gs_acc, params['gs_data'], dset='GS')
        save_acc(i, ks_verbs, params, ks_acc, params['ks_data'], dset='KS')

        rows.append(dict([('accuracy_GS', gs_acc), ('accuracy_KS', ks_acc), ('id', i)] + list(par)))
        print '\n~~~~~  best GS: {}   best KS: {}\nTime: {}\n\t{}'.format(gs_acc, ks_acc, t2-t1, par)

    pd.DataFrame(rows).to_csv(params['out_dir'] + '/grid_accuracy.csv')





# ----------------------------------------------------------------------------------------------------
# + saving to temp files to reduce RAM


def verb_fun2(params):
    P = test_to_params(params)
    verbs = train_verbs(P)
    save_verbs(verbs, P['temp_file'])
    # This is to remove( non-params from P
    return dict(par2tuple(P))


def train_trials_grid_parallel4(params, grid_params):
    """
    Same as 3 but with saving to temp files to drastically reduce RAM with big experiments.

    """
    n_trials = params['n_trials']

    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]
    iter_ = enumerate(combinations(*it_params))

    # ----------------------------------------------------------------------------------------
    # Run experiment

    # Prepare list of parameters to send to parallel process
    map_P = [dict(i) for idx, grid in iter_ for i in [params.items() + list(grid) + [('grid_idx', idx)]] * n_trials]
    for mpi_idx, P in enumerate(map_P):
        P['mpi_idx'] = mpi_idx
        P['temp_file'] = P['temp_file'].format(P['grid_idx'], mpi_idx)

    # Save all parameters so if it cuts out early, we can still use the temp files
    params_file = params['out_dir'].format('grid-temps')
    np.save(params_file, [dict(par2tuple(P)) for P in map_P])

    t1 = time.time()
    parfor = MPI_map(verb_fun2, map_P, progress_bar=False)
    t2 = time.time()
    print 'Training done. Time: {}\n'.format(t2-t1)

    # ----------------------------------------------------------------------------------------
    # Save best-scoring parameters, record accuracy for each trial

    verb_bins = defaultdict(lambda: list())
    grid_pars = {}
    for P in parfor:  
        grid_idx = P['grid_idx']
        temp_file = P['temp_file']
        verb_bins[grid_idx].append(temp_file)
        grid_pars[grid_idx] = P
        for k in ['grid_idx', 'mpi_idx', 'temp_file']:
            del grid_pars[grid_idx][k]

    rows = []
    for i, temp_files in verb_bins.items():

        trial_verbs = [load_verbs(t) for t in temp_files]

        gs_inputs = [[i] + [verbs] + [params, params['gs_data'], 'GS'] for verbs in trial_verbs]
        ks_inputs = [[i] + [verbs] + [params, params['ks_data'], 'KS'] for verbs in trial_verbs]

        t1 = time.time()
        gs_verbs, gs_acc = sorted( MPI_map(save_acc_par, gs_inputs, progress_bar=False), key=lambda x: -x[1])[0]
        ks_verbs, ks_acc = sorted( MPI_map(save_acc_par, ks_inputs, progress_bar=False), key=lambda x: -x[1])[0]
        t2 = time.time()

        save_acc(i, gs_verbs, params, gs_acc, params['gs_data'], dset='GS')
        save_acc(i, ks_verbs, params, ks_acc, params['ks_data'], dset='KS')

        P = par2tuple(grid_pars[i])
        rows.append(dict([('accuracy_GS', gs_acc), ('accuracy_KS', ks_acc), ('id', i)] + list(P)))
        print '\n~~~~~  best GS: {}   best KS: {}\nTime: {}\n\t{}'.format(gs_acc, ks_acc, t2-t1, P)

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
        'norm':     ['L1', 'L2'],
        'lamb_P':   [1., 1e-1, 1e-2],
        'lamb_Q':   [1., 1e-1, 1e-2],
        'lamb_R':   [1., 1e-1, 1e-2],
        'rank':     [10, 20, 40],
        #'rho':      [0.9, 0.95, 0.99],
        #'init_restarts': [1, 1000],
        #'stop_t':   [0, 1e-6],
        #'learning_rate': [1.0, 2.0, 3.0],
        #'batch_size': [20, 50],
        # 'eps':      [1e-5, 1e-6, 1e-7],
        # 'lamb':     [0.1, 0.2, ...]       # Regularization parameter, when we have that...
    }



    # ------------------------------------------------------------------------
    # Parameters

    params = {   
        'out_dir'       : 'data/out/run-{}',
        'temp_dir'      : 'data/temp-{}',
        'verbose'       : False,
        'rank'          : 20,
        'batch_size'    : 20,
        'epochs'        : 500,
        'n_trials'      : 300,
        'learning_rate' : 2.0,
        'init_noise'    : 0.1,
        'init_restarts' : 1,
        'optimizer'     : 'ADAD',  # | 'SGD',
        'rho'           : 0.95,
        'eps'           : 1e-6,
        'cg'            : 0,
        'ck'            : 0,
        'n_stop'        : 0.1,
        'stop_t'        : 1e-9,
        'norm'          : 'L1',
        'lamb_P'        : 0,
        'lamb_Q'        : 1e-2,
        'lamb_R'        : 1e-2,
    }

    # Slurm job number as command-line arg
    dir_n = sys.argv[1]

    params['out_dir'] = params['out_dir'].format(dir_n)
    make_path(params['out_dir'])


    params['temp_dir'] = params['temp_dir'].format(dir_n) 
    params['temp_file'] = params['temp_dir'] + '/{}-{}.npy'
    make_path(params['temp_dir'])



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
    #train_trials_grid_parallel(params, grid_params)
    #train_trials_grid_parallel2(params, grid_params)
    #train_trials_grid_parallel3(params, grid_params)
    train_trials_grid_parallel4(params, grid_params)

    print '\nTesting done, clearing temp directory'
    os.system('rm -r ' + params['temp_dir'])

