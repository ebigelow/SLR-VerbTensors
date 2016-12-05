from SGD_real import *
from pyina.ez_map import ez_map
# TODO: install dill, mpi4py, pyina

IGNORE = ['w2v_nn', 'w2v_svo', 'w2v_svo_full', 'test_data', 'gs_data', 'ks_data']



def parameterize(func, params_orig):
    func_args = func.__code__.co_varnames

    params = params_orig.copy()
    for p in params:
        if p not in func_args:
            del params[p]

    return func(**params)




def test_to_params(params_orig):
    params = params_orig.copy()
    w2v_svo_full, n_stop = (params['w2v_svo_full'], params['n_stop'])
    w2v_svo, w2v_svo_test = split_test(w2v_svo_full, n_stop=n_stop)
    test_data = {k:format_data(w2v_nn, s_o) for k,s_o in w2v_svo_test.items()}

    params['w2v_svo']   = w2v_svo
    params['test_data'] = test_data
    del params['w2v_svo_full']

    return params




def train_grid(params_orig, grid_params):

    df = pd.DataFrame()
    it_params = [zip([k]*len(v), v) for k, v in grid_params.items()]

    for i, grid_iter in tqdm(list(itertools.product(*it_params))):
        params_iter = dict(grid_iter + params_orig.items())

        best_acc_gs = 0.0
        best_acc_ks = 0.0
        for k in trange(n_trials):
            params = test_to_params(**params_iter)

            verbs = []
            verbs = parameterize(train_verbs, params)

            curr_acc_gs = test_verbs(verbs, params['w2v_nn'], params['gs_data'], dset='GS')[0]
            if curr_acc_gs > best_acc_gs:
                save_verbs(verbs, save_file + '-GS-{}.npy'.format(i))
                # save_meta(save_file + '-GS-{}_meta.npy'.format(i))

            curr_acc_ks = test_verbs(verbs, params['w2v_nn'], params['ks_data'], dset='KS')[0]
            if curr_acc_ks > best_acc_ks:
                save_verbs(verbs, save_file + '-KS-{}.npy'.format(i))
                # save_meta(save_file + '-KS-{}_meta.npy'.format(i))

        df.addrow(dict([('accuracy_GS', best_acc_gs), ('accuracy_KS', best_acc_ks), 
                        ('id', i) ]  +  [(k,v) for k,v in params.items() if k not in IGNORE]  ))
        pd.save_TODO(df, TODO_FILENAME)
        # TODO: filename



def train_standard(params):
    best_acc_gs = 0.0
    best_acc_ks = 0.0

    for k in trange(n_trials):
        P = test_to_params(params)

        # Train verbs
        verbs = parameterize(train_verbs, P)

        # Update saved weights for best-scoring parameters
        curr_acc_gs = test_verbs(verbs, P['w2v_nn'], P['gs_data'], dset='GS')[0]
        if curr_acc_gs > best_acc_gs:
            save_verbs(verbs, save_file + '-GS.npy')

        curr_acc_ks = test_verbs(verbs, P['w2v_nn'], P['ks_data'], dset='KS')[0]
        if curr_acc_ks > best_acc_ks:
            save_verbs(verbs, save_file + '-KS.npy')

    # Save metadata for this run
    save_meta(params, save_file + '_meta.npy')



def save_meta(params, fname):
    d = {k:v for k,v in params.items() if k not in IGNORE}
    np.save(fname, d)


def verb_parfun(v, s_o, params):
    """Build and train model for each verb"""
    P = params
    # P['test_data'] = test_data[v]
    verb = parameterize(P, Verb)
    P['sentences'],  P['subjects'], P['objects'] = format_data(w2v_nn, s_o)

    if optimizer == 'SGD':
        parameterize(P, verb.SGD)
    elif optimizer == 'ADAD':
        parameterize(P, verb.ADA_delta)

    verb.v = v
    return verb


def train_verbs_parallel(params):
    # Split test data before, to minimize RAM overhead
    map_inputs = []
    for v, s_o in w2v_svo.items():
        P = params.copy()
        P['test_data'] = test_data[v]
        map_inputs.append((v, s_o, P))

    parfor = ez_map(verb_parfun, map_inputs, nnodes=n_nodes)

    verbs = {verb.v:verb for verb in parfor}
    return verbs





if __name__ == '__main__':



    # ------------------------------------------------------------------------
    # Parameters

    params = {   
        'save_file'     : 'data/grid1/run',
        'train'         : True,
        'rank'          : 5,
        'batch_size'    : 20,
        'epochs'        : 500,
        'n_trials'      : 5,
        'learning_rate' : 1.0,
        'init_noise'    : 0.1,
        'optimizer'     : 'ADAD'  # | 'SGD',
        'rho'           : 0.9,
        'eps'           : 1e-6,
        'cg'            : 0,      # set to 0 for full data,
        'ck'            : -1,     # set to -1 for full data  (minus 1 point),
        'n_stop'        : 0.1,
        'stop_t'        : 0,
        'n_nodes'       : 4,
    }


    # ------------------------------------------------------------------------
    # Grid-search parameters

    grid_params = {
        'rank':     [1, 5, 10, 20, 30, 40, 50, 100],    
        'rho':      [0.85, 0.9, 0.95, 0.99],
        'eps':      [1e-5, 1e-6, 1e-7],
        'stop_t':   [0, 0.003, 0.01, 0.03],
        # 'lamb':     [0.1, 0.2, ...]       # Regularization parameter, when we have that...
    }




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

    params += {
        'w2v_nn':        w2v_nn,
        'w2v_svo_full':  w2v_svo_full,
        'test_data':     test_data,     # 10% triplets used for early stopping
        'gs_data':       gs_data,
        'ks_data':       ks_data,
    }

    # ------------------------------------------------------------------------
    # Train / load verb parameters

    if train:
        TODO


    else:
        verbs = load_verbs(save_file + '.npy')

    test_verbs(verbs, w2v_nn, gs_data, dset='GS', verbal=True)
    test_verbs(verbs, w2v_nn, ks_data, dset='KS', verbal=True)




