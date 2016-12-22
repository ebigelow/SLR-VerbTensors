from utils import *
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from verb import Verb
from tqdm import tqdm, trange


def test_verbs(verbs, w2v_nn, test_data, dset='GS', verbose=False):

    # Predict similarity for GS test data, using learned verb representations
    test_pairs = []
    if verbose: print '\n\nTesting on '+dset+' data . . .'

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
    if verbose: print '\trho: {}\n\tpvalue: {}'.format(rho_, pvalue)
    return rho_, pvalue 


def train_verbs(params):
    # Build and train model for each verb
    verbs = {}

    it = params['w2v_svo'].items()
    loop = tqdm(it, desc='', leave=True) if params['verbose'] else it

    for v, s_o in loop:  
        if params['verbose']:  loop.set_description('Training: "' + v + '"')

        P = params.copy()
        P['test_data'] = params['test_data'][v]
        P['sentences'], P['subjects'], P['objects'] = format_data(P['w2v_nn'], s_o)

        verbs[v] = parameterize(Verb, P)

        if P['optimizer'] == 'SGD':
            parameterize(verbs[v].SGD, P)
        elif P['optimizer'] == 'ADAD':
            parameterize(verbs[v].ADA_delta, P)

    return verbs



def L_combined(verb, train_data, test_ratio):
    L_test  = verb.L(*verb.test_data)  *  test_ratio
    L_train = verb.L(*train_data)      *  (1 - test_ratio)


def get_best_verbs(trained_verbs, train_data, test_ratio):
    all_verbs = defaultdict(lambda: list())
    for verbs in trained_verbs:
        for v, verb in verbs.items():
            all_verbs[v].append(verb)

    for v, verbs in all_verbs.items():
        L = lambda verb: L_combined(verb, train_data, test_ratio)
        all_verbs[v] = sorted(verbs, key=L)

    return {v:verbs[0] for v,verbs in all_verbs.items()}










def train_trials(params):
    """
    Train all verbs `n_trials` individual times.

    """
    best_acc_gs = 0.0
    best_acc_ks = 0.0

    loop = trange if params['verbose'] else range
    trained_verbs = []

    for k in loop(params['n_trials']):
        P = test_to_params(params)

        # Train verbs
        verbs = train_verbs(P)

        # Update saved weights for best-scoring parameters
        curr_acc_gs = test_verbs(verbs, P['w2v_nn'], P['gs_data'], dset='GS', verbose=P['verbose'])[0]
        if curr_acc_gs > best_acc_gs:
            save_verbs(verbs, P['save_file'] + '-GS.npy')
            best_acc_gs = curr_acc_gs

        curr_acc_ks = test_verbs(verbs, P['w2v_nn'], P['ks_data'], dset='KS', verbose=P['verbose'])[0]
        if curr_acc_ks > best_acc_ks:
            save_verbs(verbs, P['save_file'] + '-KS.npy')
            best_acc_ks = curr_acc_ks

        trained_verbs.append(verbs)

    # Save metadata for this run
    save_meta(params, P['save_file'] + '_meta.npy')

    print '\n\n\n~~~~ best individual verbs ~~~~'
    curr_acc_gs = test_verbs(verbs, P['w2v_nn'], P['gs_data'], dset='GS', verbose=P['verbose'])[0]
    curr_acc_ks = test_verbs(verbs, P['w2v_nn'], P['ks_data'], dset='KS', verbose=P['verbose'])[0]



if __name__ == '__main__':



    # ------------------------------------------------------------------------
    # Parameters

    params = {   
        'save_file'     : 'data/nonsparse',
        'verbose'       : True,
        'train'         : True,
        'rank'          : 20,
        'batch_size'    : 20,
        'epochs'        : 500,
        'n_trials'      : 10,
        'learning_rate' : 1.0,
        'init_noise'    : 0.1,
        'optimizer'     : 'ADAD',
        'rho'           : 0.95,
        'eps'           : 1e-6,
        'cg'            : 0,      # set to 0 for full data,
        'ck'            : 0,     # set to -1 for full data  (minus 1 point),
        'n_stop'        : 0.1,
        'data_ratio'    : 0.1,
        'stop_t'        : 1e-6,
        'norm'          : 'L1',
        'lamb_P'        : 1e-2,
        'lamb_Q'        : 1e-0,
        'lamb_R'        : 1e-1,
    }



    # ------------------------------------------------------------------------
    # Load & filter test data

    gs_file = 'data/eval/GS2011data.txt'
    ks_file = 'data/eval/KS2014.txt'
    gs_data, ks_data, test_vs = load_test_data(params['cg'], params['ck'], gs_file=gs_file, ks_file=ks_file)

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

    train_trials(params)

    # verbs = load_verbs(params['save_file'] + '.npy')
    # test_verbs(verbs, w2v_nn, gs_data, dset='GS', verbal=True)
    # test_verbs(verbs, w2v_nn, ks_data, dset='KS', verbal=True)




