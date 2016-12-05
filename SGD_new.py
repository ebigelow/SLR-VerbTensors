# from utils import format_data, parameterize, test_to_params, save_meta, load_test_data, load_word2vec
from utils import *
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from verb import Verb, load_verbs, save_verbs
from tqdm import tqdm, trange


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


def train_verbs(params, verbose=False):
    # Build and train model for each verb
    verbs = {}

    it = params['w2v_svo'].items()
    loop = tqdm(it, desc='', leave=True) if verbose else it

    for v, s_o in tq:  
        if verbose:  tq.set_description('Training: "' + v + '"')

        P = params.copy()
        P['test_data'] = params['test_data'][v]
        P['sentences'], P['subjects'], P['objects'] = format_data(P['w2v_nn'], s_o)

        verbs[v] = parameterize(Verb, P)

        if P['optimizer'] == 'SGD':
            parameterize(verbs[v].SGD, P)
        elif P['optimizer'] == 'ADAD':
            parameterize(verbs[v].ADA_delta, P)

    return verbs







def train_trials(params, parallel=False, verbose=False):
    """
    Train all verbs `n_trials` individual times.

    """
    best_acc_gs = 0.0
    best_acc_ks = 0.0

    for k in trange(params['n_trials']):
        P = test_to_params(params)

        # Train verbs
        #verbs = train_verbs_parallel(P) if parallel else train_verbs(P, verbose=verbose)
        verbs = train_verbs(P, verbose=verbose)
        
        # Update saved weights for best-scoring parameters
        curr_acc_gs = test_verbs(verbs, P['w2v_nn'], P['gs_data'], dset='GS', verbal=True)[0]
        if curr_acc_gs > best_acc_gs:
            save_verbs(verbs, P['save_file'] + '-GS.npy')
            best_acc_gs = curr_acc_gs

        curr_acc_ks = test_verbs(verbs, P['w2v_nn'], P['ks_data'], dset='KS', verbal=True)[0]
        if curr_acc_ks > best_acc_ks:
            save_verbs(verbs, P['save_file'] + '-KS.npy')
            best_acc_ks = curr_acc_ks

    # Save metadata for this run
    save_meta(params, P['save_file'] + '_meta.npy')



if __name__ == '__main__':



    # ------------------------------------------------------------------------
    # Parameters

    params = {   
        'save_file'     : 'data/test-1',
        'train'         : True,
        'rank'          : 5,
        'batch_size'    : 20,
        'epochs'        : 500,
        'n_trials'      : 5,
        'learning_rate' : 1.0,
        'init_noise'    : 0.1,
        'optimizer'     : 'ADAD',  # | 'SGD',
        'rho'           : 0.9,
        'eps'           : 1e-6,
        'cg'            : 5,      # set to 0 for full data,
        'ck'            : 5,     # set to -1 for full data  (minus 1 point),
        'n_stop'        : 0.1,
        'stop_t'        : 0,
        'n_nodes'       : 4,
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




