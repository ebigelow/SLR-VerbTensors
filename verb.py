import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from tqdm import tqdm, trange


class Verb:

    def __init__(self, test_data=None, stop_t=0.01, rank=50, svec=100, nvec=100, 
                 lamb_P=1e-3, lamb_Q=1e-1, lamb_R=1e-1, init_noise=0.1, init_restarts=1):
        self.lamb = (lamb_P, lamb_Q, lamb_R) 
        self.test_data, self.stop_t = (test_data, stop_t)
        self.rank, self.svec, self.nvec = (rank, svec, nvec)
        self.init_weights(init_noise, init_restarts)

    def init_weights(self, init_noise, init_restarts=1):
        best_L = float('inf')
        best_params = ()
        for i in range(init_restarts):
            self.P = init_noise * np.random.rand(self.rank, self.svec)
            self.Q = init_noise * np.random.rand(self.rank, self.svec)
            self.R = init_noise * np.random.rand(self.rank, self.svec)
            L = self.L(*self.test_data)
            if i==0 or L < best_L:
                best_L = L
                best_params = (self.P.copy(), self.Q.copy(), self.R.copy())

        self.P, self.Q, self.R = best_params

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
            epochs=100, batch_size=4, learning_rate=1.0, verbose=False):
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

        loop = trange(epochs) if verbose else range(epochs)
        for e in loop:
            
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
                  rho=0.95, eps=1e-6, verbose=False, norm=None):
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

        if norm is None:
            prox = lambda a, lamb: a
        elif norm == 'L1':
            prox = lambda a, lamb: np.sgn(a) * np.maximum(0, abs(a) - lamb)
        elif norm == 'L2':
            prox = lambda a, lamb: np.maximum(0, 1 - lamb / norm(a, 2))

        loop = trange(epochs) if verbose else range(epochs)
        for e in loop:

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
                    g_ = Qs_Ro.dot(Qs_Ro.T).dot(P)  -  t.dot(Qs_Ro.T).T
                    g = prox(g_)
                    E_g2  = rho * E_g2_prev[0]  +  (1-rho) * g**2
                    dL_dP = -g * np.sqrt(E_dx2_prev[0] + eps) / np.sqrt(E_g2 + eps)
                    E_dx2 = rho * E_dx2_prev[0]  +  (1-rho) * dL_dP**2

                    self.P += lr * dL_dP
                    E_g2_prev[0] = E_g2
                    E_dx2_prev[0] = E_dx2

                elif e % 3 == 1:
                    g_ = ( Ro * (P.dot(P.T).dot(Qs_Ro)  -  P.dot(t)) ).dot(s.T)
                    g = prox(g_)
                    E_g2  = rho * E_g2_prev[1]  +  (1-rho) * g**2
                    dL_dQ = -g * np.sqrt(E_dx2_prev[1] + eps) / np.sqrt(E_g2 + eps)
                    E_dx2 = rho * E_dx2_prev[1]  +  (1-rho) * dL_dQ**2

                    self.Q += lr * dL_dQ
                    E_g2_prev[1] = E_g2
                    E_dx2_prev[1] = E_dx2

                elif e % 3 == 2:
                    g_ = ( Qs * (P.dot(P.T).dot(Qs_Ro)  -  P.dot(t)) ).dot(o.T)
                    g = prox(g_)
                    E_g2  = rho * E_g2_prev[2]  +  (1-rho) * g**2
                    dL_dR = -g * np.sqrt(E_dx2_prev[2] + eps) / np.sqrt(E_g2 + eps)
                    E_dx2 = rho * E_dx2_prev[2]  +  (1-rho) * dL_dR**2

                    self.R += lr * dL_dR
                    E_g2_prev[2] = E_g2
                    E_dx2_prev[2] = E_dx2

            if self.stop_early():
                # print 'stopped at : {}'.format(e)
                break
            elif e % (epochs / 10) == 0:
                print 'epoch: {}   |   L: {}'.format(e, self.prev_loss)

        self.P = self.min_params['P']
        self.Q = self.min_params['Q']
        self.R = self.min_params['R']
        del self.min_params
