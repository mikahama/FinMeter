"""
bilingual dictionary induction helpers

author: fyl
"""
import numpy as np
from .cupy_utils import *
from .math import *


def get_projection_matrix(X_src, X_trg, orthogonal, direction='forward', out=None):
    """
    X_src: ndarray
    X_trg: ndarray
    orthogonal: bool
    direction: str
        returns W_src if 'forward', W_trg otherwise
    """
    xp = get_array_module(X_src, X_trg)
    if orthogonal:
        if direction == 'forward':
            u, s, vt = xp.linalg.svd(xp.dot(X_trg.T, X_src))
            W = xp.dot(vt.T, u.T, out=out)
        elif direction == 'backward':
            u, s, vt = xp.linalg.svd(xp.dot(X_src.T, X_trg))
            W = xp.dot(vt.T, u.T, out=out)
    else:
        if direction == 'forward':
            W = xp.dot(xp.linalg.pinv(X_src), X_trg, out=out)
        elif direction == 'backward':
            W = xp.dot(xp.linalg.pinv(X_trg), X_src, out=out)
    return W


def get_unsupervised_init_dict(src_emb, trg_emb, cutoff_size, csls, norm_actions, direction, threshold=-float('inf')):
    """
    Given source embedding and target embedding, return a initial bilingual
    dictionary base on similarity distribution.

    src_emb: ndarray of shape (src_size, vec_dim)
    trg_emb: ndarray of shape (trg_size, vec_dim)
    cutoff_size: int
    csls: int
    norm_actions: list[str]
    direction: str
    threshold: float

    Returns: ndarray of shape (dict_size, 2)
    """
    xp = get_array_module(src_emb, trg_emb)
    sim_size = min(src_emb.shape[0], trg_emb.shape[0], cutoff_size) if cutoff_size > 0 else min(src_emb.shape[0], trg_emb.shape[0])
    u, s, vt = xp.linalg.svd(src_emb[:sim_size], full_matrices=False)
    src_sim = (u * s) @ u.T
    u, s, vt = xp.linalg.svd(trg_emb[:sim_size], full_matrices=False)
    trg_sim = (u * s) @ u.T
    del u, s, vt

    src_sim.sort(axis=1)
    trg_sim.sort(axis=1)
    normalize(src_sim, norm_actions)
    normalize(trg_sim, norm_actions)
    sim = xp.dot(src_sim, trg_sim.T)
    del src_sim, trg_sim
    src_knn_sim = top_k_mean(sim, csls, inplace=False)
    trg_knn_sim = top_k_mean(sim.T, csls, inplace=False)
    sim -= src_knn_sim[:, xp.newaxis] / 2 + trg_knn_sim / 2

    fwd_valid = xp.max(sim, axis=1) > threshold
    bwd_valid = xp.max(sim, axis=0) > threshold

    if direction == 'forward':
        init_dict = xp.stack([xp.arange(sim_size), xp.argmax(sim, axis=1)], axis=1)[fwd_valid]
    elif direction == 'backward':
        init_dict = xp.stack([xp.argmax(sim, axis=0), xp.arange(sim_size)], axis=1)[bwd_valid]
    elif direction == 'union':
        init_dict = xp.stack([xp.concatenate((xp.arange(sim_size)[fwd_valid], xp.argmax(sim, axis=0)[bwd_valid])),
                              xp.concatenate((xp.argmax(sim, axis=1)[fwd_valid], xp.arange(sim_size)[bwd_valid]))], axis=1)
    return init_dict


class VIArray(object):

    def __init__(self, X, vocab):
        if vocab.shape[0] != X.shape[0]:
            raise ValueError('Array sizes do not match')
        xp = get_array_module(X, vocab)
        self.vocab = vocab.copy()
        self.X = X.copy()
        nmax = int(self.vocab.max())
        self.translate = xp.full(nmax + 1, nmax + 1, dtype=xp.int32)
        self.translate[self.vocab] = xp.arange(self.vocab.shape[0])

    def __getitem__(self, key):
        return self.X[self.translate[key]]


class BDI(object):
    """
    Helper class for bilingual dictionary induction.

    Parameters
    ----------
    src_emb: np.ndarray of shape (src_emb_size, vec_dim)
    trg_emb: np.ndarray of shape (trg_emb_size, vec_dim)
    batch_size: int
    scorer: str, (dot / cos / euclidean)
    """

    def __init__(self, src_emb, trg_emb, batch_size=5000, cutoff_size=10000, cutoff_type='both',
                 direction=None, csls=10, batch_size_val=1000, scorer='dot',
                 src_val_ind=None, trg_val_ind=None):
        if cutoff_type == 'oneway' and csls > 0:
            raise ValueEror("cutoff_type='both' and csls > 0 not supported")  # TODO
        if scorer not in ('dot', 'cos', 'euclidean'):
            raise ValueError('Invalid scorer: %s' % scorer)

        xp = get_array_module(src_emb, trg_emb, src_val_ind, trg_val_ind)
        self.xp = xp
        self.trg_emb = xp.array(trg_emb, dtype=xp.float32)
        self.batch_size = batch_size
        self.cutoff_size = cutoff_size
        self.cutoff_type = cutoff_type
        self.direction = direction
        self.csls = csls
        self.batch_size_val = batch_size_val
        self.scorer = scorer

        self.src_size = src_emb.shape[0]
        self.trg_size = trg_emb.shape[0]
        self.W_src = self.W_trg = xp.identity(src_emb.shape[1], dtype=xp.float32)

        if src_val_ind is None:
            src_val_ind = xp.arange(src_size)
        src_val_ind = np.union1d(asnumpy(src_val_ind), np.arange(self.cutoff_size))
        if trg_val_ind is None:
            trg_val_ind = xp.arange(trg_size)
        trg_val_ind = np.union1d(asnumpy(trg_val_ind), np.arange(self.cutoff_size))
        self.src_val_ind = xp.array(src_val_ind)
        self.trg_val_ind = xp.array(trg_val_ind)

        self.src_emb = VIArray(xp.array(src_emb[src_val_ind], dtype=xp.float32), xp.array(src_val_ind, dtype=xp.int32))
        self.src_proj_emb = VIArray(xp.array(src_emb[src_val_ind], dtype=xp.float32), xp.array(src_val_ind, dtype=xp.int32))
        self.trg_proj_emb = self.trg_emb.copy()

        if direction in ('forward', 'union') or csls > 0:
            self.fwd_src_size = cutoff_size
            self.fwd_trg_size = cutoff_size if cutoff_type == 'both' else trg_size
            self.fwd_ind = xp.arange(self.fwd_src_size, dtype=xp.int32)
            self.fwd_trg = xp.empty(self.fwd_src_size, dtype=xp.int32)
            self.fwd_sim = xp.empty((batch_size, self.fwd_trg_size), dtype=xp.float32)
            self.best_fwd_sim = xp.empty(self.fwd_src_size)
        if direction in ('backward', 'union') or csls > 0:
            self.bwd_trg_size = cutoff_size
            self.bwd_src_size = cutoff_size if cutoff_type == 'both' else src_size
            self.bwd_ind = xp.arange(self.bwd_trg_size, dtype=xp.int32)
            self.bwd_src = xp.arange(self.bwd_trg_size, dtype=xp.int32)
            self.bwd_sim = xp.empty((batch_size, self.bwd_src_size), dtype=xp.float32)
            self.best_bwd_sim = xp.empty(self.bwd_trg_size)
        self.sim_val = xp.empty((batch_size_val, self.trg_size), dtype=xp.float32)
        self.dict_size = cutoff_size * 2 if direction == 'union' else cutoff_size
        self.dict = xp.empty((self.dict_size, 2), dtype=xp.int32)

        self.trg_sqr_norm = xp.ones(self.trg_size, dtype=xp.float32)
        self.src_sqr_norm = xp.ones(self.bwd_src_size, dtype=xp.float32)

        self.src_avr_norm = xp.mean(l2norm(self.src_emb[:self.cutoff_size]))
        self.trg_avr_norm = xp.mean(l2norm(self.trg_emb[:self.cutoff_size]))
        self.src_factor = 1
        self.trg_factor = 1

        if direction in ('forward', 'union'):
            self.fwd_knn_sim = xp.zeros(self.fwd_trg_size, dtype=xp.float32)
        if direction in ('backward', 'union'):
            self.bwd_knn_sim = xp.zeros(self.bwd_src_size, dtype=xp.float32)

        self.project(xp.identity(src_emb.shape[1], dtype=xp.float32), 'forward')
        self.project(xp.identity(trg_emb.shape[1], dtype=xp.float32), 'backward', full_trg=True)

    def project(self, W, direction='backward', unit_norm=False, scale=False, full_trg=False):
        """
        W_target: ndarray of shape (vec_dim, vec_dim)
        unit_norm: bool

        Returns: self
        """
        xp = self.xp
        if direction == 'forward':
            xp.dot(self.src_emb.X, W, out=self.src_proj_emb.X)

            self.W_src = W.copy()
            if unit_norm:
                length_normalize(self.src_proj_emb.X, inplace=True)
            if scale:
                avr_norm = xp.mean(l2norm(self.src_proj_emb[:self.cutoff_size]))
                self.src_factor = self.src_avr_norm / avr_norm
                self.src_proj_emb.X *= self.src_factor
        else:
            # proj_size = self.trg_size if full_trg else self.cutoff_size
            proj_ind = xp.arange(self.trg_size) if full_trg else self.trg_val_ind
            if full_trg:
                # matmul(self.trg_emb[proj_ind], W, out=self.trg_proj_emb[proj_ind])
                xp.dot(self.trg_emb, W, out=self.trg_proj_emb)
            else:
                self.trg_proj_emb[proj_ind] = xp.dot(self.trg_emb[proj_ind], W)
            self.W_trg = W.copy()
            if unit_norm:
                if full_trg:
                    length_normalize(self.trg_proj_emb, inplace=True)
                else:
                    self.trg_proj_emb[proj_ind] = length_normalize(self.trg_proj_emb[proj_ind], inplace=False)
            if scale:
                avr_norm = xp.mean(l2norm(self.trg_proj_emb[:self.cutoff_size]))
                self.trg_factor = self.trg_avr_norm / avr_norm
                self.trg_proj_emb[proj_ind] *= self.trg_factor
        return self

    def get_bilingual_dict_with_cutoff(self, keep_prob=1.):
        """
        keep_prob: float

        Returns: ndarray of shape (dict_size, 2)
        """
        xp = self.xp
        if self.direction in ('forward', 'union'):
            if self.scorer in ('cos', 'euclidean'):
                xp.sum(self.trg_proj_emb[:self.fwd_trg_size]**2, axis=1, out=self.trg_sqr_norm[:self.fwd_trg_size])
                self.trg_sqr_norm[:self.fwd_trg_size][self.trg_sqr_norm[:self.fwd_trg_size] == 0] = 1

            if self.csls > 0:
                for i in range(0, self.fwd_trg_size, self.batch_size):
                    j = min(self.fwd_trg_size, i + self.batch_size)
                    xp.dot(self.trg_proj_emb[i:j], self.src_proj_emb[:self.fwd_src_size].T, out=self.bwd_sim[:j - i])
                    self.fwd_knn_sim[i:j] = top_k_mean(self.bwd_sim[:j - i], self.csls, inplace=True)

            for i in range(0, self.fwd_src_size, self.batch_size):
                j = min(self.fwd_src_size, i + self.batch_size)
                xp.dot(self.src_proj_emb[i:j], self.trg_proj_emb[:self.fwd_trg_size].T, out=self.fwd_sim[:j - i])

                self.fwd_sim[:j - i].max(axis=1, out=self.best_fwd_sim[i:j])

                self.fwd_sim[:j - i] -= self.fwd_knn_sim / 2
                if self.scorer == 'cos':
                    self.fwd_sim[:j - i] /= xp.sqrt(self.trg_sqr_norm[:self.fwd_trg_size])
                elif self.scorer == 'euclidean':
                    self.fwd_sim[:j - i] -= self.trg_sqr_norm[:self.fwd_trg_size]
                dropout(self.fwd_sim[:j - i], keep_prob, inplace=True).argmax(axis=1, out=self.fwd_trg[i:j])

        if self.direction in ('backward', 'union'):
            if self.scorer in ('cos', 'euclidean'):
                xp.sum(self.src_proj_emb[:self.bwd_src_size]**2, axis=1, out=self.src_sqr_norm[:self.bwd_src_size])
                self.src_sqr_norm[:self.bwd_src_size][self.src_sqr_norm[:self.bwd_src_size] == 0] = 1

            if self.csls > 0:
                for i in range(0, self.bwd_src_size, self.batch_size):
                    j = min(self.bwd_src_size, i + self.batch_size)
                    xp.dot(self.src_proj_emb[i:j], self.trg_proj_emb[:self.bwd_trg_size].T, out=self.fwd_sim[:j - i])
                    self.bwd_knn_sim[i:j] = top_k_mean(self.fwd_sim[:j - i], self.csls, inplace=True)

            for i in range(0, self.bwd_trg_size, self.batch_size):
                j = min(self.bwd_trg_size, i + self.batch_size)
                xp.dot(self.trg_proj_emb[i:j], self.src_proj_emb[:self.bwd_src_size].T, out=self.bwd_sim[:j - i])

                self.bwd_sim[:j - i].max(axis=1, out=self.best_bwd_sim[i:j])

                self.bwd_sim[:j - i] -= self.bwd_knn_sim / 2
                if self.scorer == 'cos':
                    self.bwd_sim[:j - i] /= xp.sqrt(self.src_sqr_norm[:self.bwd_src_size])
                elif self.scorer == 'euclidean':
                    self.bwd_sim[:j - i] -= self.src_sqr_norm[:self.bwd_src_size] / 2
                dropout(self.bwd_sim[:j - i], keep_prob, inplace=True).argmax(axis=1, out=self.bwd_src[i:j])
        if self.direction == 'forward':
            xp.stack([self.fwd_ind, self.fwd_trg], axis=1, out=self.dict)
            self.objective = self.best_fwd_sim.mean()
        elif self.direction == 'backward':
            xp.stack([self.bwd_src, self.bwd_ind], axis=1, out=self.dict)
            self.objective = self.best_bwd_sim.mean()
        elif self.direction == 'union':
            self.dict[:, 0] = xp.concatenate((self.fwd_ind, self.bwd_src))
            self.dict[:, 1] = xp.concatenate((self.fwd_trg, self.bwd_ind))
            self.objective = (self.best_fwd_sim.mean() + self.best_bwd_sim.mean()) / 2
        return self.dict.copy()

    def get_target_indices(self, src_ind):
        """
        src_ind: np.ndarray of shape (dict_size,)

        Returns: np.ndarray of shape (dict_size,)
        """
        xp = self.xp
        size = src_ind.shape[0]
        trg_ind = xp.empty(size, dtype=xp.int32)
        xsrc = self.src_proj_emb[src_ind]
        if self.scorer in ('cos', 'euclidean'):
            xp.sum(self.trg_proj_emb**2, axis=1, out=self.trg_sqr_norm)
            self.trg_sqr_norm[self.trg_sqr_norm == 0] = 1
        for i in range(0, size, self.batch_size_val):
            j = min(i + self.batch_size_val, size)
            xp.dot(xsrc[i:j], self.trg_proj_emb.T, out=self.sim_val[: j - i])
            if self.scorer == 'cos':
                self.sim_val[: j - i] /= self.trg_sqr_norm
            elif self.scorer == 'euclidean':
                self.sim_val[: j - i] -= self.trg_sqr_norm / 2
            xp.argmax(self.sim_val[:j - i], axis=1, out=trg_ind[i:j])
        return trg_ind
