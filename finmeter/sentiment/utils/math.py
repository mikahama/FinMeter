"""
math helpers

author: fyl
"""

from .cupy_utils import *


NORM_BATCH_SIZE = 200000
SORT_BATCH_SIZE = 10000
DOT_BATCH_SIZE = 100000


def spectral_norm(X):
    """
    compute the spectral norm of X

    X: ndarray of rank 2

    returns: int
    """
    xp = get_array_module(X)
    _, s, _ = xp.linalg.svd(X)
    return s.max()


def l2norm(X):
    """
    equivalent to np.linalg.norm

    X: ndarray of rank 2

    returns: ndarray of shape (X.shape[0],)
    """
    xp = get_array_module(X)
    xsize = X.shape[0]
    norms = xp.zeros(xsize, dtype=xp.float32)
    for i in range(0, xsize, NORM_BATCH_SIZE):
        j = min(xsize, i + NORM_BATCH_SIZE)
        norms[i:j] = xp.linalg.norm(X[i:j], axis=1)
    return norms


def length_normalize(X, inplace=True):
    """
    Normalize rows of X to unit length.

    X: np.ndarray (or cupy.ndarray)
    inplace: bool

    Returns: None or np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)

    norms = l2norm(X)
    norms[norms == 0.] = 1.
    if inplace:
        X /= norms[:, xp.newaxis]
    else:
        X = X / norms[:, xp.newaxis]
    return X


def mean_center(X, inplace=True):
    """
    X: np.ndarray (or cupy.ndarray)
    inplace: bool

    Returns: None or np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    if inplace:
        X -= xp.mean(X, axis=0)
    else:
        X = X - xp.mean(X, axis=0)
    return X


def normalize(X, actions, inplace=True):
    """
    X: np.ndarray (or cupy.ndarray)
    actions = list[str]
    inplace: bool

    Returns: None or np.ndarray (or cupy.ndarray)
    """
    for action in actions:
        if action == 'unit':
            X = length_normalize(X, inplace)
        elif action == 'center':
            X = mean_center(X, inplace)
    return X


def top_k_mean(X, k, inplace=False):
    """
    Average of top-k similarites.

    X: np.ndarray (or cupy.ndarray)
    k: int
    inplace: bool

    Returns: np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    size = X.shape[0]
    ans = xp.zeros(size, dtype=xp.float32)
    if k == 0:
        return ans
    if not inplace:
        X = X.copy()
    min_val = X.min()
    ind0 = xp.arange(size)
    ind1 = xp.zeros(size, dtype=xp.int32)
    for i in range(k):
        xp.argmax(X, axis=1, out=ind1)
        ans += X[ind0, ind1]
        X[ind0, ind1] = min_val
    ans /= k
    return ans


def dropout(X, keep_prob, inplace=True):
    """
    Randomly set entries of X to zeros.

    X: np.ndarray (or cupy.ndarray)
    keep_prob: float
    inplace: bool

    Returns: np.ndarray (or cupy.ndarray)
    """
    xp = get_array_module(X)
    if keep_prob >= 1.:
        return X if inplace else X.copy()
    mask = xp.random.rand(*X.shape) < keep_prob
    if inplace:
        X *= mask
    else:
        X = X * mask
    return X


def sample(X, Y, num_sample):
    """
    X: ndarray
    Y: ndarray
    num_sample: int

    Returns: ndarray
    """
    xp = get_array_module(X, Y)
    x_idx = xp.random.randint(0, X.shape[0], num_sample)
    y_idx = xp.random.randint(0, Y.shape[0], num_sample)
    return X[x_idx], Y[y_idx]


def sort(X, axis):
    """
    equivalent to np.linalg.norm

    X: ndarray of rank 2

    returns: ndarray of shape (X.shape[0],)
    """
    xp = get_array_module(X)
    xsize = X.shape[0]
    for i in range(0, xsize, SORT_BATCH_SIZE):
        j = min(xsize, i + SORT_BATCH_SIZE)
        X[i:j].sort(axis=axis)


def matmul(X, Y, out=None):
    """
    equivalent to np.matmul

    X: ndarray of rank 2
    Y: ndarray of rank 2

    returns X @ Y
    """
    if out is None:
        raise ValueError('"out" cannot be None')
    xp = get_array_module(X)
    xsize = X.shape[0]
    for i in range(0, xsize, DOT_BATCH_SIZE):
        j = min(xsize, i + DOT_BATCH_SIZE)
        xp.dot(X[i:j], out=out[i:j])
    return res
