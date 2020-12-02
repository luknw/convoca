from itertools import product, count

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import time
from IPython.display import clear_output


def fixed_aspect_ratio(ratio):
    """
    Set a fixed aspect ratio on matplotlib plots
    regardless of axis units
    """
    xvals, yvals = (plt.gca().axes.get_xlim(),
                    plt.gca().axes.get_ylim())

    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    plt.gca().set_aspect(ratio * (xrange / yrange), adjustable='box')


def better_savefig(name, dpi=72, pad=0.0, remove_border=True):
    """
    This function is for saving images without a bounding box and at the proper resolution
        The tiff files produced are huge because compression is not supported py matplotlib


    name : str
        The string containing the name of the desired save file and its resolution

    dpi : int
        The desired dots per linear inch

    pad : float
        Add a tiny amount of whitespace if necessary

    remove_border : bool
        Whether to remove axes and padding (for example, for images)

    """
    if remove_border:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1 + pad, bottom=0 + pad, right=1 + pad, left=0 + pad,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=dpi)


def cmap1D(all_col, N):
    """Generate a continuous colormap between two values

    Parameters
    ----------

    all_col : list of 3-tuples
        The colors to linearly interpolate

    N : int
        The number of values to interpolate

    Returns
    -------

    col_list : list of tuples
        An ordered list of colors for the colormap

    """

    n_col = len(all_col)
    all_col = [np.array([item / 255. for item in col]) for col in all_col]

    all_vr = list()
    runlens = [len(thing) for thing in np.array_split(range(N), n_col - 1)]
    for col1, col2, runlen in zip(all_col[:-1], all_col[1:], runlens):
        vr = list()
        for ii in range(3):
            vr.append(np.linspace(col1[ii], col2[ii], runlen))
        vr = np.array(vr).T
        all_vr.extend(vr)
    return [tuple(thing) for thing in all_vr]


def tup2str(tup, delim=''):
    """Convert a tuple to an ordered string"""
    return delim.join([str(item) for item in tup])


def get_slope(vec):
    m, b = np.polyfit(np.arange(0, len(vec)), vec, 1)
    return m, b


def bin2int(arr, axis=0):
    """
    Convert a binary array to an integer along the 
    specified axis

    Dev: this overflows when the size of the numbers is greater
    than 64 bits
    """
    pow2 = 2 ** np.arange(arr.shape[axis], dtype=np.uint64)
    return np.sum(arr * pow2, axis=axis).astype(int)


def all_combinations(m, shape):
    """
    Make an array of all `shape` dimensional inputs
    consisting of `m` possible values
    """
    indices = np.tile(np.array([np.arange(m)]).T, np.product(shape))
    all_combos = list(product(*list(indices.T)))
    return np.reshape(np.array(all_combos), (-1, *shape))

def relu(arr0):
    arr = np.copy(arr0)
    arr[arr <= 0] = 0
    return arr


def normalize_hist(hist_dict0):
    """
    Given a histogram in dictionary form consisting
    of 'key' : count, generate a new histogram normalized
    by the count totals
    """

    hist_dict = hist_dict0.copy()

    all_vals = list(hist_dict.values())
    sum_vals = np.sum(all_vals)

    # modify in place
    hist_dict.update((k, v / sum_vals) for k, v in hist_dict.items())

    return hist_dict


def shannon_entropy(pi_set0):
    """
    Given a set of probabilities, compute the Shannon
    entropy, dropping any zeros
    """
    pi_set = np.array(pi_set0)
    pi_set_nonzero = np.copy(pi_set[pi_set > 0])

    hi = pi_set_nonzero.dot(np.log2(pi_set_nonzero))

    out = -np.sum(hi)

    return out

def find_dead(arr, axis=-1):
    """
    Given an array, count the number of axes where
    all samples evaluated to the same value

    Inputs:
    arra : np.array
        an array of shape (n_samples, n_features)

    Returns:
    where_dead : list
        The axes of the dead neurons
    """
    where_dead = list()
    for ax_ind in range(arr.shape[axis]):
        vals = arr[..., ax_ind]
        val_med = np.median(vals)
        if np.allclose(vals, val_med):
            where_dead.append(ax_ind)

    return where_dead

def plot_state(state):
    clear_output(wait=True)
    plt.imshow(state)
    plt.axis(False)
    plt.show()

def run_ca(ca, size=(100, 100), p_alive=0.5, iters=None):
    iters = range(iters) if iters else count()
    
    state = np.random.choice([0, 1], size=size, p=[1-p_alive, p_alive])
    m, n = state.shape
    state[m//2, n//2] = 1 if p_alive > 0 else 0 # ensure at least 1 living cell
    plot_state(state)
    for _ in iters:
        time.sleep(0.1)
        state = ca(state)
        plot_state(state) 
        
# adapted from: https://github.com/thearn/game-of-life
from numpy.fft import fft2, ifft2

def fft_convolve2d(x, y):
    """
    2D convolution, using FFT
    """
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m, n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, - int(m / 2) + 1, axis=0)
    cc = np.roll(cc, - int(n / 2) + 1, axis=1)
    return cc.round()

def n_sum_ca(born, survives):
    def ca(state):
        kernel = np.zeros_like(state)
        m, n = kernel.shape
        kernel[m//2-1 : m//2+2, n//2-1 : n//2+2] = np.pad([[0]], 1, constant_values=1)

        neighbours_alive = fft_convolve2d(state, kernel)

        new_state = np.zeros_like(neighbours_alive)
        new_state[np.where((state == 0) & np.isin(neighbours_alive, born))] = 1
        new_state[np.where((state == 1) & np.isin(neighbours_alive, survives))] = 1

        return new_state
    return ca