import numpy as np

def contains_silence(seq, thresh=0.05):
    """If more than <thresh> of <seq> is 0, return True"""
    return sum(seq==0)/len(seq) > thresh


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def too_stable(seq, dev_thresh=5, perc_thresh=0.63, window=200):
    """If a sufficient proportion of <seq> is "stable" return True"""
    if window > len(seq):
        window=len(seq)
    mu_ = seq[:window-1]
    mu = np.concatenate([mu_, moving_average(seq, window)])

    dev_arr = abs(mu-seq)
    dev_seq = dev_arr[np.where(~np.isnan(dev_arr))]
    bel_thresh = dev_seq < dev_thresh

    perc = np.count_nonzero(bel_thresh)/len(dev_seq)

    if perc >= perc_thresh:
        is_stable = 1
    else:
        is_stable = 0
    
    return is_stable


def start_with_silence(seq):
    return any([seq[0] == 0, all(seq[:100]==0)])


def min_gap(seq, length=86):
    seq2 = np.trim_zeros(seq)
    m1 = np.r_[False, seq2==0, False]
    idx = np.flatnonzero(m1[:-1] != m1[1:])
    if len(idx) > 0:
        out = (idx[1::2]-idx[::2])
        if any(o >= length for o in out):
            return True
    return False