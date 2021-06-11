import os
from time import time as timer

import numpy as np
from sklearn.cluster import DBSCAN
import stumpy
import tqdm

from src.utils import get_timestamp
from src.io import load_if_exists, create_if_not_exists

def get_occurrences(ss, pitch, l, n_occ, mask):
    """
    For subsequence pitch[<ss>:<ss>+<l>], compute distance between itself and 
    every other subsequence in pitch. Return top <N> unique subsequences.
    """
    stmass = stumpy.core.mass(pitch[ss:ss+int(l)], pitch, normalize=False)
    
    mask = mask[:-int(l)]

    stmass[mask] = np.Inf

    n_returned_occ = 0
    occurences = []
    distances = []
    while n_returned_occ < n_occ:
        ix = stmass.argmin()
        occurences.append(ix)
        distances.append(stmass[ix]/l)
        stmass[max(0,int(ix-l)):min(len(stmass),int(ix+l))] = np.Inf

        n_returned_occ += 1

    return occurences, distances


def get_motif_clusters(mp, pitch, lengths, top_n, n_occ, exclusion_mask, thresh=None, min_occ=None):
        
    if not min_occ:
        min_occ = 2

    coparr = np.array(mp)  

    mask = np.where(exclusion_mask[:len(mp)])[0]
    
    # Exclude sequences we dont care about
    coparr[mask] = np.Inf
    
    n_returned_groups = 0
    subseqs = []
    distances = []
    subseq_lengths = []

    while n_returned_groups < top_n:      
        # Get most salient pattern
        ix = coparr.argmin()

        # Get length
        l = lengths[ix]

        if coparr[ix] == np.Inf:
            # No more patterns to be found
            break

        # Identify other occurences of pattern 
        pats, dists = get_occurrences(ix, pitch, l, n_occ, mask)
        
        
        # Prune those with low importance
        if thresh:
            prune_lim = get_prune_lim(dists, thresh)

            pats = pats[:prune_lim]
            dists = dists[:prune_lim]

        if len(pats) < min_occ:
            # Remove pattern and continue search
            coparr[max(0, int(pats[0]-l)):min(int(pats[0]+l),len(coparr))] = np.Inf
            continue

        # Store pattern start element and importance (euclidean distance to parent)
        subseqs.append(pats)
        distances.append(dists)
        subseq_lengths.append(l)
        
        # Set to infinite to ensure same region is not returned elsewhere
        for p in pats:
            coparr[max(0,int(p-l)):min(int(p+l),len(coparr))] = np.Inf
        
        n_returned_groups += 1

        n_occurrences = len(pats)
        print(f'{n_occurrences} occurrences in motif group {n_returned_groups-1} (length, {l})')
    
    return subseqs, distances, subseq_lengths


def get_prune_lim(dists, thresh):
    for i,d in enumerate(dists):
        if d > thresh:
            return i


def get_exclusion_mask(pitch, lengths, exclusion_funcs, path=None):
    # Can take some time depending on exclusion_funcs
    if path:
        exclusion_cache_path = os.path.join(path + f'funcs={str([x.__name__ for x in exclusion_funcs])}.tsv')
        to_return = load_if_exists(exclusion_cache_path, dtype=int)
        if not to_return is None:
            print(f'loaded from cache {exclusion_cache_path}')
            return to_return

    to_return = []
    for i in tqdm.tqdm(list(range(len(pitch)))):
        
        to_return.append(any(er(pitch[i:int(i+lengths[i])]) for er in exclusion_funcs))

    exclusion_mask = np.array(to_return)
    if path:
        create_if_not_exists(exclusion_cache_path)
        print('caching exclusion mask')
        np.savetxt(exclusion_cache_path, exclusion_mask, fmt='%d')
    return exclusion_mask
