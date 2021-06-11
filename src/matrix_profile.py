import os
from time import time as timer

import numpy as np
import stumpy

from src.io import load_if_exists, create_if_not_exists

def get_matrix_profile(pitch, lengths, path=None):
    if path:
        mp_path = os.path.join(path, f'matrix_profile_{str(lengths)}.csv')
        mpl_path = os.path.join(path, f'matrix_profile_lengths_{str(lengths)}.csv')
        
        mp = load_if_exists(mp_path, dtype=float)
        mpl = load_if_exists(mpl_path, dtype=float)

        if all([not mp is None, not mpl is None]):
            print('loaded from cache')
            return mp, mpl

    if not isinstance(lengths, list):
        lengths = [lengths]

    all_mp = compute_matrix_profiles(pitch, lengths)
    mp, mpi, mpl = combine_mp(all_mp, lengths, len(pitch))

    if path:
        create_if_not_exists(mp_path)
        create_if_not_exists(mpl_path)
        print(f'caching at {path}')
        np.savetxt(mp_path, mp, fmt='%f')
        np.savetxt(mpl_path, mpl, fmt='%d')
    return mp, mpl
    

def combine_mp(mp_arr, mp_len, n):
    MP = []
    MPI = []
    for i in range(len(mp_arr)):
        mp = mp_arr[i][:,0]/mp_len[i]
        l = len(mp)
        d = n - l
        space = np.zeros(d)
        space[:] = np.inf
        MP.append(np.concatenate([mp,space]))

        mpi = mp_arr[i][:,1]
        MPI.append(np.concatenate([mpi,space]))

    MP = np.array(MP)
    MPI = np.array(MPI)

    argmins = [MP[:,i].argmin() for i in range(n)]

    MP = [MP[:,i][argmins[i]] for i in range(n)]
    MPI = [MPI[:,i][argmins[i]] for i in range(n)]
    MPL = [mp_len[argmins[i]] for i in range(n)]

    return np.array(MP), np.array(MPI), np.array(MPL)


def compute_matrix_profiles(pitch, lengths, path=None):
    all_mp = []
    for m in lengths:
        print(f'Computing MP for subsequence length {m}')
        t = timer()
        mp = stumpy.stump(pitch, m=m, normalize=False)
        all_mp.append(mp)
        t2 = timer()
        print(f'    time={round((t2-t)/60,2)} minutes')
    return all_mp