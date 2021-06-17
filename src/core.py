import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter1d
import essentia.standard as estd
import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

from src.utils import get_timestamp, interpolate_below_length
from src.visualisation import plot_all_sequences, double_timeseries, plot_subsequence
from src.iam import unpack_saraga_metadata
from src.io import write_all_sequence_audio, write_json, load_json, load_yaml, load_tonic
from src.sequence import contains_silence, too_stable, start_with_silence, min_gap
from src.matrix_profile import get_matrix_profile
from src.motif import get_motif_clusters, get_exclusion_mask
from src.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents

from pathlib import Path



def find_motifs(conf, exclusion_conf, plot_conf):
    # Params
    ########
    audio_path = conf['audio_path']
    tonic = plot_conf['tonic']

    sampling_rate = conf['sampling_rate'] # defined on load in next cell
    frameSize = conf['frameSize'] # For Melodia pitch extraction
    hopSize = conf['hopSize'] # For Melodia pitch extraction
    gap_interp = conf['gap_interp'] # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]
    smooth = conf['smooth'] # sigma for gaussian smoothing of pitch track [set to None to skip]

    m_secs = conf['m_secs']
    if conf['cache']:
        cache_dir = os.path.join(Path(audio_path).parent.absolute(), '.matrix_profile', f'gap_interp={gap_interp}__smooth={smooth}','')

    exclusion_funcs = [x['func'] for x in exclusion_conf['exclusion_funcs']]

    # For Carnatic music, we automatically define yticks_dict
    plot_kwargs = plot_conf
    output_audio = conf['output_audio']
    output_plots = conf['output_plots']

    # Maximum number of unique motif groups to return
    top_n = conf['top_n']

    # Maximum number of occurrences to return in each motif group
    n_occ = conf['n_occ']

    # Minimum number of occurrences to return in each motif group
    min_occ = conf['min_occ']

    thresh = conf['thresh'] # patterns with parent distances above this threshold are not considered

    out_dir = conf['out_dir']


    ## Pitch Extraction
    ###################
    print('#######')
    print('Extracting Pitch...')
    print('#######')
    print('')
    # load raw audio for display later
    audio_loaded, sr = librosa.load(audio_path, sr=sampling_rate)

    # Run spleeter on track to remove the background
    separator = Separator('spleeter:2stems')
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(audio_path, sample_rate=sampling_rate)
    prediction = separator.separate(waveform=waveform)
    clean_vocal = prediction['vocals']

    # Prepare audio for pitch extraction
    audio_mono = clean_vocal.sum(axis=1) / 2
    audio_mono_eqloud = estd.EqualLoudness(sampleRate=sampling_rate)(audio_mono)

    # Extract pitch using Melodia algorithm from Essentia
    pitch_extractor = estd.PredominantPitchMelodia(frameSize=frameSize, hopSize=hopSize)
    raw_pitch, _ = pitch_extractor(audio_mono_eqloud)
    raw_pitch_ = np.append(raw_pitch, 0.0)
    time = np.linspace(0.0, len(audio_mono_eqloud) / sampling_rate, len(raw_pitch))

    timestep = time[4]-time[3] # resolution of time track

    # Gap interpolation
    if gap_interp:
        print(f'Interpolating gaps of {gap_interp} or less')
        raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))
        
    # Gaussian smoothing
    if smooth:
        print(f'Gaussian smoothing with sigma={smooth}')
        pitch = gaussian_filter1d(raw_pitch, smooth)
    else:
        pitch = raw_pitch[:]


    ## Matrix Profile
    #################
    print('#######')
    print('Finding motifs')
    print('#######')
    print('')
    # Convert to elements
    if isinstance(m_secs, list):
        m_el = [int(x/timestep) for x in m_secs]
    else:
        m_el = int(m_secs/timestep)


    matrix_profile, matrix_profile_length = get_matrix_profile(pitch, m_el, path=cache_dir)
    # Can take some time depending on exclusion_funcs
    if cache_dir:
        exclusion_cache_path = os.path.join(cache_dir + f'.exclusion_mask_final3/{str(m_el)}/')
    else: 
        exclusion_cache_path = None
    
    print(f'Applying exclusion funcs: {[x.__name__ for x in exclusion_funcs]}')
    exclusion_mask = get_exclusion_mask(pitch, matrix_profile_length, exclusion_funcs, path=exclusion_cache_path)
    print(f'{round(sum(exclusion_mask)*100/len(exclusion_mask), 2)}% subsequences excluded')


    motifs, distances, motif_len = get_motif_clusters(matrix_profile, pitch, matrix_profile_length, top_n, n_occ, exclusion_mask, thresh=thresh, min_occ=min_occ)

    ## Output
    #########
    # If the directory does not exist, it will be created
    print(f'Output directory: {out_dir}')

    if output_plots:
        print('Writing plots')
        plot_all_sequences(raw_pitch, time, motif_len, motifs, out_dir, distances=distances, clear_dir=True, plot_kwargs=plot_kwargs)
    if output_audio:
        print('Writing audio')
        write_all_sequence_audio(audio_path, motifs, motif_len, timestep, out_dir)

    parameters = {
        'top_n': top_n,
        'n_occ':n_occ,
        'min_occ':min_occ,
        'thresh':thresh,
        'sampling_rate': sampling_rate,
        'frameSize': frameSize,
        'hopSize': hopSize,
        'gap_interp': gap_interp,
        'smooth': smooth,
        'audio_path': audio_path
    }
    print('Writing metadata')
    write_json(parameters, os.path.join(out_dir, 'parameters.json'))