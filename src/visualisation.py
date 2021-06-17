import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import numpy.ma as ma
import scipy.signal
import shutil

from src.pitch import (pitch_seq_to_cents, pitch_to_cents)
from src.utils import get_timestamp, myround
from src.io import create_if_not_exists, write_json


style.use('seaborn-dark-palette')


iam_kwargs = {
    'window': scipy.signal.get_window('hann', int(0.0464*44100)),
    'NFFT': int(0.0464*44100), # window length x sample rate
}

def spectrogram(audio, sampleRate=44100, ylim=None, kwargs=iam_kwargs):
    """
    Plot spectrogram of input audio single

    :param audio: [vector_real] audio signal
    :type audio: iterable
    :param sampleRate: frame size for Fourier Transform
    :type sampleRate: int
    :param ylim: [min,max] frequency limits, default [20, 20000]
    :type ylim: iterable
    :param kwargs: Dict of keyword arguments for matplotlib.pyplot.specgram, default {}
    :type kwargs: dict

    :return:  spectrogram/waveform plot
    :rtype: matplotlib.pyplot
    """
    plt.title('Spectrogram')
    plt.specgram(audio, Fs=sampleRate, **kwargs)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.ylim(ylim)

    return plt


def double_timeseries(x, y1, y2, y1label='', y2label='', xlabel='', xfreq=5, yfreq=50, linewidth=1, ylim1=None, ylim2=None):
    """
    Plot and two time series (with the same x) on the same axis

    :param x: x data (e.g time)
    :type array: numpy.array
    :param y1: y data of top plot
    :type y1: numpy.array
    :param y2: y data of bottom plot
    :type y2: numpy.array
    :param y1label: y label of top plot (y1 data)
    :type y1label: str
    :param y2label: y label of bottom plot (y2 data)
    :type y2label: str
    :param xfreq: y tick frequency
    :type xfreq: int
    :param yfreq: y tick frequency
    :type yfreq: int
    
    :return: tuple of plot objects, (fig, np.array([ax1, ax2]))
    :rtype: (matplotlib.figure.Figure, numpy.array([matplotlib.axes._subplots.AxesSubplot, ...]))
    """
    timestep = x[1]-x[0]

    l = len(x)
    samp_len = int(l*timestep)

    fig, axs = plt.subplots(2, sharex=True)
    fig.set_size_inches(170*samp_len/540, 10.5)
    axs[0].plot(x, y1, linewidth=linewidth)
    axs[0].grid()
    axs[0].set_ylabel(y1label)
    axs[0].set_xticks(np.arange(min(x), max(x)+1, xfreq))
    axs[0].set_yticks(np.arange(min(y1), max(y1)+1, yfreq))
    if ylim1:
        axs[0].set_ylim(ylim1)

    axs[1].plot(x[:len(y2)], y2, color='green', linewidth=linewidth)
    axs[1].grid()
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(y2label)
    axs[1].set_xticks(np.arange(min(x[:len(y2)]), max(x[:len(y2)])+1, xfreq))
    if ylim2:
        axs[1].set_ylim(ylim2)

    return fig, axs


def plot_annotate_save(x, y, matrix_profile, seqs, m, path, y1label='', y2label='', xlabel='Time', sample=False):
    """
    Plot and annotate time series and matrix_profile with 
    subsequences of length <m>, save png to <path>

    :param x: x data (e.g time)
    :type array: numpy.array
    :param y: y data (e.g. pitch) 
    :type y: numpy.array
    :param matrix_profile: Matrix profile of data in x,y
    :type matrix_profile: numpy.array
    :param seqs: List of subsequence start points to annotate on plot
    :type seqs: iterable
    :param m: Fixed length of subsequences in <seq>
    :type m: int
    :param path: Path to save png plot to
    :type path: str
    :param y1label: y label for time series
    :type y1label: str
    :param y2label: y label for matrix profile (distance measure)
    :type y2label: str
    :param xlabel: x label for time series, default 'Time'
    :type xlabel: str
    :param sample: If True, only show parts of the plot that contain sequences in <seqs>
    :type sample: bool
    """
    timestep = x[1]-x[0]

    if sample:
        min_s = min(seqs) - 2*m # pad plot with 2 sequence lengths
        max_s = max(seqs) + m + 2*m # pad plot with 2 sequence legnths
    
        x = x[min_s:max_s]
        y = y[min_s:max_s]
        matrix_profile = matrix_profile[min_s:max_s]
        
        seqs = [s-min_s for s in seqs]

    fig, axs = double_timeseries(x, y, timestep, matrix_profile, y1label, y2label, xlabel)
    axs = annotate_plot(axs, seqs, m, timestep, linewidth=2)
    plt.savefig(path)
    plt.close('all')


def annotate_plot(axs, seqs, m, timestep, linewidth=2):
    """
    Annotate time series and matrix profile with sequences in <seqs>

    :param axs: list of two subplots, time series and matrix_profile
    :type axs: [matplotlib.axes._subplots.AxesSubplot, matplotlib.axes._subplots.AxesSubplot]
    :param seqs: iterable of subsequence start points to annotate
    :type seqs: numpy.array
    :param m: Fixed length of subsequences
    :type m: int
    :param linewidth: linewidth of shaded area of plot, default 2
    :type linewidth: float
    
    :return: list of two subplots, time series and matrix_profile, annotated
    :rtype: [matplotlib.axes._subplots.AxesSubplot, matplotlib.axes._subplots.AxesSubplot]
    """
    x_d = axs[0].lines[0].get_xdata()
    y_d = axs[0].lines[0].get_ydata()

    for c in seqs:
        x = x_d[c:c+m]
        y = y_d[c:c+m]
        axs[0].plot(x, y, linewidth=linewidth, color='burlywood')
        axs[1].axvline(x=x_d[c], linestyle="dashed", color='red')

    max_y = axs[0].get_ylim()[1]

    for c in seqs:
        rect = Rectangle((x_d[c], 0), m*timestep, max_y, facecolor='lightgrey')
        axs[0].add_patch(rect)

    return axs


def plot_pitch(
    pitch, times, s_len=None, mask=None, yticks_dict=None, cents=False, 
    tonic=None, emphasize=[], figsize=None, title=None, xlabel=None, ylabel=None, 
    grid=True, ylim=None, xlim=None):
    """
    Plot graph of pitch over time

    :param pitch: Array of pitch values in Hz
    :type pitch: np.array
    :param times: Array of time values in seconds
    :type times: np.array
    :param s_len: If not None, take first <s_len> elements of <pitch> and <time> to plot
    :type s_len:  int
    :param mask: Array of bools indicating elements in <pitch> and <time> NOT to be plotted
    :type mask: np.array
    :param yticks_dict: Dict of {frequency name: frequency value (Hz)}
        ...if not None, yticks will be replaced in the relevant places with these names
    :type yticks_dict: dict(str, float)
    :param cents: Whether or not to convert frequency to cents above <tonic> 
    :type cents: bool
    :param tonic: Tonic to make cent conversion in reference to - only relevant if <cents> is True.
    :type tonic: float
    :param emphasize: list of keys in yticks_dict to emphasize on plot (horizontal red line)
    :type emphasize: list(str)
    :param figsize: Tuple of figure size values 
    :type figsize: tuple
    :param title: Title of figure, default None
    :type title: str
    :param xlabel: x axis label, default Time (s)
    :type xlabel: str
    :param ylabel: y axis label
        defaults to 'Cents Above Tonic of <tonic>Hz' if <cents>==True else 'Pitch (Hz)'
    :type ylabel: str
    :param grid: Whether to plot grid
    :type grid: bool
    :param ylim: Tuple of y limits, defaults to max and min in <pitch>
    :type ylim: bool
    :param xlim: Tuple of x limits, defaults to max and min in <time>
    :type xlim: bool

    :returns: Matplotlib objects for desired plot
    :rtype: (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)
    """
    if cents:
        assert tonic, \
            "Cannot convert pitch to cents without reference <tonic>"
        p1 = pitch_seq_to_cents(pitch, tonic)
    else:
        p1 = pitch

    if mask is None:
        # If no masking required, create clear mask
        mask = np.full((len(pitch),), False)
    
    if s_len:
        assert s_len <= len(pitch), \
            "Sample length is longer than length of pitch input"
    else:
        s_len = len(pitch)
        
    if figsize:
        assert isinstance(figsize, (tuple,list)), \
            "<figsize> should be a tuple of (width, height)"
        assert len(figsize) == 2, \
            "<figsize> should be a tuple of (width, height)"
    else:
        figsize = (170*s_len/186047, 10.5)

    if not xlabel:
        xlabel = 'Time (s)'
    if not ylabel:
        ylabel = f'Cents Above Tonic of {round(tonic)}Hz' \
                    if cents else 'Pitch (Hz)'

    pitch_masked = np.ma.masked_where(mask, p1)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid()

    if xlim:
        xmin, xmax = xlim
    else:
        xmin = myround(min(times[:s_len]), 5)
        xmax = max(times[:s_len])
        
    if ylim:
        ymin, ymax = ylim
    else:
        sample = pitch_masked.data[:s_len]
        if not set(sample) == {None}:
            ymin_ = min([x for x in sample if x is not None])
            ymin = myround(ymin_, 50)
            ymax = max([x for x in sample if x is not None])
        else:
            ymin=0
            ymax=1000
    
    for s in emphasize:
        assert yticks_dict, \
            "Empasize is for highlighting certain ticks in <yticks_dict>"
        if s in yticks_dict:
            if cents:
                p_ = pitch_to_cents(yticks_dict[s], tonic)
            else:
                p_ = yticks_dict[s]
            ax.axhline(p_, color='#db1f1f', linestyle='--', linewidth=1)

    times_samp = times[:s_len]
    pitch_masked_samp = pitch_masked[:s_len]

    times_samp = times_samp[:min(len(times_samp), len(pitch_masked_samp))]
    pitch_masked_samp = times_samp[:min(len(times_samp), len(pitch_masked_samp))]
    plt.plot(times_samp, pitch_masked_samp, linewidth=0.7)

    if yticks_dict:
        tick_names = list(yticks_dict.keys())
        tick_loc = [pitch_to_cents(p, tonic) if cents else p \
                    for p in yticks_dict.values()]
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(tick_names)
    
    ax.set_xticks(np.arange(xmin, xmax+1, 1))

    plt.xticks(fontsize=8.5)
    ax.set_facecolor('#f2f2f2')

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((xmin, xmax))

    if title:
        plt.title(title)

    return fig, ax


def plot_subsequence(sp, l, pitch, times, timestep, path=None, plot_kwargs={}):
    
    this_pitch = pitch[int(max(sp-l,0)):int(sp+2*l)]
    this_times = times[int(max(sp-l,0)):int(sp+2*l)]
    this_mask = this_pitch==0
    
    fig, ax = plot_pitch(
        this_pitch, this_times, mask=this_mask,
        xlim=(min(this_times), max(this_times)), **plot_kwargs)
    
    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l,sp)):int(l+min(l,sp))]
    y = y_d[int(min(l,sp)):int(l+min(l,sp))]
    
    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    rect = Rectangle((x_d[int(min(l,sp))], min_y), l*timestep, max_y-min_y, facecolor='lightgrey')
    ax.add_patch(rect)
    #import ipdb; ipdb.set_trace()
    ax.plot(x, y, linewidth=0.7, color='darkorange')
    ax.axvline(x=x_d[int(min(l,sp))], linestyle="dashed", color='black', linewidth=0.8)

    if path:
        plt.savefig(path, dpi=90)
        plt.close('all')
    else:
        return plt


def plot_subsequence_paper(sp, l, pitch, times, timestep, path=None, plot_kwargs={}):
    
    this_pitch = pitch[int(sp):int(sp+l)]
    this_times = times[int(sp):int(sp+l)]
    this_mask = this_pitch==0
    
    fig, ax = plot_pitch(
        this_pitch, this_times, mask=this_mask,
        xlim=(min(this_times), max(this_times)), **plot_kwargs)
    
    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l,sp)):int(l+min(l,sp))]
    y = y_d[int(min(l,sp)):int(l+min(l,sp))]
    
    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    #rect = Rectangle((x_d[int(min(l,sp))], min_y), l*timestep, max_y-min_y, facecolor='lightgrey')
    #ax.add_patch(rect)
    #import ipdb; ipdb.set_trace()
    #ax.plot(x, y, linewidth=0.7, color='darkorange')
    #ax.axvline(x=x_d[int(min(l,sp))], linestyle="dashed", color='black', linewidth=0.8)

    if path:
        plt.savefig(path, dpi=90)
        plt.close('all')
    else:
        return plt        


def plot_all_sequences(pitch, times, lengths, seq_list, direc, distances=None, clear_dir=False, svara_annot=None, plot_kwargs={}):
    timestep = times[1] - times[0]
    if clear_dir:
        try:
            shutil.rmtree(direc)
        except:
            pass
    for i, seqs in enumerate(seq_list):
        if isinstance(lengths, (np.ndarray, list)):
            l = lengths[i]
        else:
            l = lengths
        for si, s in enumerate(seqs):
            t_sec = s*timestep
            str_pos = get_timestamp(t_sec)
            sp = int(s)
            plot_path = os.path.join(direc, f'motif_{i}/{si}_time={str_pos}.png')
            create_if_not_exists(plot_path)
            plot_subsequence(
                sp, l, pitch, times, timestep, path=plot_path, plot_kwargs=plot_kwargs
            )
    if distances:
        imps = {i:dict(enumerate(imp)) for i,imp in enumerate(distances)}
        imp_path = os.path.join(direc, f'distances.json')
        create_if_not_exists(imp_path)
        write_json(imps, imp_path)


