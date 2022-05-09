import numpy as np
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
import os
import obspy
import ast

from obspy.signal.tf_misfit import cwt
import scipy
from scipy import signal, stats
from obspy.core.utcdatetime import UTCDateTime

## Features
## Waveforms
## W0  -  

def compute_autocorr(x):

    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def create_Trace(time, amp, bandpass = [], detrend = True, differentiate=False):

    """
    Create obspy trace from time and amplitude arrays
    """
        
    tr = obspy.Trace()
    tr.data        = amp.copy()
    tr.stats.delta = abs( time[1] - time[0] )
    
    if differentiate:
        tr.differentiate()
            
    if bandpass:
        tr.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1], zerophase=True)

    if detrend:
        tr.detrend("polynomial", order=1)
    
    return tr

def integrate_signal(times, amp_):
        int_ = []
        for i_time, time_ in enumerate(times):
                int_.append( np.trapz(amp_[:i_time], x=times[:i_time]) )
        return np.array(int_)  

def setup_name_parameters(options):
        
    feature_name = {
        'W0': 'Ratio of the mean over the maximum of the envelope signal',
        'W1': 'Ratio of the median over the maximum of the envelope signal',
        'W2': 'Kurtosis of the raw signal (peakness of the signal)',
        'W3': 'Kurtosis of the envelope',
        'W4': 'Skewness of the raw signal',
        'W5': 'Skewness of the envelope',
        'W6': 'Number of peaks in the autocorrelation function',
        'W7': 'Energy in the first third part of the autocorrelation function',
        'W8': 'Energy in the remaining part of the autocorrelation function',
        'W9': 'W7/W8',
        'W10': 'Maximum of the envelope signal',
        
        'S0': 'Mean of the Fourier transform (FT)',
        'S1': 'Maximum of the FT',
        'S2': 'Frequency at the FT maximum',
        'S3': 'Frequency at the FT centroid',
        'S4': 'Frequency at the FT 1st quartile',
        'S5': 'Frequency at the FT 2nd quartile',
        'S6': 'Median of the normalized FT',
        'S7': 'Variance of the normalized FT',
        'S8': 'Number of Fourier transform peaks ($>0.75$ FT max.)',
        'S9': 'Mean of FT peaks (S8)',
        #'S10': 'spectral-centroid',
        'S10': 'Gyration radius',
        #'S12': 'spectral-centroid-width',
        
        'FT0': 'Kurtosis of the maximum of all Fourier transforms (FTs) as a function of time',
        'FT1': 'Kurtosis of the maximum of all FTs as a function of frequency',
        'FT2': 'Mean ratio between the maximum and the mean of all FTs',
        'FT3': 'Mean ratio between the maximum and the median of all FTs',
        'FT4': 'Number of peaks in the curve showing the temporal evolution of the FTs maximum',
        'FT5': 'Number of peaks in the curve showing the temporal evolution of the FTs mean',
        'FT6': 'Number of peaks in the curve showing the temporal evolution of the FTs median',
        'FT7': 'FT4/FT5',
        'FT8': 'FT4/FT6',
        'FT9': 'Number of peaks in the curve of the temporal evolution of the FTs central frequency',
        'FT10': 'Number of peaks in the curve of the temporal evolution of the FTs maximum frequency',
        'FT11': 'FT9/FT10',
        'FT12': 'Mean distance between the curves of the temporal evolution of the FTs maximum and mean frequency',
        'FT13': 'Mean distance between the curves of the temporal evolution of the FTs maximum and median frequency',
        'FT14': 'Mean distance between the 1st quartile and the median of all FTs as a function of time',
        'FT15': 'Mean distance between the 3rd quartile and the median of all FTs as a function of tim',
        'FT16': 'Mean distance between the 3rd quartile and the 1st quartile of all FTs as a function of time',
    }

    i = 11
    format_energy = 'Energy of the signal filtered in {f0}-{f1} Hz'
    for iband, band in enumerate(options['list_freq_bans']):
        loc_dict = {'f0': band[0], 'f1': band[1]}
        feature_name['W'+str(i)] = format_energy.format(**loc_dict)#'energy-freq-' +str(band[0])+ '-' + str(band[1])
        i += 1 

    format_kurtosis = 'Kurtosis of the signal filtered in {f0}-{f1} Hz'
    for iband, band in enumerate(options['list_freq_bans']):
        loc_dict = {'f0': band[0], 'f1': band[1]}
        feature_name['W'+str(i)] = format_kurtosis.format(**loc_dict)#'kurtosis-freq-' +str(band[0])+ '-' + str(band[1])
        i += 1 

    i = 11
    list_bounds_Nyf = np.array([0.5, 0.75, 1.])
    format_Nyf = 'Energy up to {frac}Nyf Hz'
    for i_, energy in enumerate(list_bounds_Nyf):
            i += i_
            loc_dict = {'frac': energy}
            feature_name['S'+str(i)] = format_Nyf.format(**loc_dict)#'energy-'+str(i_+1)+'-Nyf'
    
    return feature_name

def get_time_freq_spectrogram(dt, window, freq_min, freq_max):
    
    """
    Return frequency and time vectors corresponding to a given window
    """
    
    time = np.arange(0., window+dt, dt)
    amp  = time*0.
    f, spectro = compute_spectrogram(time, amp, freq_min, freq_max, w0=8, freq_time_factor=1)
    
    return time, f

def compute_features_as_spectrograms(signal_, type_data, freq_min, freq_max):

    """
    Return input data as a spectrogram, instead of standard waveform features
    """

    features = {}
    
    try:
        time = signal_[:,0]
        amp  = signal_[:,1]
    except:
        time = signal_.times().copy()
        amp  = signal_.data.copy()
            
    f, spectro = compute_spectrogram(time, amp, freq_min, freq_max, w0=8, freq_time_factor=1)
    
    return spectro

def compute_params_one_waveform(signal_, type_data, options):

    features = {}
    
    try:
        time = signal_[:,0]
        amp  = signal_[:,1]
    except:
        time = signal_.times().copy()
        amp  = signal_.data.copy()
    
    ## Auxiliaries
    enveloppe = abs(signal.hilbert(amp))
    autocorr = compute_autocorr(amp)
    peaks_autocorr, _  = signal.find_peaks(autocorr, height=np.mean(autocorr))
    autocorr_integrate = integrate_signal(time, autocorr)
    loc_threequarter   = 1 * len(time) // 3
    
    energy, kurt = [], []
    for iband, band in enumerate(options['list_freq_bans']):
        tr = create_Trace(time, amp, bandpass = band, detrend = False)
        energy.append( np.sum(np.trapz(tr.data, x=tr.times())) )
        kurt.append( stats.kurtosis(tr.data, fisher=False) )
    
    ## Features
    features['W0'] = np.mean(enveloppe) / np.max(enveloppe)
    features['W1'] = np.median(enveloppe) / np.max(enveloppe)
    features['W2'] = stats.kurtosis(amp, fisher=False)
    features['W3'] = stats.kurtosis(enveloppe, fisher=False)
    features['W4'] = stats.skew(amp)
    features['W5'] = stats.skew(enveloppe)
    features['W6'] = peaks_autocorr.size
    features['W7'] = np.sum(autocorr_integrate[:loc_threequarter])
    features['W8'] = np.sum(autocorr_integrate[loc_threequarter:])
    features['W9'] = features['W7'] / features['W8']
    features['W10'] = np.max(enveloppe)
    
    i = 11
    for iband, band in enumerate(options['list_freq_bans']):
       features['W'+str(i)] = energy[iband]
       i += 1 

    for iband, band in enumerate(options['list_freq_bans']):
       features['W'+str(i)] = kurt[iband]
       i += 1 
    
    return features

def compute_fft(time_, amp_):
        
    N = time_.size
    T = abs(time_[1]-time_[0])
    f = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_amp_ = scipy.fftpack.fft(amp_)
    fft_amp_ = abs(fft_amp_[:N//2])
    
    return f, fft_amp_

def get_energy_Nyf(f, fft_amp):

    fft_integrate = integrate_signal(f, fft_amp)
    Nyf = f[-1]
    list_bounds_Nyf  = np.array([0.5, 0.75, 1.]) * Nyf
    list_ibounds_Nyf = [ np.argmin(abs(f-f_)) for f_ in list_bounds_Nyf ]
    energy_Nyf      = [np.sum(fft_integrate[:ibound]) if ibound > 0 else 0. for ibound in list_ibounds_Nyf ]
    
    return energy_Nyf

def compute_params_one_spectrum(signal_, type_data, options):

    features = {}
    
    try:
            time = signal_[:,0]
            amp  = signal_[:,1]
    except:
            time = signal_.times().copy()
            amp  = signal_.data.copy()
            
    ## Auxiliaries
    f, fft_amp = compute_fft(time, amp)
    
    ## Remove contributions of first two frequencies (aliasing)
    f = f[2:]
    fft_amp = fft_amp[2:]
    
    fft_amp_normalized = fft_amp/time.size
    quantiles    = np.quantile(fft_amp, [0.25,0.5])
    peaks_fft, _ = signal.find_peaks(fft_amp, height=0.75*np.max(fft_amp))
    spectral_centroid = np.sum(f*fft_amp) / np.sum(fft_amp)
    
    centroid   = np.quantile(fft_amp, 0.5, axis=0)
    f_centroid = f[ np.argmin( abs(fft_amp-centroid) ) ]
    
    #fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False); axs[0].plot(time, amp); axs[1].plot(f, fft_amp); axs[1].set_ylim([0., 0.01]); plt.savefig('./figures/test/'+options['name_test'])
    
    ## Features
    features['S0'] = np.mean(fft_amp)
    features['S1'] = np.max(fft_amp)
    features['S2'] = f[np.argmax(fft_amp)]
    features['S3'] = f_centroid
    features['S4'] = f[np.argmin(abs(0.5*quantiles[0]-fft_amp))]
    features['S5'] = f[np.argmin(abs(0.5*quantiles[1]-fft_amp))]
    features['S6'] = np.median(fft_amp_normalized)
    features['S7'] = np.var(fft_amp_normalized)
    features['S8'] = peaks_fft.size
    features['S9'] = np.mean(peaks_fft)
    
    i = 10
    #features['S'+str(i)] = stats.moment(fft_amp, moment=2) / stats.moment(fft_amp, moment=1)
    #i += 1
    features['S'+str(i)] = np.sqrt(stats.moment(fft_amp, moment=3) / stats.moment(fft_amp, moment=2))
    
    #i += 1
    #features['S'+str(i)] = np.sqrt(stats.moment(fft_amp, moment=1)**2 - stats.moment(fft_amp, moment=2)**2)
    
    ## Get energy in different frequency bands
    energy_Nyf = get_energy_Nyf(f, fft_amp)
    i += 1
    for i_, energy in enumerate(energy_Nyf):
            i += i_
            features['S'+str(i)] = energy

    return features

def compute_spectrogram(time, amp, freq_min, freq_max, w0=8, freq_time_factor=1):
        
    """
    Return a spectrogram computed using morlet wavelets
    """
        
    npts = time.size
    dt   = abs(time[1] - time[0])
    spectro = cwt(amp, dt, w0=w0, fmin=freq_min, fmax=freq_max, nf=len(time)*freq_time_factor)
    f = np.logspace(np.log10(freq_min), np.log10(freq_max), spectro.shape[0])
    
    return f, np.real(spectro)
        
def compute_params_one_spectrogram(signal_, type_data, options):
        
    """
    Compute spectrogram-based features
    """
        
    features = {}
    
    try:
        time = signal_[:,0]
        amp  = signal_[:,1]
    except:
        time = signal_.times().copy()
        amp  = signal_.data.copy()
            
    #print('- compute spectro')
    ## Auxiliaries
    f, spectro = compute_spectrogram(time, amp, max(options['freq_min'], 1./time.max()), options['freq_max'], w0=8, freq_time_factor=1)
    
    #print('- find peaks max')
    list_time_max = np.max(spectro, axis=0)
    list_freq_max = np.max(spectro, axis=1)
    peaks_list_time_max, _ = signal.find_peaks(list_time_max, height=np.mean(list_time_max))
    
    #print('- find peaks median')
    list_time_median = np.median(spectro, axis=0)
    peaks_list_time_median, _ = signal.find_peaks(list_time_median, height=np.mean(list_time_median))
    
    #print('- compute peaks mean')
    list_time_mean = np.mean(spectro, axis=0)
    peaks_list_time_mean, _ = signal.find_peaks(list_time_mean, height=np.mean(list_time_mean))
    
    #print('- compute central freq')
    #F, T = np.meshgrid(f, t)
    list_time_maxfreq = np.array([f[iarg] for iarg in np.argmax(spectro, axis=0)])
    list_quantiles = np.quantile(spectro, 0.5, axis=0)
    list_time_centralfreq = np.array([f[ np.argmin( abs(quantile - spectro[:,iquantile]) ) ] for iquantile, quantile in enumerate(list_quantiles)])
    
    #print('- compute central max freq')
    peaks_list_time_maxfreq, _ = signal.find_peaks(list_time_maxfreq, height=np.mean(list_time_maxfreq))
    peaks_list_time_centralfreq, _ = signal.find_peaks(list_time_centralfreq, height=np.mean(list_time_centralfreq))
    
    #print('- get attributes')
    ## Features
    features['FT0'] = stats.kurtosis(list_freq_max, fisher=False)
    features['FT1'] = stats.kurtosis(list_time_max, fisher=False)
    features['FT2'] = spectro.max() / spectro.mean()
    features['FT3'] = spectro.max() / np.median(spectro)
    features['FT4'] = peaks_list_time_max.size
    features['FT5'] = peaks_list_time_mean.size
    features['FT6'] = peaks_list_time_median.size
    features['FT7'] = features['FT4']/features['FT5']  if features['FT5'] > 0 else np.nan
    features['FT8'] = features['FT4']/features['FT6'] if features['FT6'] > 0 else np.nan
    features['FT9'] = peaks_list_time_centralfreq.size
    features['FT10'] = peaks_list_time_maxfreq.size
    features['FT11'] = features['FT9'] / features['FT10']  if features['FT10'] > 0 else np.nan
    features['FT12'] = abs(list_time_max-list_time_mean).mean()
    features['FT13'] = abs(list_time_max-list_time_median).mean()
    features['FT14'] = abs(np.percentile(spectro, 25, axis=0)-list_time_median).mean()
    features['FT15'] = abs(np.percentile(spectro, 75, axis=0)-list_time_median).mean()
    features['FT16'] = abs(np.percentile(spectro, 75, axis=0)-np.percentile(spectro, 25, axis=0)).mean()
    
    if type_data == 'arrival' and False:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True); axs[0].pcolormesh(time, f, spectro); axs[1].plot(time, amp); axs[1].scatter(time[peaks_list_time_max], amp[peaks_list_time_max]); axs[0].scatter(time, f[np.argmax(spectro, axis=0)], marker='x', c='black'); plt.show()
        bp()
        
    return features
        
