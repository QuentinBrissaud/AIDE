import numpy as np
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
import os
import obspy
import ast
import sys

from scipy import signal, interpolate
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.tf_misfit import cwt, plot_tfr, plot_tf_gofs
from sklearn.metrics import roc_curve

import compute_params_waveform

from multiprocessing import get_context
from functools import partial

## Remove obspy warnings
import warnings
warnings.simplefilter("ignore", Warning)

def get_window(sampling, window):

    """
    Get the right time window size based on input sampling
    """

    list_samplings = np.array([max_sampling for max_sampling in window if not max_sampling == -1])
    loc_sampling = np.argmin(abs(list_samplings - sampling))
    while list_samplings[loc_sampling] < sampling:
        loc_sampling += 1
        
    return window[list_samplings[loc_sampling]]

def extract_arrival_window_parameters(times, waveform, params, window, t_duration, 
                                      min_overlap_label, flag_testing, default_size_overlap=10):

    """
    Compute time window parameters
    """
    
    ## If testing data with no arrival, we use default parameters
    size_subset = np.argmin( abs((times-times[0]) - window) ) + 1
    size_overlap  = default_size_overlap
    iarrival_time = -1
    if not flag_testing:
        t_arrival = params['arrival-time']
        iarrival_time = np.argmin( abs(waveform['time_s'].values[:] - t_arrival) )
         
        #t_duration   = params['diff-t'] + shift_detection_after_max
        ## Maximum shift in arrival time introduced for better training
        size_overlap = np.argmin( abs((times-times[0]) - t_duration*(1.-min_overlap_label)) ) + 1
        
    return size_subset, iarrival_time, size_overlap

def downsample_trace(tr, standard_sampling):
    
    """
    Downsample high sampling rate data to make the dataset uniform in terms of sampling rate
    """

    if 1./tr.stats.sampling_rate < standard_sampling-0.1:
    
        ## Sampling rate ratio
        factor = int(np.round(standard_sampling/np.round(1./tr.stats.sampling_rate)))
        
        ## The Fourier method used for decimation only work up to factor 16, so we do the decimation in two steps
        ## Also, if sampling factor is over 15, it means that the sampling is 1s, i.e., factor 30.
        if factor > 16:
            factor_ = 15
            tr.decimate(factor=factor_, strict_length=False, no_filter=True)
            factor_ = 2
            tr.decimate(factor=factor_, strict_length=False, no_filter=True)
        elif factor > 1:
            tr.decimate(factor=factor, strict_length=False, no_filter=True)

def pre_process_waveform(times, vTEC, i0, iend, window, detrend=True, bandpass=[], standard_sampling=30.):

    """
    Create and process trace from TEC waveform data
    """
    
    ## HERE MODIFY & CHECK BANDPASSING AFTER CUTTING END
    tr = compute_params_waveform.create_Trace(times[:iend+1], vTEC[:iend+1], detrend=detrend, bandpass=bandpass, differentiate=True)
    #tr = compute_params_waveform.create_Trace(times, vTEC, detrend=detrend, bandpass=bandpass, differentiate=True)
    t0, tend = tr.times()[i0], tr.times()[iend]
    
    #plt.plot(tr.times()+times[0], tr.data); plt.axvline(times[i0]); plt.axvline(times[iend])
    ## Downsample traces if sampling higher than standard_sampling
    downsample_trace(tr, standard_sampling)
    
    i0   = np.argmin(abs(tr.times()-t0))
    iend = np.argmin(abs(tr.times()-tend))
    #print(i0, iend)
    iend_new = np.arange(0., window+standard_sampling, standard_sampling).size
    iend = i0 + iend_new - 1
    #print(iend_new, iend, tr.times().size)
    ## If iend is outside the time range, we shift the window slightly in time
    if iend > tr.times().size-1:
        shift = iend-tr.times().size+1
        i0 -= shift
        i0 = max(i0, 0)
        iend -= shift
    
    
    tr.trim(tr.stats.starttime + tr.times()[i0], tr.stats.starttime + tr.times()[iend])
    
    return tr, i0, iend

def extract_features_based_on_input_type(tr, type_data, type, options):

    """
    Extract the right feature depending on the requested feature type: features or spectrogram
    """
    
    ## Use discrete and predifined features
    features = {}
    if type == 'features':
        features.update( compute_params_waveform.compute_params_one_waveform(tr, type_data, options) )
        features.update( compute_params_waveform.compute_params_one_spectrum(tr, type_data, options) )
        features.update( compute_params_waveform.compute_params_one_spectrogram(tr, type_data, options) )
    ## Use raw spectrogram
    elif type == 'spectrogram':
        spectro = compute_params_waveform.compute_features_as_spectrograms(\
                    tr, type_data, options['freq_min'], options['freq_max'])
        features['spectro'] = spectro
    else:
        sys.exit('Input data type not recognized')

    features = pd.DataFrame([features])

    return features

def get_right_var(tr, tr_artificial, snr, min_factor=0.01, max_factor=2., nb_tests=500):

    """
    Return modified trace where SNR has been updated
    """

    # test_module.get_right_var(tr, tr_artificial, snr, min_factor=0.01, max_factor=2., nb_tests=500)

    factors = np.linspace(min_factor, max_factor, nb_tests)
    mat_factors = np.repeat(factors[:, None], tr.data.size, axis=1)
    mat = np.repeat(tr.data[None,:], nb_tests, axis=0)
    mat *= mat_factors
    mat_artificial = np.repeat(tr_artificial.data[None,:], nb_tests, axis=0)
    mat += mat_artificial
    test_snr = (np.var(mat, axis=1) - np.var(mat_artificial, axis=1)) / np.var(mat_artificial, axis=1)
    iSNR = np.argmin(abs(test_snr - snr))
    
    tr_output = tr.copy()
    tr_output.data = tr.data*factors[iSNR] + tr_artificial.data
    
    return tr_output

## TODO: remove iarrival_time
def extract_one_feature(times, waveform, params, tduration, i0, iend, window, event, satellite, station, type_data, options, 
                        i0_noise_toadd=-1, iend_noise_toadd=-1, snr=-1, type='features', add_gaussian_noise=False, detrend=True):
    
    """
    Construct features for one time window
    """
    
    #i0_, iend_ = i0, iend
    #tr_total, _, _ = pre_process_waveform(times, waveform['vTEC'].values, 0, waveform['vTEC'].values.size-1, 100000., detrend=False, bandpass=[options['freq_min'], options['freq_max']])
    tr, i0_, iend_ = pre_process_waveform(times, waveform['vTEC'].values, i0, iend, window, detrend=detrend, bandpass=[options['freq_min'], options['freq_max']])
    #tr, i0_, iend_ = pre_process_waveform(times[:iend+1], waveform['vTEC'].values[:iend+1], i0, iend, window, detrend=False, bandpass=[options['freq_min'], options['freq_max']])
    
    ## Add random white noise to input waveform
    if add_gaussian_noise:
        p1 = np.var(tr.data)
        tr.data += np.sqrt(p1/snr)*np.random.randn(tr.data.size)
        #tr=tr_.copy(); tr.data += np.sqrt(p1/1.)*np.random.randn(tr.data.size)
        
        #tr_artificial = tr.copy(); tr_artificial.data = np.sqrt(nn)*np.random.randn(tr.data.size); tr__ = get_right_var(tr_, tr_artificial, snr, min_factor=0.01, max_factor=2., nb_tests=1000)
    
    #if not type_data == 'noise':
    #bp()
    #tr_, i0_, iend_ = pre_process_waveform(times, waveform['vTEC'].values, i0, iend, window, detrend=False, bandpass=[options['freq_min'], options['freq_max']]); tr=tr_.copy(); tr.data += np.sqrt(p1/snr)*np.random.randn(tr.data.size)
    #plt.plot(tr_.times(), tr_.data, label='orig'); plt.plot(tr.times(), tr.data, label='gaussian'); plt.plot(tr__.times(), tr__.data, label='gaussian_art'); plt.legend(); plt.show()
    #bp()
    
    
    ## If arrival and artificial noise requested, add artifial noise
    if not type_data == 'noise' and not i0_noise_toadd == -1:  
    
        tr_artificial, _, _ = pre_process_waveform(times, waveform['vTEC'].values, i0_noise_toadd, iend_noise_toadd, window, detrend=False, bandpass=[options['freq_min'], options['freq_max']])
        tr = get_right_var(tr, tr_artificial, snr, min_factor=0.01, max_factor=2., nb_tests=1000)
        #plt.plot(tr_.times(), tr_.data); plt.plot(tr.times(), tr.data); plt.show()
        
        #tr_=tr.copy(); tr_.data = tr_.data/snr + tr_artificial.data
        #tr_total, _, _ = pre_process_waveform(times, waveform['vTEC'].values, 0, waveform['vTEC'].values.size-1, 100000., detrend=False, bandpass=[options['freq_min'], options['freq_max']])
        #plt.plot(tr_total.times(), tr_total.data); plt.plot(tr.times()+tr_total.times()[i0], tr.data); plt.plot(tr_.times()+tr_total.times()[i0], tr_.data); plt.plot(tr_artificial.times()+tr_total.times()[i0_noise_toadd], tr_artificial.data); plt.show()
    
    if type_data == 'arrival' and False:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        axs[0].plot(waveform['time_s'].values, waveform['vTEC'].values); 
        axs[0].axvline(waveform['time_s'].values[i0], color='red')
        #axs[0].axvline(waveform['time_s'].values[iarrival_time], color='green')
        axs[0].axvline(waveform['time_s'].values[iend], color='red')
        
        tr_all, _, _ = pre_process_waveform(times, waveform['vTEC'].values, 0, len(times)-2, window,
                              detrend=False, bandpass=[options['freq_min'], options['freq_max']])
        
        axs[1].plot(waveform['time_s'].values[0] + tr_all.times(), tr_all.data)        
        axs[1].plot(waveform['time_s'].values[i0] + tr.times(), tr.data)
        plt.show()
        bp()
    
    
    ## Extract features from waveform
    features = extract_features_based_on_input_type(tr, type_data, type, options)
        
    ## Add static parameters
    features['event']     = event
    features['satellite'] = satellite
    features['station']   = station
    features['type-data'] = type_data
    features['sampling']  = waveform.sampling.iloc[0]
    features['arrival-time']  = times[i0]
    features['snr'] = snr
    
    return features

def get_noise_window(nb_picks, iarrival_time_in, window, min_deviation, size_subset, 
                     times, noise_pick_shift, flag_testing):

    """
    Find index locations for noise waveform bounds
    """
    
    ## If this data is just for testing (i.e., no arrival in TEC) we choose a random arrival time
    if flag_testing:
        #arrival_time  = times[0] + np.random.rand(1)[0]*(times[-1]-times[0]-window)*0.25
        #iarrival_time = np.argmin( abs((times-times[0]) - arrival_time) ) + 1
        available_range = np.arange(len(times)-size_subset)
    
    else:
        ideviation    = np.argmin( abs((times-times[0]) - min_deviation) ) 
        iarrival_time = iarrival_time_in
        ## Find index size where noise can not be picked
        inoise_pick_shift = np.argmin( abs((times-times[0]) - noise_pick_shift) ) + 1
        available_range   = np.arange(0, len(times)-size_subset, ideviation)
        available_range   = \
            available_range[ (available_range < iarrival_time-size_subset) 
                            | (available_range >= iarrival_time+inoise_pick_shift) ]
   
    np.random.shuffle(available_range)
   
    ## Extract a random starting index in noise
    #i0   = np.random.choice(available_range)
    i0 = available_range[:nb_picks]
    iend = i0 + size_subset
    
    ## TO remove
    #rand_shift = np.random.rand(1)[0] * ( times[-1] - 2*window - arrival_time - (noise_pick_shift) )
    #inoise     = np.argmin( abs(times - (arrival_time + noise_pick_shift + rand_shift)) )
                
    return i0, iend

def find_perturbed_window_bounds(nb_picks, times, window, min_deviation, size_subset, 
                                 iarrival_time, min_overlap_label, duration):

    """
    Find time indexes for a perturbed window around the arrival
    """

    ideviation       = np.argmin( abs((times-times[0]) - min_deviation) ) 
    iduration_signal = np.argmin( abs((times-times[0]) - duration) ) 
    ioverlap         = np.argmin( abs((times-times[0]) - window*(min_overlap_label)) ) 
    ioverlap         = min(ioverlap, np.argmin( abs((times-times[0]) - duration*(min_overlap_label)) ) )
    iremains_window  = size_subset - ioverlap
    #iremains_window = np.argmin( abs((times-times[0]) - window*(1.-min_overlap_label)) ) 
    available_range  = np.arange(max(iarrival_time - iremains_window, 0), 
                                 min(iarrival_time + iduration_signal - ioverlap, len(times)-1-size_subset), 
                                 ideviation)
    np.random.shuffle(available_range)

    """
    iduration_signal = np.argmin( abs((times-times[0]) - duration) ) 
    ioverlap         = np.argmin( abs((times-times[0]) - window*(min_overlap_label)) ) 
    ioverlap         = min(ioverlap, np.argmin( abs((times-times[0]) - duration*(min_overlap_label)) ) )
    iremains_window  = size_subset - ioverlap
    #iremains_window = np.argmin( abs((times-times[0]) - window*(1.-min_overlap_label)) ) 
    available_range  = np.arange(max(iarrival_time - iremains_window, 0), 
                                 min(iarrival_time + iduration_signal - ioverlap, len(times)-1-size_subset))
                                 
    ## Arrival time is at the end of the timeseries
    if available_range.size == 0:
        i0 = len(times)-1-size_subset
        available_range = [i0]
    else:
        i0 = np.random.choice(available_range)
     
    ## Check if i0 has already been used
    if not list_i0_used:
        list_i0_used.append(i0)
    else:
        cpt = 0
        ## We check other possible choices 
        while np.min(abs(i0-np.array(list_i0_used))) < int(0.2*len(available_range)) and cpt < 10:
            cpt += 1
            i0 = np.random.choice(available_range)
        
        if np.min(abs(i0-np.array(list_i0_used))) < 3:
            i0 = -1
    """
    
    ## The end index of the timeseries depends on the window, i.e., size_subset
    i0   = available_range[:nb_picks]
    iend = i0 + size_subset
    
    return i0, iend

def from_np_array(array_string):

    """
    Import array from dataframe csv file
    """

    array_converted = []

    rows = array_string.split(']\n [')
    for row in rows: 
        row = row.replace('[[', '').replace(']]', '').replace('\n', '').split()
        columns = []
        for column in row:
            columns.append( float(column) )
        array_converted.append( columns )
    
    return np.array(array_converted, dtype=float)

def load_features(dir_features, type): 

    """
    Load previously dumped features as dataframe
    """
    
    args = {}
    if type == 'spectrogram':
        args['converters'] = {'spectro': from_np_array}
    dtype = {'station': str, 'satellite': str, 'event': str, 'type_data': str}
    features_pd_loaded = pd.read_csv(dir_features, sep=',', header=[0], dtype=dtype, **args)
    
    return features_pd_loaded
 
def compute_features_one_set(tec_data, tec_data_param, options, type, grouped_data):

    features_pd = pd.DataFrame()

    ## Select one specific waveform
    #grouped_data = tec_data.groupby(['event', 'satellite', 'station'])
    for idata, data in grouped_data.iterrows():
    
        #event, satellite, station = group
        event, satellite, station = data.event, data.satellite, data.station
        
        waveform = tec_data.loc[(tec_data['station'] == station) 
                        & (tec_data['satellite'] == satellite) 
                        & (tec_data['event'] == event)]
    
        print('Processing station/satellite/event: ', station,'/', satellite,'/', event)
        params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                    & (tec_data_param['satellite'] == satellite) 
                    & (tec_data_param['event'] == event), : ]
        
        ## Flag data with missing parameters -> it means that there is no arrival, it is just for testing
        flag_testing = False
        if params.size == 0:
            flag_testing = True
        
        ## Extract basic waveform parameters
        if not flag_testing:
            params   = params.iloc[0]
            
        times    = waveform['UT'].values[:]*3600.
        sampling = waveform.iloc[0]['sampling']
        window = get_window(sampling, options['window'])
        
        tduration = options['signal_duration'][event]
        size_subset, iarrival_time, size_overlap = \
            extract_arrival_window_parameters(times, waveform, params, window, tduration, 
                                              options['min_overlap_label'], flag_testing, 
                                              default_size_overlap=20)
        
        ## Determine number of perturbed windows depending on the data class
        extra_windows = {'noise': options['nb_noise_windows_noarrival']}
        if not flag_testing:
            extra_windows['noise']   = options['nb_noise_windows']
            extra_windows['arrival'] = options['nb_arrival_windows']
            
        ## Loop over arrival times of waveform types
        for type_data in extra_windows:
            
            ## Perturb the arrival time to get some noise in the training data
            if type_data == 'noise':    
                l_i0, l_iend = get_noise_window(extra_windows[type_data], iarrival_time, window, 
                                                options['min_deviation'], size_subset, 
                                                times, options['noise_pick_shift'], flag_testing)
            else:
                
                l_i0, l_iend = find_perturbed_window_bounds(extra_windows[type_data], times, window, 
                                                            options['min_deviation'], size_subset, iarrival_time, 
                                                            options['min_overlap_label'], tduration)
            
            #if type_data == 'noise' and station == '0201' and event == 'Tohoku_1s':
            #    bp()
            
            for ishift_ in range(len(l_i0)):
            
                ## Select window bounds indexes
                i0   = l_i0[ishift_]
                iend = l_iend[ishift_]
                
                ## Skip waveforms where window has not been found
                if i0 == -1:
                    continue
                
                ## Skip waveforms for which we do not have arrivals
                if options['check_new_features'] and 'features' in options['load']:
                    features_pd_test = features_pd.loc[(features_pd['event'] == event) 
                                                        & (features_pd['satellite'] == satellite)
                                                        & (features_pd['event'] == event)
                                                        & (abs(features_pd['arrival-time'] - times[i0]) < 1e-5), :]
                    if(features_pd_test.size > 0):
                        continue
                
                ## Create a set of artificial noise to add to arrival waveforms
                l_i0_noise_toadd, l_iend_noise_toadd, l_snr = [], [], []
                if options['nb_windows_artificial_noise'] > 0:
                
                    l_i0_arrival_toadd, l_iend_arrival_toadd = find_perturbed_window_bounds(options['nb_windows_artificial_noise'], times, window, 
                                                            options['min_deviation'], size_subset, iarrival_time, 
                                                            options['min_overlap_label'], tduration)
                
                    l_i0_noise_toadd, l_iend_noise_toadd = get_noise_window(options['nb_windows_artificial_noise'], iarrival_time, window, 
                                                options['min_deviation'], size_subset, 
                                                times, options['noise_pick_shift'], flag_testing)
                                                
                    l_snr = np.random.uniform(options['augment_noise_snr_min'], options['augment_noise_snr_max'], len(l_iend_noise_toadd))
                
                elif options['add_gaussian_noise']:
                    l_snr = np.random.uniform(options['augment_noise_snr_min'], options['augment_noise_snr_max'], 1)
                    
                ## If arrival and artificial noise requested, add artifial noise
                if not type_data == 'noise' and not len(l_i0_noise_toadd) == 0:  
                    for i0_noise_toadd, iend_noise_toadd, i0_arrival_toadd, iend_arrival_toadd, snr in zip(l_i0_noise_toadd, l_iend_noise_toadd, l_i0_arrival_toadd, l_iend_arrival_toadd, l_snr):
                    
                        ## Extract features from waveform
                        features = extract_one_feature(times, waveform, params, tduration, i0_arrival_toadd, iend_arrival_toadd, 
                                                       window, event, satellite, station, type_data, options, i0_noise_toadd=i0_noise_toadd, 
                                                       iend_noise_toadd=iend_noise_toadd,snr=snr, type=type, add_gaussian_noise=options['add_gaussian_noise'])
                                                       
                        features_pd = features_pd.append( features ) 
                
                #options['name_test'] = '{event}__{satellite}_{station}_{type}_{i}.png'.format(event=event, satellite=satellite, station=station, type=type_data, i=ishift_)
                ## Extract features from waveform
                if len(l_snr) > 0:
                    features = extract_one_feature(times, waveform, params, tduration, i0, iend, window, event, satellite, station, type_data, options, type=type, add_gaussian_noise=options['add_gaussian_noise'], snr=l_snr[0])
                else:
                    features = extract_one_feature(times, waveform, params, tduration, i0, iend, window, event, satellite, station, type_data, options, type=type, add_gaussian_noise=options['add_gaussian_noise'])
                features_pd = features_pd.append( features )
        
    return features_pd
 
def compute_features_all_waveforms(tec_data_in, tec_data_param, options, seed=1, type='features', nb_CPU=1):

    """
    Compute features for each waveform in the TEC database
    """
    
    ## Only compute features for data with arrivals
    #tec_data = tec_data_in.loc[~tec_data_in['only_for_testing'], :]
    tec_data = tec_data_in
    
    if 'features' in options['load']:
        features_pd_loaded = load_features(options['load']['features'], type)
        
    ## Setup partial function to compute features over on tec_data dataset
    compute_features_one_set_partial = partial(compute_features_one_set, tec_data, tec_data_param, options, type)
        
    ## Nb waveforms
    grouped_data = tec_data.groupby(['event', 'satellite', 'station']).first().reset_index()[['event', 'satellite', 'station']]
    nb_data = grouped_data.shape[0]
    N = min(nb_data, nb_CPU)
    
    ## Initialize features
    features_pd = pd.DataFrame()
    if (options['check_new_features'] and 'features' in options['load']) \
        or not 'features' in options['load']:
        
        ## Get same randomization across runs for a given seed
        np.random.seed(seed)
        
        ## Retrieve sampling
        sampling = tec_data.iloc[0]['sampling']
        window = get_window(sampling, options['window'])
        
        ## Deploy feature extraction on CPUs
        if N == 1:
            features_pd = compute_features_one_set_partial(grouped_data)
        
        else:
            step_idx =  nb_data//N
            list_of_lists = []
            for i in range(N):
                idx = np.arange(i*step_idx, (i+1)*step_idx)
                if i == N-1:
                    idx = np.arange(i*step_idx, nb_data)
                list_of_lists.append( grouped_data.iloc[idx] )
            
            with get_context("spawn").Pool(processes = N) as p:
                results = p.map(compute_features_one_set_partial, list_of_lists)

            for result in results:
                features_pd = features_pd.append( result )
            
        """
        ## Select one specific waveform
        grouped_data = tec_data.groupby(['event', 'satellite', 'station'])
        for group, waveform in grouped_data:
        
            event, satellite, station = group
            
            print('Processing station/satellite/event: ', station,'/', satellite,'/', event)
            params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                        & (tec_data_param['satellite'] == satellite) 
                        & (tec_data_param['event'] == event), : ]
            
            ## Flag data with missing parameters -> it means that there is no arrival, it is just for testing
            flag_testing = False
            if params.size == 0:
                flag_testing = True
            
            ## Extract basic waveform parameters
            if not flag_testing:
                params   = params.iloc[0]
                
            times    = waveform['UT'].values[:]*3600.
            sampling = waveform.iloc[0]['sampling']
            window = get_window(sampling, options['window'])
            
            tduration = options['signal_duration'][event]
            size_subset, iarrival_time, size_overlap = \
                extract_arrival_window_parameters(times, waveform, params, window, tduration, 
                                                  options['min_overlap_label'], flag_testing, 
                                                  default_size_overlap=20)
            
            ## Determine number of perturbed windows depending on the data class
            extra_windows = {'noise': options['nb_noise_windows_noarrival']}
            if not flag_testing:
                extra_windows['noise']   = options['nb_noise_windows']
                extra_windows['arrival'] = options['nb_arrival_windows']
                
            ## Loop over arrival times of waveform types
            for type_data in extra_windows:
                
                ## Perturb the arrival time to get some noise in the training data
                if type_data == 'noise':    
                    l_i0, l_iend = get_noise_window(extra_windows[type_data], iarrival_time, window, 
                                                    options['min_deviation'], size_subset, 
                                                    times, options['noise_pick_shift'], flag_testing)
                else:
                    
                    l_i0, l_iend = find_perturbed_window_bounds(extra_windows[type_data], times, window, 
                                                                options['min_deviation'], size_subset, iarrival_time, 
                                                                options['min_overlap_label'], tduration)
                
                #if type_data == 'noise' and station == '0201' and event == 'Tohoku_1s':
                #    bp()
                
                for ishift_ in range(len(l_i0)):
                
                    ## Select window bounds indexes
                    i0   = l_i0[ishift_]
                    iend = l_iend[ishift_]
                    
                    ## Skip waveforms where window has not been found
                    if i0 == -1:
                        continue
                    
                    ## Skip waveforms for which we do not have arrivals
                    if options['check_new_features'] and 'features' in options['load']:
                        features_pd_test = features_pd.loc[(features_pd['event'] == event) 
                                                            & (features_pd['satellite'] == satellite)
                                                            & (features_pd['event'] == event)
                                                            & (abs(features_pd['arrival-time'] - times[i0]) < 1e-5), :]
                        if(features_pd_test.size > 0):
                            continue
                    
                    ## Create a set of artificial noise to add to arrival waveforms
                    l_i0_noise_toadd, l_iend_noise_toadd, [] = [], [], []
                    if options['nb_windows_artificial_noise'] > 0:
                        l_i0_noise_toadd, l_iend_noise_toadd = get_noise_window(options['nb_windows_artificial_noise'], iarrival_time, window, 
                                                    options['min_deviation'], size_subset, 
                                                    times, options['noise_pick_shift'], flag_testing)
                        l_snr = np.random.uniform(1., 2., len(l_iend_noise_toadd))
                    
                    ## If arrival and artificial noise requested, add artifial noise
                    if not type_data == 'noise' and not len(l_i0_noise_toadd) == 0:  
                        for i0_noise_toadd, iend_noise_toadd, snr in zip(l_i0_noise_toadd, l_iend_noise_toadd, l_snr):
                            ## Extract features from waveform
                            features = extract_one_feature(times, waveform, i0, iend, window, event, satellite, station, type_data, options,
                                                           i0_noise_toadd=i0_noise_toadd, iend_noise_toadd=iend_noise_toadd, snr=snr, type=type)
                            features['snr'] = snr
                            features_pd = features_pd.append( features ) 
                    
                    ## Extract features from waveform
                    features = extract_one_feature(times, waveform, i0, iend, window, event, satellite, station, type_data, options, type=type)
                    features['snr'] = 0.
                    features_pd = features_pd.append( features )  
            """
            
        ## Append previously generated features with new ones
        if 'features' in options['load']:
            features_pd = features_pd.append( features_pd_loaded )
            features_pd.drop_duplicates(inplace=True, subset=['event', 'satellite', 'station', 'arrival-time', 'snr'], keep='last')
    
    else:
    
        features_pd = features_pd_loaded
    
    features_pd.reset_index(drop=True, inplace=True)
    
    ## Save features to file
    if (options['check_new_features'] and 'features' in options['load']) \
        or not 'features' in options['load']:
        
        if options['save_features']:
            features_pd.to_csv(options['DIR_DATA'] + 'features_'+type+'_m'+str(options['min_overlap_label'])+'_w'+str(window) +'.csv', sep=',', header=True, index=False)
    
    ## Plot statistics
    if options['plot_correlations']:
        list_corr = [key for key in features_pd.loc[:, ~features_pd.columns.isin(['type-data', 'event', 'satellite', 'station'])].keys()]
        plot_statistics(features_pd, list_corr, options)   

    return features_pd       
  
def get_event_name(filepath):

    """
    Determine event name from root folder name with template: ML_eventname_sampling
    """

    event = filepath.replace('//', '/')
    event = event.split('/')[-2]
    event = '_'.join(event.split('_')[1:])
    return event

def load_one_param(filepath, default_duration=150.):
    
    """
    Load one param file as a DataFrame
    """
    
    file = filepath.split('/')[-1]
    name  = file.split('_')
    event = get_event_name(filepath)
    
    tec_data_param_ = pd.read_csv(filepath, header=None, delim_whitespace=True)
    dtype = {'station': str, 'epoch': int, 'time': float, 'amp-TEC': float, 't-ampmax-TEC': int, 'ampmax-TEC': float, 'diff-t': float, 'amp': float, 'slope': float}
    if tec_data_param_.shape[1] == 9:
        columns = ['station', 'epoch', 'time', 'amp-TEC', 't-ampmax-TEC', 'ampmax-TEC', 'diff-t', 'amp', 'slope']
        tec_data_param_ = pd.read_csv(filepath, header=None, delim_whitespace=True, dtype=dtype, names=columns)
        if tec_data_param_['diff-t'].iloc[0] == 0.:
            tec_data_param_['diff-t'] = default_duration
        
    ## TODO: we should not have any data with only 5 columns. Remove his eventually
    if tec_data_param_.shape[1] == 5:
        sys.exit('Number of input parameters in param file not recognized!')
        #columns = ['station', 'epoch', 'time', 't-ampmax-TEC', 'ampmax-TEC']
        #tec_data_param_ = pd.read_csv(filepath, header=None, delim_whitespace=True, dtype=dtype, names=columns)
        
    ## Different format for ML_Sumatra_2
    if tec_data_param_.shape[1] == 4:
        columns = ['station', 'epoch', 't-ampmax-TEC', 'ampmax-TEC']
        tec_data_param_ = pd.read_csv(filepath, header=None, delim_whitespace=True, dtype=dtype, names=columns)
        tec_data_param_['diff-t'] = default_duration
       
    ## Make sure that string is 4 characters
    station = tec_data_param_.station.iloc[0]
    while len(station) < 4:
        station = '0' + station
    
    tec_data_param_['station']   = station
    tec_data_param_['satellite'] = name[2]
    tec_data_param_['doy']       = int(name[3])
    tec_data_param_['year']      = int(name[4])
    tec_data_param_['event']     = event
    
    return tec_data_param_
  
def load_one_rTEC(filepath):
    
    """
    Load one rTEC file as a DataFrame
    """
    
    file = filepath.split('/')[-1]
    
    try:
     tec_data_ = pd.read_csv(filepath, header=None, delim_whitespace=True)
    except:
     tec_data_ = pd.read_csv(filepath, header=None, delim_whitespace=True, skiprows=[0])
     
    name  = file.split('_')
    event = get_event_name(filepath)
    
    try:
     tec_data_.columns = ['epoch', 'UT', 'LOS', 'az', 'lat', 'lon', 'sTEC', 'vTEC']
    except:
     tec_data_.columns = ['epoch', 'UT', 'LOS', 'az', 'lat', 'lon', 'sTEC', 'vTEC', 'dummy']
     
    tec_data_['time_s']    = tec_data_['UT']*3600. # Fix to make sure that the first column is actually seconds and not epoch
    tec_data_['station']   = str(name[1])
    tec_data_['satellite'] = name[2]
    tec_data_['doy']       = int(name[3])
    tec_data_['year']      = int(name[4])
    tec_data_['event']     = event
    tec_data_['sampling']  = np.round(tec_data_['time_s'].iloc[1] - tec_data_['time_s'].iloc[0])
    tec_data_['file'] = file # to remove
    
    return tec_data_
  
def load_tec_files(file_tec_data, file_tec_data_param, options):
    
    """
    Load previously stored TEC data in csv files
    """

    file_tec_data       = options['DIR_DATA'] + file_tec_data
    file_tec_data_param = options['DIR_DATA'] + file_tec_data_param

    if not os.path.isfile(file_tec_data) or not os.path.isfile(file_tec_data_param):
        sys.exit('TEC data files not found in: ' + options['DIR_DATA'])
        
    dtype = {'epoch': int, 'UT': float, 'LOS': float, 'az': float, 'lat': float, 'lon': float, 'sTEC': float, 'vTEC': float, 
             'time_s': float, 'station': str, 'satellite': str, 'doy': int, 'year': int, 'event': str, 'sampling': float}
    tec_data       = pd.read_csv(options['DIR_DATA'] + 'tec_data.csv', header=[0], dtype=dtype)
    
    dtype = {'station': str, 'epoch': int, 'time': float, 'amp-TEC': float, 't-ampmax-TEC': int, 'ampmax-TEC': float,
             'diff-t': float, 'amp': float, 'slope': float, 'satellite': str, 'doy': int, 'year': int, 'event': str,
             'arrival-time': float}
    tec_data_param = pd.read_csv(options['DIR_DATA'] + 'tec_data_param.csv', header=[0], dtype=dtype)
    
    return tec_data, tec_data_param
  
def remove_duplicates(tec_data, waveform):

    """
    Remove duplicate epochs in waveforms
    """
    
    diff = np.diff(waveform.epoch.values)
    
    loc_duplicate = np.where(diff == 0)[0]
    for i in loc_duplicate:
        row = waveform.iloc[i+1] # remove last duplicate
        tec_data.loc[tec_data.index == row.name, 'passed_quality_check'] = False
     
def find_gaps(diff, waveform_without_duplicates):
    
    """
    Find gaps for a given sampling because epoch can have different meaning:
        - 1 epoch = 1s
        - 1 epoch = sampling
    """
    
    loc_gaps, elements_per_gap = np.empty([]), np.empty([])
    
    ## Only find gaps if there are gaps
    if waveform_without_duplicates.size > 0 and diff.size > 0:
    
        ## Check the consistency between 1 epoch and sampling rate
        d_epoch1 = waveform_without_duplicates['epoch'].iloc[1] \
                  - waveform_without_duplicates['epoch'].iloc[0]
        d_epoch2 = waveform_without_duplicates['epoch'].iloc[1] \
                  - waveform_without_duplicates['epoch'].iloc[0]
        d_epoch = min(d_epoch1, d_epoch2) # Just to get right depoch if data are missing at the beginning
        threshold_gap = 1
        if d_epoch > 1:
            d_UT = 3600.*( waveform_without_duplicates['UT'].iloc[1] 
                         - waveform_without_duplicates['UT'].iloc[0] )
            if abs(d_UT - waveform_without_duplicates['sampling'].iloc[0]) < 1e-10:
                threshold_gap = d_UT*1.5
        
        loc_gaps      = np.where(diff > threshold_gap)[0]
        size_gaps = diff[loc_gaps]
        elements_per_gap = np.zeros(size_gaps.shape, dtype=int)
        for isize, size_gap in enumerate(size_gaps):
            elements_per_gap[isize] = len(np.arange(d_epoch, size_gap, d_epoch))
    
    return loc_gaps, elements_per_gap
     
def find_gaps_on_quality_checked_waveform(tec_data, station, satellite, event):
    waveform_without_duplicates = get_one_entry(tec_data, station, satellite, event)
    waveform_without_duplicates = \
        waveform_without_duplicates.loc[waveform_without_duplicates['passed_quality_check'], :]
    diff = np.diff(waveform_without_duplicates.epoch.values)    
    loc_gaps, elements_per_gap = find_gaps(diff, waveform_without_duplicates)
    return loc_gaps, elements_per_gap, waveform_without_duplicates
     
def do_one_interpolation(waveform_without_duplicates, column_to_interp):
    
    time = waveform_without_duplicates['time_s'].values
    amp  = waveform_without_duplicates[column_to_interp].values
    tr = compute_params_waveform.create_Trace(time, amp, bandpass = [], detrend = False, differentiate=False)
    
    f = interpolate.interp1d(waveform_without_duplicates['epoch'].values, tr.data, kind='cubic', fill_value='extrapolate')
    
    return f
    
def interpolate_gaps(tec_data, station, satellite, event, param, max_gap_size=80):

    """
    Interpolate waveforms to fix data gaps
    """
    
    loc_gaps, elements_per_gap, waveform_without_duplicates = \
        find_gaps_on_quality_checked_waveform(tec_data, station, satellite, event)
    remove_data_after_large_gaps(tec_data, waveform_without_duplicates, param, 
                                 loc_gaps, elements_per_gap, max_gap_size)
    loc_gaps, elements_per_gap, waveform_without_duplicates = \
        find_gaps_on_quality_checked_waveform(tec_data, station, satellite, event)
    
    ## Only interpolates if there are gaps and a valid waveform
    if waveform_without_duplicates.size > 0 and loc_gaps.size > 0:
    
        columns_to_interp = ['LOS', 'az', 'lat', 'lon', 'sTEC', 'vTEC']
        f = {}
        for column_to_interp in columns_to_interp:
           f[column_to_interp] = do_one_interpolation(waveform_without_duplicates, column_to_interp)
           
        ## Interpolate over each missing data point
        for i, nb_i in zip(loc_gaps, elements_per_gap):
            new_row  = waveform_without_duplicates.iloc[i].copy()
                
            for j in range(nb_i):
                new_row['epoch']  += 1
                new_row['UT']     += new_row['sampling']/3600.
                new_row['time_s'] += new_row['sampling']
                
                for column_to_interp in columns_to_interp:
                    new_row[column_to_interp] = f[column_to_interp](new_row['epoch'])
                #tec_data = pd.concat([tec_data.iloc[:new_row.name+1], 
                #                      pd.DataFrame([new_row]), tec_data.iloc[new_row.name+1:]], 
                #                      ignore_index=True).reset_index(drop=True)
                tec_data = tec_data.append(new_row.copy())    
    
    #if loc_gaps.size > 0: bp()
    # event, satellite, station = new_row.event, new_row.satellite, new_row.station
    # waveform = get_one_entry(tec_data, station, satellite, event)
    # tr = create_Trace(waveform['time_s'].values, waveform['vTEC'].values)
    return tec_data
  
def remove_data_after_large_gaps(tec_data, waveform, param, loc_gaps, elements_per_gap, max_gap_size):
    
    """
    Remove large data gaps in data.
    If large data gaps (gaps > max_gap_size) are present we either: 
        1) only keep the data chunk where the arrival is present if arrival before the first or after the last data gap
        2) remove all data if arrival between data gaps
    """
    
    loc_large = np.where(elements_per_gap > max_gap_size)[0] # location of all large data gaps in loc_gaps
    loc_large = loc_gaps[loc_large] # locations of data gaps in waveform
    
    if loc_large.size > 0:
        
        ## If there is an arrival, we remove some of the data
        if param.size > 0:
            arrival_time = param.iloc[0]['epoch']
            loc_arrival_time = np.argmin(abs(waveform['epoch'] - arrival_time))
            minlarge, maxlarge = min(loc_large), max(loc_large)
            if loc_arrival_time < minlarge:
                waveform_ = waveform.iloc[minlarge:]
                tec_data.loc[tec_data.index.isin(waveform_.index), 'passed_quality_check'] = False
                
            elif loc_arrival_time > maxlarge:
                waveform_ = waveform.iloc[:maxlarge]
                tec_data.loc[tec_data.index.isin(waveform_.index), 'passed_quality_check'] = False
                
            else:
                tec_data.loc[tec_data.index.isin(waveform.index), 'passed_quality_check'] = False
                
        ## If no arrival, remove all waveform
        else:
            tec_data.loc[tec_data.index.isin(waveform.index), 'passed_quality_check'] = False
  
def get_one_entry(tec_data, station, satellite, event):
    
    """
    Get one waveform from a given event, satellite, and station
    """
    waveform = tec_data.loc[ (tec_data['station'] == station) 
                        & (tec_data['satellite'] == satellite) 
                        & (tec_data['event'] == event), :]
  
    return waveform
  
def update_sampling(tec_data, station, satellite, event):
    
    """
    Update sampling after duplicated have been removed
    """
    
    waveform = get_one_entry(tec_data, station, satellite, event)
    waveform_without_duplicates = waveform.loc[waveform['passed_quality_check'], :]
    dt = waveform_without_duplicates['time_s'].iloc[1] \
        - waveform_without_duplicates['time_s'].iloc[0]
    tec_data.loc[tec_data.index.isin(waveform.index), 'sampling'] = dt
  
def check_and_convert_epoch_to_arrival_times(tec_data, tec_data_param):

    """
    Convert epochs
    """
    
    print('Data quality check')
    
    ## Initialize column for arrival times in seconds from beginning of the day
    tec_data_param['arrival-time'] = -1.
    
    missing_data = pd.DataFrame()
    tec_data['passed_quality_check'] = True
    tec_data['only_for_testing']     = False
    tec_data.reset_index(inplace=True, drop=True)
    grouped_tec = tec_data.groupby(['station', 'satellite', 'event'])
    for group, waveform in grouped_tec:
    #for iparam, param in tec_data_param.iterrows():
        
        #station, satellite, event = param['station'], param['satellite'], param['event']
        station, satellite, event = group
        #if int(station) < 1000: continue
        print('- Checking param for: ', station, satellite, event)
        param    = get_one_entry(tec_data_param, station, satellite, event)

        ## For waveforms without param files, data are not labelled so we only use for testing
        if param.size == 0:
            tec_data.loc[tec_data.index.isin(waveform.index), 'only_for_testing'] = True
        
        ## Find gaps and duplicates in data and correct
        remove_duplicates(tec_data, waveform)
        update_sampling(tec_data, station, satellite, event)
        waveform_save = get_one_entry(tec_data, station, satellite, event)
        tec_data = interpolate_gaps(tec_data, station, satellite, event, param, max_gap_size=50)
        waveform = get_one_entry(tec_data, station, satellite, event)
        waveform_without_duplicates = waveform.loc[waveform['passed_quality_check'], :]
        
        if param.size > 0 and waveform_without_duplicates.size > 0: # param data is present, i.e., there is an arrival
            param = param.iloc[0]
            f = interpolate.interp1d(waveform_without_duplicates.epoch.values, 
                                      waveform_without_duplicates.UT.values, 
                                      kind='cubic', fill_value='extrapolate')
            arrival_time = f(param['epoch'])
            if arrival_time > waveform.UT.max():
                bp() # problem with arrival time not found in current waveform
            tec_data_param.loc[tec_data_param.index == param.name, 'arrival-time'] = arrival_time*3600.
            
    tec_data = tec_data.loc[tec_data['passed_quality_check'], ~tec_data.columns.isin(['passed_quality_check'])].reset_index(drop=True)
    tec_data.sort_values(by=['event', 'satellite', 'station', 'epoch'], inplace=True)
    #tec_data.loc[tec_data.file == 'rtTEC_0166_G26_070_11', :]
    return tec_data, tec_data_param
  
def read_data_folders(options):

    """
    Read all TEC data and parameters from data/ folder
    """

    file_tec_data       = 'tec_data.csv'
    file_tec_data_param = 'tec_data_param.csv'
    files_to_exclude = [file_tec_data, file_tec_data_param]
    if options['load_tec_data']:
        tec_data_loaded, tec_data_param_loaded = load_tec_files(file_tec_data, file_tec_data_param, options)
        
    tec_data, tec_data_param = pd.DataFrame(), pd.DataFrame()
    if (options['check_new_tec_data'] and options['load_tec_data']) \
        or not options['load_tec_data']:

        print('Start reading data')
        for  subdir, dirs, files in os.walk(options['DIR_DATA']):
            
            for file in files:
            
                filepath = subdir + os.sep + file
                
                ## Skip csv tec_data files
                if file in files_to_exclude:
                    continue
                
                ## TODO: remove this
                #if not 'ML_Illapel_30s' in filepath:
                #    continue
                
                if 'param' in file:
                    
                    tec_data_param_ = load_one_param(filepath, default_duration=options['default_duration'])
                    tec_data_param = tec_data_param.append( tec_data_param_ )
                
                elif 'rtTEC' in file:
                
                    ## If the current rTEC file is already in database, we skip
                    if options['check_new_tec_data'] and options['load_tec_data']:
                        tec_data_loaded_item = tec_data_loaded.loc[tec_data_loaded['file'] == file, :]
                        if tec_data_loaded_item.size > 0:
                            continue
                
                    tec_data_ = load_one_rTEC(filepath)
                    tec_data = tec_data.append( tec_data_ )
                    
        tec_data.reset_index(drop=True, inplace=True)
        tec_data_param.reset_index(drop=True, inplace=True)
        tec_data, tec_data_param = check_and_convert_epoch_to_arrival_times(tec_data, tec_data_param)
        
    ## Append loaded data
    if options['load_tec_data']:
        tec_data = tec_data.append( tec_data_loaded )
        tec_data_param = tec_data_param.append( tec_data_param_loaded )
    
    ## Post processing to add default values if no wignal duration provided
    tec_data.drop_duplicates(inplace=True, subset=['event', 'satellite', 'station', 'time_s'], keep='last')
    tec_data.reset_index(drop=True, inplace=True)
    tec_data_param.drop_duplicates(inplace=True, subset=['event', 'satellite', 'station'], keep='last')
    tec_data_param.loc[tec_data_param['diff-t'] == 0., :] = options['default_duration']
    tec_data_param.reset_index(drop=True, inplace=True)
    
    if (options['check_new_tec_data'] and options['load_tec_data']) \
        or not options['load_tec_data']:
        
        if options['save_tec_data']:
            tec_data.to_csv(options['DIR_DATA'] + 'tec_data.csv', header=True, index=False)
            tec_data_param.to_csv(options['DIR_DATA'] + 'tec_data_param.csv', header=True, index=False)
    
    return tec_data, tec_data_param

def get_str_options(options_in):

    """
    Create name for dumping data and figures based on chosen options
    """

    options = options_in.copy()
    list_windows = [window for window in options['window'].keys()]
    options['window_picked'] = options['window'][list_windows[0]]
    list_labels = ['shift_detection_after_max', 'min_overlap_label', 'noise_pick_shift', 'window_picked'] 
    str_options = [label[0] + str(options[label]) for label in list_labels]
    
    return '_'.join(str_options)
   