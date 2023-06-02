#!/usr/bin/env python3
import numpy as np
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import itertools
import seaborn as sns
from matplotlib.collections import QuadMesh
import string
import copy
from obspy.signal.trigger import classic_sta_lta

from multiprocessing import get_context
from functools import partial

import read_data, train_est


def dfinterface_process_timeseries_with_forest(x, time_end, est, tec_data, tec_data_param, 
                                               columns_in, options, plot_probas):
    
    """
    #Compute probabilities using a given estimator for a single event/satellite/station
    """
    
    event, satellite, station = x.event, x.satellite, x.station
    obs = train_est.process_timeseries_with_forest(time_end, est, tec_data, tec_data_param, 
                                                   event, satellite, station, 
                                                   columns_in, options, 
                                                   plot_probas=plot_probas) 
    for key in obs:
        x[key] = obs[key]
        
    return x 
       
def process_batch_satellites(satellites, stations, event, time_end, est, 
                             tec_data, tec_data_param, data, columns_in, 
                             options, plot_probas=False):
        
    """
    #Compute probabilities using a given estimator for a list of event/satellite/station
    """
        
    ## Select available event/satellite/station from a list of requested event/satellite/station
    data_to_process = tec_data.loc[(tec_data['station'].isin(stations)) 
                        & (tec_data['satellite'].isin(satellites)) 
                        & (tec_data['event'] == event), 
                        ['event', 'satellite', 'station']].drop_duplicates()
                    
    ## Compute probabilities for each station
    args = [time_end, est, tec_data, tec_data_param, columns_in, options, plot_probas]
    data_to_process = \
        data_to_process.apply(dfinterface_process_timeseries_with_forest, args=args, axis=1)
    
    return data_to_process

def test_detections_with_forest(est, tec_data, tec_data_param, data, options, detection_parameters, plot_probas=False):
                
    """
    #Run detection process on a list of stations/satellites for a given event up to time detection_parameters['time_end']
    """
                
    name  = detection_parameters['name']
    event = detection_parameters['event']
    satellites = detection_parameters['satellites']
    stations   = detection_parameters['stations']
    time_end   = detection_parameters['time_end']
    #sampling   = detection_parameters['sampling']
    if not 'detections' in options['load'].keys():
        
        input_columns = train_est.data_without_info_columns(data).columns
        results = process_batch_satellites(satellites, stations, event, time_end, est, 
                                           tec_data, tec_data_param, data, input_columns, 
                                           options, plot_probas=plot_probas)
        name_f = options['DIR_FIGURES'] + 'detections_' + name + '_' + read_data.get_str_options(options) + '.pkl'
        with open(name_f, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        
        name_f = options['load']['detections']
        with open(name_f, 'rb') as f:
                results = pickle.load(f)
    
    return results

########################
## OLD ROUTINES ABOVE ##
########################
    
def analytical_detector_one_sample(vTEC, thresholds_one_sampling):

    """Check if an array of vTEC points fulfill certain threshold conditions
    
    :param vTEC: Input vTEC to check for detection
    :type vTEC: np.array
    :param thresholds_one_sampling: Dictionnary of thresholds 's1', 's2', ...
    :type thresholds_one_sampling: dict
    :return: Whether or not there is a detection in this array of vTEC 
    :rtype: Integer
    """
    
    detection = 1
    for i_s in range(1,len(thresholds_one_sampling)+1):
        s = abs(vTEC[i_s] - vTEC[0])
        if s < thresholds_one_sampling['s'+str(i_s)]:
            detection = 0
    
    return detection
    
def analytical_detector(waveform, thresholds, window, true_arrival, duration, factor=0.7, nb_pts_for_class=12):
 
    """Detect arrivals on a given vTEC trace and on a threshold dictionnary
    
    :param waveform: Input vTEC trace
    :type waveform: pd.DataFrame
    :param thresholds: Dictionnary of thresholds 's1', 's2', ... for different sampling rates
    :type thresholds: dict
    :return: DataFrame of detection against time
    :rtype: pd.DataFrame
    """
    
    times = waveform['time_s'].values
    vTEC  = waveform['vTEC'].values
    current_sampling = waveform['sampling'].iloc[0]
    current_sampling = int(np.round(current_sampling))
    
    if not current_sampling in thresholds:
        sys.exit('Trace sampling rate has no corresponding threshold condition')
    
    thresholds_one_sampling = thresholds[current_sampling]
    N_thresholds = len(thresholds_one_sampling)
    
    detections = pd.DataFrame(columns=['time', 'class'])
    current_arrival_class = 0
    for itime in range(0, len(times)-N_thresholds):
        vTEC_chunk = vTEC[itime:itime+N_thresholds+1]
        detection = analytical_detector_one_sample(vTEC_chunk, thresholds_one_sampling)
        
        ## If this sample is noise we update the arrival class number for the next sample
        if detection == 0:
            current_arrival_class += 1
            
        loc_dict = {
            'time': times[itime],
            'class': detection,
            'arrival_class': current_arrival_class,
        }
        detections = detections.append( [loc_dict] )
    
    ## Only keep arrival classes with at least "nb_pts_for_class" points
    detections['old_class'] = detections['arrival_class']
    #detections['arrival_class'] = detections['old_class']
    detections.loc[detections['class']==0, 'arrival_class'] = -1
    detections['count'] = detections.groupby('arrival_class')['class'].transform('count')
    detections.loc[detections['count'] < nb_pts_for_class, 'arrival_class'] = -1
    
    
    """
    best_detections = detections.loc[detections.arrival_class > -1, :].groupby('arrival_class').first().reset_index()
    plt.plot(times, vTEC); 
    for idetect, detect in best_detections.iterrows(): plt.axvline(detect.time);
    plt.xlabel('Time (UT)')
    plt.ylabel('vTEC')
    entry = waveform.iloc[0]
    station, satellite, event = entry.station, entry.satellite, entry.event
    plt.title(event +' - '+satellite+' - ' + station)
    plt.show()
    plt.savefig('analytical_cris_Illapel.pdf')
    """
    result = compute_metric_FTPN_detector(times, window, detections, true_arrival, duration, factor=0.7)
    
    return detections, result
    
def recursively_find_wavetrains_STA_LTA(times, cft, threshold_in, threshold_out):

    """
    Find all STA/LTA detections based on detection threshold (threshold_in) 
    and end of detection threshold (threshold_out)
    """

    detections = pd.DataFrame(columns=['time', 'proba', 'arrival_class'])
    detections['time']  = times
    detections['proba'] = cft
    detections['arrival_class'] = -1
    detections.reset_index(inplace=True, drop=True)

    ## All indexes over detection threshold
    loc_over_threshold  = np.where(cft>=threshold_in)[0]
    
    if loc_over_threshold.size > 0:
    
        loc          = loc_over_threshold[0]
        loc_previous = -1
        iarrival_class = -1
        while loc < loc_over_threshold.max() and loc > loc_previous:
        
            ## Save current state
            loc_previous = loc
        
            ## Update class number
            iarrival_class += 1
        
            ## All indexes over detection threshold
            loc_below_threshold = np.where(cft[loc:]>=threshold_out)[0]
            diff_loc = np.diff(loc_below_threshold)
            loc_diff_over_one = np.where(diff_loc > 1)[0]
            
            ## If there are gaps in points below the end-of-detection threshold
            if loc_diff_over_one.size >= 0:
                if loc_diff_over_one.size == 0:
                    index_end = loc + len(loc_below_threshold) - 1
                else:
                    index_end = loc + loc_diff_over_one[0]
                for time, cft_point in zip(times[loc:index_end], cft[loc:index_end]):
                    loc_dict = {
                        'time': time,
                        'proba': cft_point,
                        'arrival_class': iarrival_class
                    }
                    detections.loc[detections.time == time, 'arrival_class'] = iarrival_class
                    #detections = detections.append( [loc_dict] )
                    
                ## We find the next group of detections
                loc = loc_over_threshold[loc_over_threshold>index_end]
                ## If not found, we exit the loop
                if loc.size > 0:
                    loc = loc[0] 
                else:
                    loc = loc_over_threshold[-1]+1
                   
            ## Otherwise we leave the loop
            else:
                loc = loc_over_threshold[-1]+1
        
    return detections
  
def compute_metric_FTPN_detector(times, window, detections, true_arrival, duration, factor=0.7):
    
    """
    Compute false and true negatives and positives using a given time sampling corresponding to RF classification
    """
    
    times_arrival = np.array([])
    if true_arrival > -1:
        min_overlap = factor*min(window, duration)
        times_arrival = times[ ((times < true_arrival) & (times+window > true_arrival+duration)) 
                            | ((times <= true_arrival+duration) & (times+window >= true_arrival+duration ) & (true_arrival + duration - times >= min_overlap ))
                            | ((times <= true_arrival) & (times+window >= true_arrival ) & ((times+window) - true_arrival >= min_overlap )) ]
    
        #times_arrival = times[ ((times < true_arrival) & (times+window > true_arrival+duration)) | ((times < true_arrival+duration) & (times+window >= true_arrival+duration ) & (true_arrival + duration - times >= min_overlap )) | ((times < true_arrival) & (times+window >= true_arrival ) & ((times+window) - true_arrival >= min_overlap )) ]
        
    result = {
        'TP': 0,
        'FP': 0,
        'TN': len(times),
        'FN': 0,
    }
    times_STA_LTA_all = np.array([])
    if true_arrival > -1:
        confirmed_detections = detections.loc[detections.arrival_class > -1, :]
        for iclass, arrival_class in confirmed_detections.groupby('arrival_class'):
            min_time = arrival_class.time.min()
            max_time = arrival_class.time.max()
            
            #times_STA_LTA = times[ ((times < min_time) & (times+window > min_time) & (times+window-min_time >= factor*window)) 
            #                        | ((times < max_time) & (times+window > max_time ))
            #                        | ((times >= min_time) & (times+window <= max_time ) & (min_time-times >= factor*window )) ]
            times_STA_LTA = times[(times>=min_time) & (times<=max_time)]

            intersection_LTA = np.intersect1d(times_STA_LTA, times_STA_LTA_all)
            times_STA_LTA = np.setdiff1d(times_STA_LTA, intersection_LTA)
            times_STA_LTA_all = np.concatenate((times_STA_LTA_all, times_STA_LTA))

            intersection, arrival_ind, STA_LTA_ind = np.intersect1d(times_arrival, times_STA_LTA, return_indices=True)

            intersection, arrival_ind, STA_LTA_ind = \
                np.intersect1d(times_arrival, times_STA_LTA, return_indices=True)
            exclusion = np.setdiff1d(times_arrival, intersection)
            _, exclusion, _ = np.intersect1d(times_arrival, exclusion, return_indices=True)
            
            ## Update time vector
            times_arrival = times_arrival[exclusion]
            
            result['TP'] += intersection.size
            result['FP'] += times_STA_LTA.size - intersection.size
            
    result['FN'] += times_arrival.size
    result['TN'] -= result['TP'] + result['FP'] + result['FN']
    
    return pd.DataFrame([result])
  
def STA_LTA_detector(waveform, times, true_arrival, duration, window, sampling, options, 
                     freq_min=0.01, freq_max=1., nb_points_STA=60., nb_points_LTA=500.,
                     STALTA_threshold_in=3., STALTA_threshold_out=0.5):

    """
    Compute STA/LTA cft coefficient for a given waveform
    and extract detections based on thresholds
    """
    
    
    ## Create obspy trace
    vTEC  = waveform['vTEC'].values
    times_TEC = waveform['time_s'].values
    #sampling = waveform.iloc[0]['sampling']
    #window = read_data.get_window(sampling, options['window'])
    window_TEC = times_TEC[-1] - times_TEC[0]
    i0, iend = 0, len(times_TEC)-1
    tr, _, _ = read_data.pre_process_waveform(times_TEC, vTEC, i0, iend, window_TEC, detrend=False, 
                                        bandpass=[freq_min, freq_max])
    
    ## Compute STA/LTA cft coefficient
    df  = tr.stats.sampling_rate
    cft = classic_sta_lta(tr.data, int(nb_points_STA * df), int(nb_points_LTA * df))
    
    ## Extract detected arrivals
    detections = recursively_find_wavetrains_STA_LTA(tr.times()+times[0], cft, 
                                                     STALTA_threshold_in, 
                                                     STALTA_threshold_out)
    """
    grouped_detections = detections.groupby('arrival_class')
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    axs[0].plot(times, vTEC)
    axs[1].plot(tr.times()+times[0], tr.data)
    axs[2].plot(tr.times()+times[0], cft)
    for group, detection in grouped_detections: axs[2].axvline(detection.time.min(), color='red'); axs[2].axvline(detection.time.max(), color='green');
    plt.show()
    """
    result = compute_metric_FTPN_detector(times, window, detections, true_arrival, duration)
    
    return detections, result
    
def process_batch_analytical(est, tec_data, tec_data_param, detection_parameters, thresholds, options, 
                             STALTA=True, anal=True, window=720, window_step=30., sampling=30.,
                             nb_pts_for_class=12, focus_arrival_shift=1000.):

    """
    Get all analytical and STA/LTA detections from a list of stations and satellites for a given event
    """


    ## If a detection file is provided, we load it instead or recomputing everything
    if 'detections_STA_LTA' in options['load']:
        STALTA = False
        detections_STA_LTA = pd.read_csv(options['load']['detections_STA_LTA'], header=[0], sep=',')
        results_LTA_STA = pd.read_csv(options['load']['results_STA_LTA'], header=[0], sep=',')
    
    if 'detections_AN' in options['load']:   
        anal = False
        detections_AN = pd.read_csv(options['load']['detections_AN'], header=[0], sep=',')
        results_AN = pd.read_csv(options['load']['results_AN'], header=[0], sep=',')
        
    ## If at least one method is requested, we loop over each event/station
    if anal or STALTA:
        name  = detection_parameters['name']
        events = detection_parameters['events']
        satellites = detection_parameters['satellites']
        stations   = detection_parameters['stations']
        time_end   = detection_parameters['time_end']
        
        ## Select available event/satellite/station from a list of requested event/satellite/station
        data_to_process = tec_data.loc[(tec_data['station'].isin(stations)) 
                            & (tec_data['satellite'].isin(satellites)) 
                            & (tec_data['event'].isin(events)) 
                            & (tec_data['time_s'] <= time_end), :]
                        
        grouped_data = data_to_process.groupby(['event', 'satellite', 'station'])
        detections_AN = pd.DataFrame()
        detections_STA_LTA = pd.DataFrame()
        results_LTA_STA = pd.DataFrame()
        results_anal    = pd.DataFrame()
        for group, waveform in grouped_data:
        
            event, satellite, station = group
            param = tec_data_param.loc[(tec_data_param['station'] == station) 
                            & (tec_data_param['satellite'] == satellite) 
                            & (tec_data_param['event'] == event), :]
                            
            waveform = tec_data.loc[(tec_data['event'] == event)
                                     & (tec_data['satellite'] == satellite)
                                     & (tec_data['station'] == station), :]
            
            t0, tend = waveform.time_s.min(), waveform.time_s.max()
            times = np.arange(t0, tend - window + sampling, sampling)
                            
            ## Find arrival time if there is one
            true_arrival = -1
            duration     = -1
            if param.size > 0:
                true_arrival = param.iloc[0]['arrival-time'] 
                duration     = options['signal_duration'][event]
        
            print(group)
            
            if STALTA:
                ## Loop over parameter set to optimize STA/LTA if needed
                for parameters_STA_LTA in options['parameters_STA_LTA']:
            
                    detection_STA_LTA, result = \
                        STA_LTA_detector(waveform, times, true_arrival, duration, window, sampling, options,
                                         freq_min=options['freq_min'], freq_max=options['freq_max'], 
                                         **parameters_STA_LTA)
                                         
                    detection_STA_LTA['true_arrival'] = true_arrival
                    detection_STA_LTA = detection_STA_LTA.assign(event = group[0])
                    detection_STA_LTA = detection_STA_LTA.assign(satellite = group[1])
                    detection_STA_LTA = detection_STA_LTA.assign(station = group[2])
                    for param in parameters_STA_LTA:
                        detection_STA_LTA[param] = parameters_STA_LTA[param]
                    detections_STA_LTA = detections_STA_LTA.append( detection_STA_LTA )
                    
                    result = result.assign(event = group[0])
                    result = result.assign(satellite = group[1])
                    result = result.assign(station = group[2])
                    for param in parameters_STA_LTA:
                        result[param] = parameters_STA_LTA[param]
                    results_LTA_STA = results_LTA_STA.append( result )
                
            if anal:
                #detections_AN = analytical_detector(waveform, thresholds)
                detections, result = analytical_detector(waveform, thresholds, window, true_arrival, duration, 
                                                   factor=0.7, nb_pts_for_class=nb_pts_for_class)
                detections['true_arrival'] = true_arrival
                detections = detections.assign(event = group[0])
                detections = detections.assign(satellite = group[1])
                detections = detections.assign(station = group[2])
                detections_AN = detections.append( detections )
                
                result = result.assign(event = group[0])
                result = result.assign(satellite = group[1])
                result = result.assign(station = group[2])
                result['nb_pts_for_class'] = nb_pts_for_class
                results_anal = results_anal.append( result )
        
        ## Save results to files    
        if STALTA:
            detections_STA_LTA.to_csv(options['DIR_FIGURES'] + 'detections_STA_LTA_'+name+'.csv', header=True, index=False)
            results_LTA_STA.to_csv(options['DIR_FIGURES'] + 'results_STA_LTA_'+name+'.csv', header=True, index=False)
        
        if anal:
            detections_AN.to_csv(options['DIR_FIGURES'] + 'detections_AN_'+name+'.csv', header=True, index=False)
            results_anal.to_csv(options['DIR_FIGURES'] + 'results_AN_'+name+'.csv', header=True, index=False)
        
    return detections_AN, results_AN, detections_STA_LTA, results_LTA_STA
    
def create_confusion(ax, precision_all, method, plot_labels=True, colorbar=False, normalize=True):

    """
    Compute and plot confusion matrix for a given method in a given ax
    """

    precision = precision_all.loc[precision_all['method'] == method, :]
    
    confusion = np.zeros((2,2))
    locs = {
        'TPR': (0, 0),
        'FPR': (1, 0),
        'FNR': (0, 1),
        'TNR': (1, 1),
    }
    
    for type in locs:
        confusion[locs[type]] = precision.loc[precision['Metric'] == type, 'vals']
    
    pd_confusion = pd.DataFrame(data=confusion, columns=['arrival', 'noise'], index=['arrival', 'noise'])
    
    sns.heatmap(pd_confusion, ax=ax, cbar=colorbar, vmin=0., vmax=1., annot=True, annot_kws={"size": 13})
    sc = ax.findobj(QuadMesh)[0]
    if plot_labels:
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    ax.set_title(method)
    
    return sc

def plot_summary_results(precision_methods, options, barplot=False, offset_label=0):

    """
    Plot performance results for different methods
    """

    ## Reshape dataframe for sns
    name_new_column = 'Metric'
    melted_precision_methods = precision_methods.melt('method', var_name=name_new_column, value_name='vals')
        
    errors_cols = ['mean_error', 'std_error']
    arrival_times = melted_precision_methods.loc[melted_precision_methods[name_new_column].isin(errors_cols), :]
    FTPN_cols = ['TP', 'TN', 'FP', 'FN']
    precision     = melted_precision_methods.loc[~(melted_precision_methods[name_new_column].isin(errors_cols) | (melted_precision_methods[name_new_column].isin(FTPN_cols))), :]
    labels_cols = precision_methods.loc[:, ~(precision_methods.columns.isin(errors_cols) \
                                        | (precision_methods.columns.isin(FTPN_cols)) \
                                        | (precision_methods.columns.isin(['method'])))].columns
    
    labels = precision_methods.method.values
    
    ## Plot figure
    if barplot:
    
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
        
        bplot = sns.barplot(data=precision, x="method", y='vals', hue=name_new_column, ax=axs[1], palette='flare')
        bplot.legend_.remove()
        #sns.catplot(x=name_new_column, y="vals",
        #            hue=name_new_column, col="method",
        #            data=precision, kind="bar",
        #            height=4, aspect=.7, ax=axs[1], palette='flare')
        axs[1].set_xlabel('Method')
        axs[1].set_ylabel('Score')
        axs[1].tick_params(axis='x', pad=25)
        
        ## Find method labels in right order
        xticks = axs[1].get_xticks()
        
        pos_labels = {}
        for ilabel, label in enumerate(labels):
            pos_labels[label] = xticks[ilabel]
            
        max_val = 0.325
        shift_ticks = np.linspace(-max_val, max_val, len(labels_cols))
        for icol, col in enumerate(labels_cols):
            for tick in xticks:
                axs[1].text(tick + shift_ticks[icol], -0.02, col, rotation=90., ha='center', va='top')
        
    else:
        
        fig = plt.figure(figsize=(10,3))
        grid = fig.add_gridspec(1, 3)
        axs = []
        
        """
        axs.append( fig.add_subplot(grid[0, :]) )
        axs[0].set_xlim([-0.5, 2.5])
        axs[0].grid()
        """
        alphabet = string.ascii_lowercase
        pos_labels = {}
        for i, method in zip(range(0, 3), labels):
            plot_labels = False
            colorbar    = False
            axs.append( fig.add_subplot(grid[0, i]) )
            if i == 0:
                plot_labels = True
            else:
                axs[i].set_yticklabels([])
            pos_labels[method] = i
            sc = create_confusion(axs[i], precision, method, plot_labels=plot_labels, colorbar=colorbar)
            axs[i].text(-0.1, 1.05, alphabet[i+offset_label]+')', ha='right', va='bottom', transform=axs[i].transAxes, 
                          bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=17., fontweight='bold')
            axs[i].tick_params(axis='both', which='both', labelsize=15.)
            axs[i].xaxis.get_label().set_fontsize(17)
            axs[i].yaxis.get_label().set_fontsize(17)
            
        """
        ## Add colorbar
        axins = inset_axes(axs[-1], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.1, 0., 1, 1.), bbox_transform=axs[-1].transAxes, borderpad=0)
        axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        cbar = plt.colorbar(sc, cax=axins, extend='both')
        cbar.ax.set_ylabel('Score', rotation=270, labelpad=16)
        """
        
    """
    ## Plot mean and std errors
    for igroup, (group, data) in enumerate(arrival_times.groupby('method')):
        points = data.loc[data[name_new_column] == 'mean_error', 'vals']
        err    = data.loc[data[name_new_column] == 'std_error', 'vals']
        #axs[0].errorbar(pos_labels[group], points, yerr=err, fmt='o', capsize=5, elinewidth=3.)
        bp()
        bplot = sns.boxplot(x="method", y="vals", hue=name_new_column, data=tips, linewidth=2.5, ax=axs[0], palette='flare')
        axs[0].text(-0.1, 1.05, 'a)', ha='right', va='bottom', transform=axs[0].transAxes, 
                    bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=15., fontweight='bold')
        
    axs[0].set_ylabel('Arrival-time picking\naccuracy (s)')
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    """
    
    #axs[0].xaxis.tick_top()
    #axs[0].xaxis.set_tick_params(labeltop=True)
    #labels = [item.get_text() for item in axs[1].get_xticklabels()]
    #axs[0].set_xticklabels(labels)
    
    fig.align_ylabels(axs)
    fig.subplots_adjust(bottom=0.2, right=0.95)
    
    ## Save figure
    fig.savefig(options['DIR_FIGURES'] + 'summary_comparison_methods.pdf')
    
def update_arrival_times_with_overlap(x, begin_time, shift_time, window):
    
    """
    Shift arrival time by shift_time which correspond to the part of the window that is new compared to the previous one
    If this is the first window, we do not do it because no previous window
    """
    
    if abs(x['time'] - begin_time) > window/2.:
        x['time'] += shift_time ## We pick the arrival time as the beginning of the new non-overlapped window
    else:
        x['time'] += 0.1*window
    
    return x
    
def compute_error_analytical_RF(RF_detections, RF_probas, est, data, tec_data, tec_data_param, 
                                detection_parameters, thresholds, options, 
                                threshold_TP=500., plot_individual_waveforms=False,
                                window=720., sampling=30., nb_pts_for_class=12,
                                STALTA=False, anal=True):
    
    """
    Compare arrival detection score between RF and analytical model across a given observation dataset
    """
    
    #window     = 500. ## TODO: find window automatically
    shift_time = window*(1. - 1./options['factor_overlap'] )
    
    detections_AN, results_AN, detections_STA_LTA, results_LTA_STA = \
        process_batch_analytical(est, tec_data, tec_data_param, detection_parameters, thresholds, options,
                                 STALTA=STALTA, anal=anal, window=window, window_step=window-shift_time, sampling=sampling,
                                 nb_pts_for_class=nb_pts_for_class)
    
    
    """
    detect = LTA_STA_detections.loc[(LTA_STA_detections.arrival_class > -1) & (LTA_STA_detections['true_arrival'] > -1), :]
    detect['error'] = detect['true_arrival'] - detect['time']
    detect['error-abs'] = abs(detect['error'])
    detect.reset_index(drop=True, inplace=True)
    detect_best = detect.loc[detect['error-abs'] < 200., :]
    
    TP_STA_LTA = detect_best.loc[detect_best.groupby(['event', 'satellite', 'station'])['error-abs'].idxmin()].reset_index()
    TP_STA_LTA = TP_STA_LTA.set_index('index')
    
    FP_STA_LTA = detect.loc[~detect.index.isin(TP_STA_LTA.index), :].groupby(['event', 'satellite', 'station', 'arrival_class']).first().reset_index(drop=True)
    
    events, satellites, stations = TP_STA_LTA.event.unique().tolist(), TP_STA_LTA.satellite.unique().tolist(), TP_STA_LTA.station.unique().tolist()
    FN_STA_LTA = tec_data_param.loc[(tec_data_param.event.isin(events)) & (tec_data_param.satellite.isin(satellites)) & (tec_data_param.station.isin(stations)), :]
    """
    ## Either load or recompute performance results
    if 'summary_results' in options['load']:
    
        ## Load results from file
        summary_results = pd.read_csv(options['load']['summary_results'], header=[0])
        precision_methods = pd.read_csv(options['load']['precision_methods'], header=[0])
    
    else:
    
        grouped_data  = RF_detections.groupby(['event', 'satellite', 'station'])
        
        ## Find RF test dataset        
        data_test = data.loc[~data.extra_test & (data.type=='test')]
        list_stations = data_test[['event', 'satellite', 'station']].values.tolist()
        
        summary_results = pd.DataFrame()
        for group, RF_local_detection in grouped_data:
            
            event, satellite, station = group
            
            ## Skip waveforms not in testing dataset
            if not [event, satellite, station] in list_stations:
                continue
                
            ## TO REMOVE
            #if not (station == '0181' and event=='Tohoku_1s'):
            #    continue
                
            try:
                station = "{:04d}".format(station)
            except:
                pass
            print(group)
            
            waveform = tec_data.loc[(tec_data['event'] == event)
                                     & (tec_data['satellite'] == satellite)
                                     & (tec_data['station'] == station), :]
            param    = tec_data_param.loc[(tec_data_param['event'] == event)
                                        & (tec_data_param['satellite'] == satellite)
                                        & (tec_data_param['station'] == station), :]
            
            probas = RF_probas.loc[(RF_probas['event'] == event)
                                & (RF_probas['satellite'] == satellite)
                                & (RF_probas['station'] == station), :]
            
            #probas = RF_probas.loc[(RF_probas['event'] == event)& (RF_probas['satellite'] == satellite)& (RF_probas['station'] == station), :]
            #RF_detection = RF_detections.loc[(RF_detections['event'] == event)& (RF_detections['satellite'] == satellite)& (RF_detections['station'] == station), :]
            
            result_AN = results_AN.loc[(results_AN['event'] == event)
                                & (results_AN['satellite'] == satellite)
                                & (results_AN['station'] == station), :].iloc[0]
            result_LTA_STA = results_LTA_STA.loc[(results_LTA_STA['event'] == event)
                                & (results_LTA_STA['satellite'] == satellite)
                                & (results_LTA_STA['station'] == station), :].iloc[0]
            
            begin_time = waveform['time_s'].min()
            there_is_arrival = False
            result = {
                'arrival-time': -1.,
                'event': event,
                'satellite': satellite,
                'station': station,
                'RF-TP': 0,
                'RF-FP': 0,
                'RF-FN': 0,
                'RF-TN': 0,
                'RF-error': -1,
                'AN-TP': 0,
                'AN-FP': 0,
                'AN-FN': 0,
                'AN-TN': 0,
                'AN-error': -1,
                'STA-TP': 0,
                'STA-FP': 0,
                'STA-FN': 0,
                'STA-TN': 0,
                'STA-error': -1,
            }
            
            """
            if param.size > 0:
                there_is_arrival = True
                param = param.iloc[0]
                result['arrival-time'] = param['arrival-time']
            """
            
            ## Find RF detections
            RF_detection = RF_detections.loc[(RF_detections['event'] == event) & (RF_detections['satellite'] == satellite) & (RF_detections['station'] == station), :]
            
            ## Waveform time samples
            t0, tend = waveform.time_s.min(), waveform.time_s.max()
            times = np.arange(t0, tend - window + sampling, sampling)
                            
            ## Find arrival time if there is one
            true_arrival = -1
            duration     = -1
            if param.size > 0:
                true_arrival = param.iloc[0]['arrival-time'] 
                result['arrival-time'] = true_arrival
                duration     = options['signal_duration'][event]
              
            result_RF = compute_metric_FTPN_detector(times, window, RF_detection, true_arrival, duration)
            result_RF = result_RF.iloc[0]
              
            """
            ## Compute number of true and false positives for RF
            #RF_local_detection = RF_detection.detection.iloc[0]
            if RF_local_detection.size > 0:
                RF_time_detections = RF_local_detection.groupby('arrival_class')[['time']].min().reset_index()
                RF_time_detections = \
                    RF_time_detections.apply(update_arrival_times_with_overlap, args=[begin_time, shift_time, window], axis=1)
                if there_is_arrival:
                    right_arrivals  = RF_time_detections.loc[abs(RF_time_detections['time'] - param['arrival-time']) < threshold_TP, :]
                    count = 0
                    if right_arrivals.size > 0:
                        count = 1
                        error = (right_arrivals.iloc[0]['time'] - param['arrival-time'])
                        #if error > 400:
                        #    bp()
                        result['RF-error'] = error
                        result['RF-TP'] += count
                    result['RF-FP'] += RF_time_detections.shape[0] - count
                else:
                    result['RF-FP'] += RF_time_detections.shape[0]
            elif there_is_arrival:
                result['RF-FN'] += 1
            else:
                result['RF-TN'] += 1
            """
            
            ## Add results all methods
            columns_to_copy = ['TP', 'FP', 'TN', 'FN']
            
            name_method = 'RF'
            for column in columns_to_copy:
                result[name_method + '-' + column] = result_RF[column]
            
            name_method = 'AN'
            for column in columns_to_copy:
                result[name_method + '-' + column] = result_AN[column]
            
            name_method = 'STA'
            for column in columns_to_copy:
                result[name_method + '-' + column] = result_LTA_STA[column]
            
            summary_results = summary_results.append( [result] )
            
            ## Plot results
            #probas     = RF_detection.probas.iloc[0]
            """
            if plot_individual_waveforms:
                train_est.plot_processed_timeseries(event, satellite, station, waveform, probas, 
                                                RF_local_detection, window, options, 
                                                true_arrival=result['arrival-time'], 
                                                analytical_detections=AN_detection, 
                                                sta_lta_detections=LTA_STA_detection_all,
                                                figsize=(10,4), add_label='e)')
            """
            
            """
            import test_module
            from importlib import reload 
            train_est.plot_processed_timeseries(event, satellite, station, waveform, probas, RF_local_detection, window, options, true_arrival=result['arrival-time'], analytical_detections=AN_detection, sta_lta_detections=LTA_STA_detection)
            """
        
        #val = 'RF'
        #summary_results[val+'-TP']+summary_results[val+'-TN']+summary_results[val+'-FP']+summary_results[val+'-FN']
        #summary_results.groupby('event', 'satellite').sum()
        
        methods = ['RF', 'AN', 'STA']
        precision_methods = pd.DataFrame()
        for method in methods:
            
            precision = {}
            precision['method'] = method
            precision['mean_error'] = summary_results.loc[summary_results[method+'-error'] > -1, method+'-error'].mean()
            precision['std_error']  = summary_results.loc[summary_results[method+'-error'] > -1, method+'-error'].std()
            precision['FP'] = summary_results[method+'-FP'].sum()
            precision['TP'] = summary_results[method+'-TP'].sum()
            precision['FN'] = summary_results[method+'-FN'].sum()
            precision['TN'] = summary_results[method+'-TN'].sum()
            precision['TPR'] = precision['TP'] / (precision['TP']+precision['FN'])
            precision['TNR'] = precision['TN'] / (precision['TN']+precision['FP'])
            precision['FPR'] = precision['FP'] / (precision['FP']+precision['TN'])
            precision['FNR'] = precision['FN'] / (precision['FN']+precision['TP'])
            precision['PPV'] = precision['TP'] / (precision['TP']+precision['FP'])
            
            precision_methods = precision_methods.append( [precision] )
        
        precision_methods.reset_index(inplace=True, drop=True)
        bp()
        ## Save results to file
        summary_results.to_csv(options['DIR_FIGURES'] + 'summary_results_new.csv', header=True, index=False)
        precision_methods.to_csv(options['DIR_FIGURES'] + 'precision_methods_new.csv', header=True, index=False)
    
    ## Plot results
    plot_summary_results(precision_methods, options)
    
    bp()
      
def get_shuffled_inputs(features_pd, nb_combi = 7):
    
    """
    Get a list of combinations of nb_combi input features
    """
    
    all_columns = features_pd.columns
    list_available_parameters = [list(item) for item in itertools.combinations(all_columns, r=nb_combi)]
    
    return list_available_parameters
  
def convert_report_to_dataframe(report):
    
    """
    Convert classification report from sklearn to dataframe
    """
    
    report_df = {}
    for class_ in report:
        if isinstance(report[class_], dict):
            for metric in report[class_]:
                report_df[class_ + '-' + metric] = report[class_][metric]
        else:
            report_df[class_] = report[class_]
        
    report_df = pd.DataFrame([report_df])

    return report_df
  
def train_estimators_with_shuffled_inputs(features_pd, tec_data_param, options, shuffle_input=False,
                                          priority_only_noise_waveforms=True, two_steps=True,
                                          exclude_events_from_training=[], split=0.8, split_by_event=False,
                                          max_proporption_one_class=0.3):
    
    """
    Compute features and train one estimator
    """
    
    columns_to_remove = []
    if shuffle_input:
        list_available_parameters = get_shuffled_inputs(features_pd, nb_combi = 7)
        report = pd.DataFrame()
        for columns_to_remove in list_available_parameters:
            bp()
            est, data = train_est.train_machine(features_pd, tec_data_param, options, columns_to_remove=columns_to_remove,
                                                plot_performance=False, plot_best_features=False,
                                                priority_only_noise_waveforms=priority_only_noise_waveforms, two_steps=two_steps,
                                                exclude_events_from_training=exclude_events_from_training, split=split,
                                                split_by_event=split_by_event, max_proporption_one_class=max_proporption_one_class)
            report_dict = train_est.compute_performance_one_model(est, data)
            report      = report.append( convert_report_to_dataframe(report_dict) )
            
    else:
        est, data = train_est.train_machine(features_pd, tec_data_param, options, 
                                            columns_to_remove=columns_to_remove,
                                            plot_performance=False, plot_best_features=False,
                                            priority_only_noise_waveforms=priority_only_noise_waveforms, 
                                            two_steps=two_steps, exclude_events_from_training=exclude_events_from_training, 
                                            split=split, split_by_event=split_by_event, max_proporption_one_class=max_proporption_one_class)
        
        #data_balanced = data.loc[~data.extra_test]
        data_balanced = data.loc[~data.event_to_testing & ~data.extra_test, :]
        
        report_dict = train_est.compute_performance_one_model(est, data_balanced)
        report = convert_report_to_dataframe(report_dict)
        
    return report, est, data
  
def recreate_options_from_list(item, optimization, options, load_features=True, load_est=True, load_data=True):
    
    """
    Create standard options dictionnary to compute features for a given combinaison in item
    """
    
    options_new = copy.deepcopy(options)
    for one_item, name in zip(item, optimization.keys()):
        options_new[name] = one_item
        
    if load_est:
        forest_template = 'forest_est_s{shift_detection_after_max}_m{min_overlap_label}_n{noise_pick_shift}_w{window:d}.pkl'
        window = [options_new['window'][key] for key in options_new['window'].keys()][0]
        current_est = {
            'shift_detection_after_max': options_new['shift_detection_after_max'], # factor (x diff-t) 
            'min_overlap_label': options_new['min_overlap_label'], # minimum overlap between perturbed waveforms extracted from arrival
            'noise_pick_shift': options_new['noise_pick_shift'], # Delay from arrival time to pick noise
            'window': window # window size
        }
        options_new['load']['est'] = options['DIR_FIGURES'] + forest_template.format(**current_est)
    
        ## Check if file exists
        if not os.path.isfile(options_new['load']['est']):
            sys.exit('Estimator file does not exist:', options['load']['est'])
    
    if load_data:
        data_template = 'data_s{shift_detection_after_max}_m{min_overlap_label}_n{noise_pick_shift}_w{window:d}.pkl'
        window = [options_new['window'][key] for key in options_new['window'].keys()][0]
        current_est = {
            'shift_detection_after_max': options_new['shift_detection_after_max'], # factor (x diff-t) 
            'min_overlap_label': options_new['min_overlap_label'], # minimum overlap between perturbed waveforms extracted from arrival
            'noise_pick_shift': options_new['noise_pick_shift'], # Delay from arrival time to pick noise
            'window': window # window size
        }
        options_new['load']['data'] = options['DIR_FIGURES'] + data_template.format(**current_est)
    
        ## Check if file exists
        if not os.path.isfile(options_new['load']['data']):
            sys.exit('Data file does not exist:', options['load']['est'])
    
    if load_features:
        features_template = 'features_features_m{min_overlap_label}_w{window:d}.csv'
        window = [options_new['window'][key] for key in options_new['window'].keys()][0]
        current_feature = {
            'min_overlap_label': options_new['min_overlap_label'], # minimum overlap between perturbed waveforms extracted from arrival
            'window': window # window size
        }
        options_new['load']['features'] = options['DIR_DATA'] + features_template.format(**current_feature)
       
        ## Check if file exists
        if not os.path.isfile(options_new['load']['features']):
            sys.exit('Feature file does not exist:', options['load']['features'])
        
    #print(options_new['load']['features'], options_new['load']['est'])
        
    return options_new
  
def build_option_list(optimization, options, load_features=True, load_est=True, load_data=True):

    """
    Create list of standard options dictionnary to compute features for all parameter combinations in optimization
    """
    
    list_available_parameters = []
    for key in optimization:
        if key == 'shuffle_inputs' or key == 'name':
            continue
        list_available_parameters.append( optimization[key] )
  
    nb_items = len([item for item in itertools.product(*list_available_parameters)])
    
    ## Deep copy are critical since we are dealing with dicts of dicts
    options_list = []
    for iitem, item in enumerate(itertools.product(*list_available_parameters)):
        local_list = recreate_options_from_list(item, optimization, options, 
                                        load_features=load_features, load_est=load_est, load_data=load_data) 
        options_list.append( copy.deepcopy(local_list) )
        
    return options_list

def plot_metrics_windows_overlap(reports, options_list, ax=None):

    """
    Plot metrics for a various estimators trained with inputs using different window sizes
    """
    
    ## Extract metrics and corresponding windows parameters
    #sampling = [sampling for sampling in options_list[0]['window']][0]
    #min_overlap_label  = [options_['min_overlap_label'] for options_ in options_list]
    #reports['min_overlap_label'] = min_overlap_label
    metrics_to_plot = reports[['noise-precision', 'arrival-precision', 'noise-recall', 'arrival-recall', 'min_overlap_label']]
    #metrics_to_plot = reports[['noise-precision', 'macro avg-f1-score', 'arrival-recall', 'min_overlap_label']]
    columns = [column.replace('-', '\n').replace('precision', 'prec.') for column in metrics_to_plot.columns]
    columns[-1] = 'Overlap (%)'
    metrics_to_plot.columns = columns
    metrics_to_plot = metrics_to_plot.groupby('Overlap (%)').median()
    #metrics_to_plot   = metrics_to_plot.set_index('Overlap (%)')
    
    ## Plot results
    new_figure = False
    if ax == None:
        new_figure = True
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        #fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.1)
    
    metric_plot = sns.heatmap(metrics_to_plot, annot=True, cbar=False, ax=ax)
    metric_plot.set_xticklabels(metric_plot.get_xticklabels(), rotation = 0)
    
    ## Save Figure
    if new_figure:
        options = options_list[0]
        fig.savefig(options['DIR_FIGURES'] + 'metrics_classification_different_overlaps.pdf')
  
def plot_metrics_windows(reports_in, options_list, ax=None):

    """
    Plot metrics for a various estimators trained with inputs using different window sizes
    """
    
    print('111111')

    ## Extract metrics and corresponding windows parameters
    sampling = [sampling for sampling in options_list[0]['window']][0]
    
    windows  = [options_['window'][sampling] for options_ in options_list]
    reports = reports_in.loc[reports_in.window.isin(windows)]
    print(windows, reports)
    reports['window'] = windows
    print('0000')
    metrics_to_plot   = reports[['noise-precision', 'arrival-precision', 'noise-recall', 'arrival-recall','window']]
    print('2222')
    columns = [column.replace('-', '\n').replace('precision', 'prec.') for column in metrics_to_plot.columns]
    columns[-1] = 'window (s)'
    metrics_to_plot.columns = columns
    metrics_to_plot = metrics_to_plot.groupby('window (s)').median()
    #metrics_to_plot   = metrics_to_plot.set_index('window (s)')
    
    ## Plot results
    new_figure = False
    if ax == None:
        new_figure = True
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        #fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.1)
    
    print('klklkl')
    
    metric_plot = sns.heatmap(metrics_to_plot, annot=True, cbar=False, ax=ax)
    metric_plot.set_xticklabels(metric_plot.get_xticklabels(), rotation = 0)
    
    ## Save Figure
    if new_figure:
        options = options_list[0]
        fig.savefig(options['DIR_FIGURES'] + 'metrics_classification_different_windows.pdf')
  
def wrapper_optimization_estimator(tec_data, tec_data_param, options, optimization={}, 
                                   priority_only_noise_waveforms=True, 
                                   load_features=True, load_est=True, load_data=True, two_steps=True,
                                   exclude_events_from_training=[], split=0.8, split_by_event=False,
                                   max_proporption_one_class=0.3, nb_CPU=1):
    
    """
    Compute features and train estimators based on a list of parameters to optimize
    """
    
    features_pd, est, data = None, None, None
    if optimization:
        
        shuffle_input = False
        if 'shuffle_inputs' in optimization.keys():
            if optimization['shuffle_inputs']:
                shuffle_input = True
    
        reports = pd.DataFrame()
        all_features = pd.DataFrame()
        ## Loop over each option list for feature extraction
        options_list = build_option_list(optimization, options, load_features=load_features, load_est=load_est, load_data=load_data)
        
        for ioption, options_ in enumerate(options_list):
        
            print('Option list', ioption)
            
            features_pd = read_data.compute_features_all_waveforms(tec_data, tec_data_param, options_, type=options_['type_input'], nb_CPU=nb_CPU)
            
            ## Loop over each option list for input feature choice
            report, est, data = train_estimators_with_shuffled_inputs(features_pd, tec_data_param, options_, shuffle_input=False,
                                                           priority_only_noise_waveforms=priority_only_noise_waveforms,
                                                           two_steps=two_steps, exclude_events_from_training=exclude_events_from_training,
                                                           split=split, split_by_event=split_by_event, max_proporption_one_class=max_proporption_one_class)
            report['no'] = ioption
            report['window']  = options_['window'][40.]
            report['min_overlap_label'] = options_['min_overlap_label']
            reports = reports.append( report )
        
            features_pd['no'] = ioption
            all_features = all_features.append( features_pd )
            
        reports.to_csv(options['DIR_FIGURES'] + 'reports_'+optimization['name']+'.csv', header=True, index=False)
        if 'window' in optimization.keys():
            plot_metrics_windows(reports, options_list)
        
        if 'min_overlap_label' in optimization.keys():
            plot_metrics_windows_overlap(reports, options_list)
        
        bp()
        
    else:
        features_pd = read_data.compute_features_all_waveforms(tec_data, tec_data_param, options, type=options['type_input'], nb_CPU=nb_CPU)
        est, data = train_est.train_machine(features_pd, tec_data_param, options, columns_to_remove=[],
                                            priority_only_noise_waveforms=priority_only_noise_waveforms,
                                            two_steps=two_steps, exclude_events_from_training=exclude_events_from_training, 
                                            split=split, split_by_event=split_by_event, max_proporption_one_class=max_proporption_one_class)
    
    return features_pd, est, data
    
def get_ampmax_and_tmax(detection, waveform, options, standard_sampling=30., offset_find_max=1):
    
    """
    Locate vTEC maximum over a given detection group
    """
    
    sampling = waveform.iloc[0]['sampling']
    window = read_data.get_window(sampling, options['window'])
    
    grouped_detection = detection.groupby('arrival_class')
    for arrival_class, one_detection in grouped_detection:
    
        one_waveform = waveform.loc[(waveform.time_s >= one_detection.time.min())
                                    & (waveform.time_s <= one_detection.time.max() + window)]
    
        ## Preprocess waveform (derivation + bandpassing)
        times  = one_waveform['time_s'].values[:]
        full_window = times[-1]
        i0, iend = 0, times.size-1
        tr, _, _ = read_data.pre_process_waveform(times, one_waveform['vTEC'].values, 
                                                  i0, iend, full_window, detrend=False, 
                                                  bandpass=[options['freq_min'], options['freq_max']],
                                                  standard_sampling=standard_sampling)
       
        ## Locate maximum
        imax = abs(tr.data).argmax()
        first_ind = max(0, imax - offset_find_max)
        imax = first_ind + one_waveform.vTEC.values[first_ind:imax+offset_find_max+1].argmax()
        tmax = one_waveform.time_s.iloc[imax]
        ampmax = one_waveform.vTEC.iloc[imax]
        
        ## Update detection list
        detection.loc[detection.index.isin(one_detection.index), 'time_max'] = tmax
        detection.loc[detection.index.isin(one_detection.index), 'amp_max'] = ampmax
    
        #plt.plot(one_waveform.time_s, one_waveform.vTEC); plt.axvline(tmax, color='red'); plt.show()
        
def compute_arrival_time_one_waveform(est, station, satellite, event, input_columns,
                                      waveform, tec_data_param, options,
                                      focus_on_arrival=False, focus_arrival_shift = 600., 
                                      est_picker=None, use_STA_LTA_for_picking=False,
                                      return_all_waveforms_used=False,
                                      plot_probas=False, stop_at_each_iter=False,
                                      nb_picks=5, activate_LTA_STA=False, 
                                      time_STA=60., time_LTA=300,
                                      STA_LTA_threshold_in=2., STA_LTA_threshold_out=0.1,
                                      add_label='', adaptative_sampling=False, standard_sampling=30.,
                                      standard_sampling_for_picker=30., add_inset=False, find_maximum=True):
    
    """
    Determine arrival time from a list of RF detections over one waveform, i.e., one station
    """
    
    true_arrival = -1
    params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                    & (tec_data_param['satellite'] == satellite) 
                    & (tec_data_param['event'] == event), : ]
                    
    if params.size > 0:
        params = params.iloc[0]
        true_arrival = params['arrival-time']
    
    ## If we are focusing on true arrivals only, we remove all times "far awat" from arrival time
    if focus_on_arrival:
        waveform = waveform.loc[(waveform['time_s'] >= true_arrival-focus_arrival_shift)
                                & (waveform['time_s'] <= true_arrival+focus_arrival_shift), :]
    
    standard_sampling_local = standard_sampling
    if adaptative_sampling:
        standard_sampling_local = np.round(waveform.time_s.iloc[1] - waveform.time_s.iloc[0])
    
    time_end = waveform.time_s.max()
    #waveform_param = pd.DataFrame()
    result = \
        train_est.process_timeseries_with_forest(time_end, est, waveform, params, 
                                                 event, satellite, station, 
                                                 input_columns, options, plot_probas=plot_probas, 
                                                 type='features', est_picker=est_picker,
                                                 use_STA_LTA_for_picking=use_STA_LTA_for_picking,
                                                 return_all_waveforms_used=return_all_waveforms_used,
                                                 nb_picks=nb_picks, activate_LTA_STA=activate_LTA_STA, 
                                                 time_STA=time_STA, time_LTA=time_LTA,
                                                 STA_LTA_threshold_in=STA_LTA_threshold_in, 
                                                 STA_LTA_threshold_out=STA_LTA_threshold_out,
                                                 add_label=add_label, 
                                                 standard_sampling=standard_sampling_local,
                                                 standard_sampling_for_picker=standard_sampling_for_picker,
                                                 adaptative_sampling=adaptative_sampling,
                                                 add_inset=add_inset)

    if stop_at_each_iter:
        bp()
    
    detection = result['detections']
    detection['event'] = event
    detection['satellite'] = satellite
    detection['station'] = station
    detection['true-arrival-time'] = true_arrival
    
    proba = result['probas']
    proba['event'] = event
    proba['satellite'] = satellite
    proba['station'] = station
    
    ## Find maximum
    if find_maximum:
        detection['time_max'] = -1
        detection['amp_max']  = 0.
        if detection.shape[0] > 0:
            get_ampmax_and_tmax(detection, waveform, options, standard_sampling=30., offset_find_max=1)
    
    all_waveforms_used = pd.DataFrame()
    if return_all_waveforms_used:
        all_waveforms_used = result['all_waveforms_used']
        all_waveforms_used['event'] = event
        all_waveforms_used['satellite'] = satellite
        all_waveforms_used['station'] = station
        all_waveforms_used['true_arrival'] = true_arrival
        
    return detection, proba, all_waveforms_used
    
def compute_arrival_times_one_station(est, input_columns, tec_data_param, options, 
                                  focus_on_arrival, focus_arrival_shift, est_picker, use_STA_LTA_for_picking,
                                  return_all_waveforms_used, plot_probas, stop_at_each_iter,
                                  nb_picks, activate_LTA_STA, time_STA, time_LTA, STA_LTA_threshold_in, 
                                  STA_LTA_threshold_out, add_label, adaptative_sampling, standard_sampling,
                                  standard_sampling_for_picker, data_to_process, add_inset, idx_to_process):
    
    """
    Compute arrival times for a list of detections
    """
    
    #grouped_data = data_to_process.groupby(['event', 'satellite', 'station'])
    detections = pd.DataFrame()
    probas     = pd.DataFrame()
    all_waveforms_used = pd.DataFrame()
    for iwaveform in range(idx_to_process.shape[0]):
    
        event, satellite, station = idx_to_process[iwaveform, :]
        
        waveform = data_to_process.loc[ (data_to_process['station'] == station) 
                & (data_to_process['satellite'] == satellite) 
                & (data_to_process['event'] == event), : ]
    
        params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                & (tec_data_param['satellite'] == satellite) 
                & (tec_data_param['event'] == event), : ]
                
        ## If we are just trying to compute arrival times for true arrivals, we skip noise waveforms
        if params.size == 0 and focus_on_arrival:
            continue
            
        detection, proba, all_waveforms_used_one_station = \
            compute_arrival_time_one_waveform(est, station, satellite, event, input_columns,
                                              waveform, tec_data_param, options, 
                                              focus_on_arrival=focus_on_arrival,
                                              focus_arrival_shift=focus_arrival_shift,
                                              est_picker=est_picker, use_STA_LTA_for_picking=use_STA_LTA_for_picking,
                                              return_all_waveforms_used=return_all_waveforms_used,
                                              plot_probas=plot_probas, stop_at_each_iter=stop_at_each_iter,
                                              nb_picks=nb_picks, activate_LTA_STA=activate_LTA_STA, 
                                              time_STA=time_STA, time_LTA=time_LTA,
                                              STA_LTA_threshold_in=STA_LTA_threshold_in, 
                                              STA_LTA_threshold_out=STA_LTA_threshold_out,
                                              add_label=add_label, 
                                              adaptative_sampling=adaptative_sampling, 
                                              standard_sampling=standard_sampling,
                                              standard_sampling_for_picker=standard_sampling_for_picker,
                                              add_inset=add_inset)

        detections = detections.append( detection )
        probas = probas.append( proba )
        if return_all_waveforms_used:
            all_waveforms_used = all_waveforms_used.append( all_waveforms_used_one_station )
    
    return detections, probas, all_waveforms_used
    
def compute_arrival_times_RF(est, data, tec_data, tec_data_param, detection_parameters, 
                             options, focus_on_arrival=False, focus_arrival_shift=600.,
                             adaptative_sampling=False, standard_sampling=30.,
                             standard_sampling_for_picker=30.,
                             use_STA_LTA_for_picking=False,
                             est_picker=None, return_all_waveforms_used=False,
                             plot_probas=False, stop_at_each_iter=False, nb_picks=5, 
                             activate_LTA_STA=False, time_STA=60., time_LTA=300,
                             STA_LTA_threshold_in=2., STA_LTA_threshold_out=0.1,
                             add_label='', nb_CPU=16, add_inset=False):

    """
    Compute arrival times for a list of events/satellites/stations
    """

    all_waveforms_used = pd.DataFrame()
    if 'detections' in options['load']:
        
        detections = pd.read_csv(options['load']['detections'], sep=',', header=[0])
        probas = pd.read_csv(options['load']['probas'], sep=',', header=[0])
        if return_all_waveforms_used:
            all_waveforms_used = pd.read_csv(options['load']['waveforms_used'], sep=',', header=[0])
        
    else:

        name   = detection_parameters['name']
        events = detection_parameters['events']
        satellites = detection_parameters['satellites']
        stations   = detection_parameters['stations']

        data_to_process = \
            tec_data.loc[(tec_data.event.isin(events)) 
                        & (tec_data.satellite.isin(satellites))
                        & (tec_data.station.isin(stations)), :]

        input_columns = [key for key in train_est.data_without_info_columns(data)]
        
        ## Only select new entries
        available_simulations = data_to_process.groupby(['event', 'satellite', 'station']).first()\
                                    .reset_index()[['event', 'satellite', 'station']].values
        nb_simulations = available_simulations.shape[0]
        compute_arrival_times_one_station_partial = \
            partial(compute_arrival_times_one_station, est, input_columns, tec_data_param, options, 
                    focus_on_arrival, focus_arrival_shift, est_picker, use_STA_LTA_for_picking,
                    return_all_waveforms_used, plot_probas, stop_at_each_iter,
                    nb_picks, activate_LTA_STA, time_STA, time_LTA, STA_LTA_threshold_in, 
                    STA_LTA_threshold_out, add_label, adaptative_sampling, standard_sampling,
                    standard_sampling_for_picker, data_to_process, add_inset)

        N = min(nb_CPU, nb_simulations)
        ## If one CPU requested, no need for deployment
        if N == 1:
            detections, probas, all_waveforms_used = \
                compute_arrival_times_one_station_partial(available_simulations)

        ## Otherwise, we pool the processes
        else:
            
            step_idx =  nb_simulations//N
            list_of_lists = []
            for i in range(N):
                idx = np.arange(i*step_idx, (i+1)*step_idx)
                if i == N-1:
                    idx = np.arange(i*step_idx, nb_simulations)
                list_of_lists.append( available_simulations[idx, :] )
            
            with get_context("spawn").Pool(processes = N) as p:
                results = p.map(compute_arrival_times_one_station_partial, list_of_lists)

            detections = pd.DataFrame()
            probas     = pd.DataFrame()
            all_waveforms_used = pd.DataFrame()
            for result in results:
                detections = detections.append( result[0] )
                probas     = probas.append( result[1] )
                all_waveforms_used = all_waveforms_used.append( result[2] )
            detections.reset_index(drop=True, inplace=True)
            probas.reset_index(drop=True, inplace=True)
            all_waveforms_used.reset_index(drop=True, inplace=True)
        
        """
        grouped_data = data_to_process.groupby(['event', 'satellite', 'station'])
        detections = pd.DataFrame()
        probas     = pd.DataFrame()
        for group, waveform in grouped_data:
            
            print(group)
            
            event, satellite, station = group
            params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                    & (tec_data_param['satellite'] == satellite) 
                    & (tec_data_param['event'] == event), : ]
                    
            ## If we are just trying to compute arrival times for true arrivals, we skip noise waveforms
            if params.size == 0 and focus_on_arrival:
                continue
                
            detection, proba, all_waveforms_used_one_station = \
                compute_arrival_time_one_waveform(est, station, satellite, event, input_columns,
                                                  waveform, tec_data_param, options, 
                                                  focus_on_arrival=focus_on_arrival,
                                                  focus_arrival_shift=focus_arrival_shift,
                                                  est_picker=est_picker, use_STA_LTA_for_picking=use_STA_LTA_for_picking,
                                                  return_all_waveforms_used=return_all_waveforms_used,
                                                  plot_probas=plot_probas, stop_at_each_iter=stop_at_each_iter,
                                                  nb_picks=nb_picks, activate_LTA_STA=activate_LTA_STA, 
                                                  time_STA=time_STA, time_LTA=time_LTA,
                                                  STA_LTA_threshold_in=STA_LTA_threshold_in, 
                                                  STA_LTA_threshold_out=STA_LTA_threshold_out,
                                                  add_label=add_label, 
                                                  adaptative_sampling=adaptative_sampling, 
                                                  standard_sampling=standard_sampling,
                                                  standard_sampling_for_picker=standard_sampling_for_picker)
        
            detections = detections.append( detection )
            probas = probas.append( proba )
            if return_all_waveforms_used:
                all_waveforms_used = all_waveforms_used.append( all_waveforms_used_one_station )
        """
        
        #test = detections.groupby(['event', 'satellite', 'station', 'arrival_class']).first().reset_index()
        #test['error'] = test['true-arrival-time'] - test['time']
        cmp = ''
        if focus_on_arrival:
            cmp = '_arrivalsonly'
            
        detections.to_csv(options['DIR_FIGURES'] + 'detected_arrivals_'+name+cmp+'.csv', header=True, index=False)
        probas.to_csv(options['DIR_FIGURES'] + 'probas_all_waveforms_'+name+cmp+'.csv', header=True, index=False)
        if return_all_waveforms_used:
            all_waveforms_used.to_csv(options['DIR_FIGURES'] + 'waveforms_used_'+name+cmp+'.csv', header=True, index=False)
    
    return detections, probas, all_waveforms_used
    
"""
def measure_sensitivity_nb_points_empirical(detections, probas, l_nb_for_class, l_nb_for_end_class, options, window=720.):

    grouped_probas = probas.groupby(['event', 'satellite', 'station'])
    
    detections_all = pd.DataFrame()
    for nb_for_class in l_nb_for_class:
        for nb_for_end_class in l_nb_for_end_class:
            for group, proba in grouped_probas:
                event, satellite, station = group
                detection = compute_arrival_time(proba, window, nb_for_class=nb_for_class, 
                                                  nb_for_end_class=nb_for_end_class)
                detection['event'] = event                                 
                detection['satellite'] = satellite
                detection['station'] = station
                detection['nb_for_class'] = nb_for_class
                detection['nb_for_end_class'] = nb_for_end_class
                
                detections_all = detections_all.append( detections )
                
    return detections_all
"""