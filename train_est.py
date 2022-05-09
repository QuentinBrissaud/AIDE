#!/usr/bin/env python3
import numpy as np
from pdb import set_trace as bp
import pandas as pd
import os
import obspy
import pickle
import joblib
import time
import string

import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scipy import signal
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.tf_misfit import cwt, plot_tfr, plot_tf_gofs
from scipy import signal, interpolate
from obspy.signal.trigger import classic_sta_lta

from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.base import clone
import itertools
      
import compute_params_waveform, read_data, detector

def create_CNN(input_shape, Nclasses=2, loss='binary_crossentropy', optimizer='adam'):

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense

    ## Create model
    model = Sequential()
    
    ## Add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    if Nclasses > 2:
        model.add(Dense(Nclasses, activation='softmax'))
    else:
        model.add(Dense(1, activation='sigmoid'))
    ## Compile model using accuracy to measure model performance
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model
    
def train_CNN(model, X_train, y_train, X_test, y_test, epochs=3):
    
    X_train_ = reshape_dataframe_spectrograms(X_train, ['spectro'])
    X_test_  = reshape_dataframe_spectrograms(X_test, ['spectro'])
    
    le = build_label_encoder(y_train)
    y_train_ = le.transform(y_train)
    y_test_  = le.transform(y_test)
    
    ## Train the model
    model.fit(X_train_, y_train_, validation_data=(X_test_, y_test_), epochs=epochs)

def count_values_over_threhsold(x, threshold):
    
    """
    Add up all correlations over a given threshold
    """

    loc_threshold = np.where(abs(x.values) >= threshold)[0]
    x['count'] = abs(x[loc_threshold]).sum()
    return x

def get_number_correlations(correlations, threshold):
    
    """
    Compute number of correlations over a threshold for each row
    """
    
    number_correlations = (abs(correlations)>=threshold).sum(1)
    number_correlations.sort_values(inplace=True, ascending=False)
    
    return number_correlations

def select_uncorrelated_input_features(data, type_corr='spearman', threshold=0.6):

    """
    Iteratively remove most correlated inputs from input features
    """
        
    list_corr    = data_without_info_columns(data).columns
    correlations = find_list_correlations(data, list_corr, type_corr)
    
    ## Number of values over the threhsold for each input
    most_correlated_inputs = correlations.apply(count_values_over_threhsold, args=[threshold], axis=1)
    most_correlated_inputs = most_correlated_inputs['count'].copy()
    most_correlated_inputs.sort_values(inplace=True, ascending=False)
    
    cpt_correlation = 0
    correlations_output        = correlations.copy()
    number_correlations_output = get_number_correlations(correlations_output, threshold)
    while cpt_correlation < len(most_correlated_inputs) and \
        number_correlations_output.sum() > correlations_output.shape[0]:
            
            correlations_output = \
                correlations_output.drop(most_correlated_inputs.index[cpt_correlation], axis=0)
            correlations_output = \
                correlations_output.drop(most_correlated_inputs.index[cpt_correlation], axis=1)
            
            number_correlations_output = get_number_correlations(correlations_output, threshold)
            
            cpt_correlation += 1
    
    return correlations, correlations_output

def find_list_correlations(data, list_corr, type_corr):

    """
    Compute cross correlations between inputs features in data[list_corr]
    """

    ## Find data to get correlations from
    list_observations_corr = data[list_corr]
    correlations = list_observations_corr.corr(method=type_corr)
    
    return correlations
    
def plot_statistics(data, list_corr, options, type_corr = 'spearman', fontize=10.):

    """
    Plot Spearman's correlation between input features.
    """
    
    ## Compute correlations
    correlations = find_list_correlations(data, list_corr, type_corr)
    correlations = correlations.dropna(thresh=2, axis='rows')
    correlations = correlations.dropna(axis='columns')
    
    ## Plot correlations
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(correlations, cbar=True, ax=ax, vmin=0., vmax=0.8, cbar_kws={'extend':'both'})
    ax.set_title(type_corr + ' correlation coefficients', fontsize=fontize)
    fig.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.95)
    fig.savefig(options['DIR_FIGURES'] + 'correlation_detections_'+read_data.get_str_options(options)+'.pdf')
    plt.close('all')
    
    return correlations
    
def generate_colormap(transparent_below_threshold):
    
    """
    Generate a colormap with a without a transparent end
    """
    
    cmap = sns.color_palette("flare", as_cmap=True)
    cmap_background = cmap(np.arange(cmap.N))
    if transparent_below_threshold:
        cmap_background[:,-1] = np.linspace(0., 1, cmap.N)
    
    cmap_background = ListedColormap(cmap_background)
    if not transparent_below_threshold:
        cmap_background.set_under(color='grey', alpha=0.2)
        
    return cmap_background

def plot_probability_curves(ax, probas, window, norm_proba, cmap, shift_window, Npoints=10000):

    """
    Plot detection probabilities as filled curves
    """

    ## Create time vector
    add_number = 100
    shift_window_ = shift_window#window*0.5/3600.
    t     = shift_window_ + probas['time'].values/3600.
    t[0]  = probas['time'].min()/3600.
    t_add = np.linspace(t[0], t[1], add_number+1)
    t     = np.concatenate((t_add, t[2:]), axis=None)
    
    ## Create probability vector
    y1     = probas['proba'].values
    y1_add = np.zeros((add_number,)) + y1[0]
    y1     = np.concatenate((y1_add, y1[1:]), axis=None)
    
    f = interpolate.interp1d(t, y1, kind='cubic', fill_value='extrapolate')
    t = np.linspace(t[0], t[-1], Npoints)
    y1 = f(t)
    
    y2 = y1*0. + 0.5
    y2 = np.ma.masked_greater(y2, 0.5)
    y3 = np.ma.masked_greater(y1, 0.5)

    ax.plot(t, y1, color=cmap[-1], linewidth=2)
    ax.plot(t, y3, color=cmap[10], linewidth=2)
    ax.fill_between(t, y1, y2, where=y2 >= y1,
                    facecolor=cmap[0], alpha=0.4, interpolate=True)
    ax.fill_between(t, y1, y2, where=y2 <= y1,
                     facecolor=cmap[-1], alpha=0.4, interpolate=True)

def plot_background_shading_arrival_RF(axs, probas_, window, shift_window, begin_time, nb_picks=5, axins_inset_vTEC=None):

    """
    Make the plot background grey is there is a RF arrival
    """

    grouped_detections = probas_.groupby('arrival_class')
    for cpt, (group, detection) in enumerate(grouped_detections):
        
        ## Only add label for first detection patch to not overload caption
        label = ''
        if cpt == 0:
            label = 'RF arrival'
        
        ## Only shift arrival time if this is not the first window
        shift_window_ = shift_window
        #shift_window_ = 0.5*window/3600.
        #if abs(detection.time.min() - begin_time) < window/2.:
        #    shift_window_ = 0.1*window/3600.
        
        #first_detection = detection.loc[detection.time == detection.time.min(), :].iloc[0]
        
        ## Find arrival time
        predicted_time = detection.time.min()/3600. + shift_window_
        first_detection = detection.loc[detection.proba == detection.proba.max(), :].iloc[0]
        if not first_detection['predicted-time'] == -1:
            #for idetect, detect in detection.iloc[:nb_picks].iterrows():
                #axs[0].axvline(detect['predicted-time']/3600., color='tab:red', alpha=0.5, zorder=15)
            predicted_time = detection['predicted-time'].quantile(q=0.5)/3600.
            
        axs[0].axvline(predicted_time, color='darkslategrey', alpha=1., linewidth=2., zorder=10)
        axs[1].axvline(predicted_time, color='darkslategrey', alpha=1., linewidth=2., zorder=10)
        axs[0].axvspan(predicted_time, \
                       detection.time.max()/3600. + shift_window_, \
                       facecolor='grey', alpha=0.18, label=label, zorder=1);
        axs[1].axvspan(predicted_time, \
                       detection.time.max()/3600. + shift_window_, \
                       facecolor='grey', alpha=0.18, zorder=1);

        if not axins_inset_vTEC == None:
            axins_inset_vTEC.axvline(predicted_time, color='darkslategrey', linewidth=2., alpha=1., zorder=10)
            axins_inset_vTEC.axvspan(predicted_time, \
                       detection.time.max()/3600. + shift_window_, \
                       facecolor='grey', alpha=0.18, label=label, zorder=1);

def plot_AN_curves(axs, no_analytical, analytical_detections):

    """
    Plot detected arrivals using Elvira's analytical method
    """

    t  = analytical_detections.time.values/3600.
    y1 = np.array(analytical_detections['class'].values, dtype='int')
    y2 = y1*0
    axs[no_analytical].plot(t, y1, color='tab:pink', linewidth=2.)
    axs[no_analytical].fill_between(t, y1, y2, where=y2 <= y1, facecolor='tab:pink', alpha=0.4, interpolate=True)
    
    AN_detected_arrivals = analytical_detections.loc[analytical_detections['class'] == 1, :]
    for cpt, (iAN_arrival, AN_arrival) in enumerate(AN_detected_arrivals.iterrows()):
        label = '_nolegend_'
        if cpt == 0:
            label = 'AN arrival'
        axs[0].axvline(AN_arrival['time']/3600., color='tab:pink', alpha=0.4, label=label, zorder=10)

    axs[no_analytical].text(0.995, 0.98, 'AN', fontsize=10, ha='right', va='top', transform = axs[no_analytical].transAxes)

def plot_STA_LTA_curves(axs, no_stalta, sta_lta_detections, cmap, options):

    """
    Plot detected arrivals and characteristic function using classic STA/LTA
    """
    
    colors = ['tab:green', 'tab:orange']
        
    t  = sta_lta_detections.time.values/3600.
    y1 = sta_lta_detections['proba'].values
    y2 = y1*0. + options['STALTA_threshold_in']
    grouped_sta_lta_detections = \
        sta_lta_detections.loc[sta_lta_detections['arrival_class'] >= 0, :].groupby('arrival_class')
    for group, detection in grouped_sta_lta_detections:
        itimemin = np.argmin(abs(t - detection.time.min()/3600.))
        itimemax = np.argmin(abs(t - detection.time.max()/3600.))
        y2[itimemin:itimemax+1] = options['STALTA_threshold_out']
        
        axs[no_stalta].axvspan(detection.time.min()/3600., \
                               detection.time.max()/3600., \
                               facecolor='grey', alpha=0.18, zorder=1);
        
    y3 = np.ma.masked_greater(y1, y2)

    axs[no_stalta].axhline(options['STALTA_threshold_in'], color='black', alpha=0.5, linestyle=':', label='thresholds')
    axs[no_stalta].axhline(options['STALTA_threshold_out'], color='black', alpha=0.5, linestyle=':')
    axs[no_stalta].plot(t, y1, color=colors[0], linewidth=2)
    axs[no_stalta].plot(t, y3, color=cmap[0], linewidth=2)
    axs[no_stalta].fill_between(t, y1, y2, where=y2 >= y1,
                    facecolor=cmap[0], alpha=0.4, interpolate=True)
    axs[no_stalta].fill_between(t, y1, y2, where=y2 <= y1,
                     facecolor=colors[0], alpha=0.4, interpolate=True)    
    axs[no_stalta].set_ylabel('Charac.\nfunction')
    axs[no_stalta].legend(loc='upper left')
    
    for cpt, (group, detection) in enumerate(grouped_sta_lta_detections):
        label = '_nolegend_'
        if cpt == 0:
            label = 'STA/LTA arrival'
        axs[0].axvline(detection['time'].min()/3600., color=colors[0], label=label, linewidth=1., zorder=110)
        axs[no_stalta].axvline(detection['time'].min()/3600., color=colors[0], label=label, linewidth=1., zorder=110)

    axs[no_stalta].text(0.995, 0.98, 'STA', fontsize=10, ha='right', va='top', transform = axs[no_stalta].transAxes)

def plot_processed_timeseries(event, satellite, station, waveform, probas, probas_, window, options, 
                              transparent_below_threshold=True, true_arrival=-1, 
                              analytical_detections=pd.DataFrame(),
                              sta_lta_detections=pd.DataFrame(), figsize=(),
                              nb_picks = 5, add_label='', fsz=15., add_inset=False) -> None:

    """
    Plot timeseries and detection probabilities for a given event/satellite/station
    """
    
    begin_time = waveform['time_s'].min()

    ## Setup figures
    #fig, axs   = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    nb_rows = 4
    no_analytical = -1
    if analytical_detections.size > 0:
        nb_rows += 1
        no_analytical = nb_rows-3
    no_stalta = -1
    if sta_lta_detections.size > 0:
        nb_rows += 1
        no_stalta = nb_rows-3
    grid = fig.add_gridspec(nb_rows, 1)
    axs = []
    axs.append( fig.add_subplot(grid[:3]) )
    axs.append( fig.add_subplot(grid[3]) )
    
    if analytical_detections.size > 0:
        axs.append( fig.add_subplot(grid[no_analytical+2]) )
    if sta_lta_detections.size > 0:
        axs.append( fig.add_subplot(grid[no_stalta+2]) )
    
    ## Background probabilities colormap
    probas_val      = np.linspace(0., 1., 100)
    cmap_background = generate_colormap(transparent_below_threshold)
    cmap_background = [cmap_background(x) for x in np.linspace(0., 1., len(probas_val))]
    
    ## Second panel colormap
    cmap = generate_colormap(False)
    bounds = np.linspace(0, 1, 3)
    norm_proba = mcolors.BoundaryNorm(bounds, cmap.N)
    cmap = [cmap(x) for x in np.linspace(0., 1., len(probas_val))]
    
    ## Plot vTEC time series
    times = waveform['time_s'].values/3600.
    axs[0].plot(times, waveform.vTEC, color='midnightblue', label='_nolegend_', zorder=100);
    axs[0].set_ylabel('vTEC (TECU)', fontsize=fsz)
    
    ## Plot inset vTEC
    axins_inset_vTEC = None
    if add_inset:
        axins_inset_vTEC = inset_axes(axs[0], width="20%", height="30%", loc='lower left', 
                                    bbox_to_anchor=(0.4, 0.6, 1, 1.), 
                                    bbox_transform=axs[0].transAxes, borderpad=0)
        axins_inset_vTEC.set_xlabel('')
        axins_inset_vTEC.set_ylabel('')
        axins_inset_vTEC.set_xticklabels([])
        axins_inset_vTEC.set_yticklabels([])
        print(waveform, true_arrival)
        loc_waveform = waveform.loc[(waveform.UT >= true_arrival/3600.-0.1) & (waveform.UT <= true_arrival/3600.+0.1)]
        axins_inset_vTEC.plot(loc_waveform.UT, loc_waveform.vTEC, color='midnightblue', label='_nolegend_', zorder=100)
        axins_inset_vTEC.set_xlim([true_arrival/3600.-0.025, true_arrival/3600.+0.025])
        axs[0].indicate_inset_zoom(axins_inset_vTEC, edgecolor="black")
        mark_inset(axs[0], axins_inset_vTEC, loc1=3, loc2=1, fc="none", ec="0.5")
        
    ## Plot background shading in probabilities to flag arrivals
    #shift_window = (window - window*(1./options['factor_overlap']))/3600.
    #shift_window = 0.7 * window/3600.
    shift_window = 1. * window/3600.
    if probas_.size > 0:
        plot_background_shading_arrival_RF(axs, probas_, window, shift_window, begin_time, nb_picks=nb_picks, axins_inset_vTEC=axins_inset_vTEC)
    
    ## Plot true arrival
    if true_arrival > -1:
        axs[0].axvline(true_arrival/3600., color='tab:red', alpha=0.5, zorder=1000, label='True arrival', linewidth=2.)
        if add_inset:
            axins_inset_vTEC.axvline(true_arrival/3600., color='tab:red', alpha=0.5, zorder=1000, label='True arrival', linewidth=2.)
            
    ## Plot analytical detector
    if analytical_detections.size > 0:
        plot_AN_curves(axs, no_analytical, analytical_detections)
    
    ## Plot STA/LTA detector
    if sta_lta_detections.size > 0:
        plot_STA_LTA_curves(axs, no_stalta, sta_lta_detections, cmap, options)
        
    ## Plot probabilities in second panel
    plot_probability_curves(axs[1], probas, window, norm_proba, cmap, shift_window)
    axs[1].text(0.995, 0.98, 'RF', fontsize=fsz-2., ha='right', va='top', transform = axs[1].transAxes)
    
    #axs[1].grid()
    axs[-1].set_xlabel('Time (UT)', fontsize=fsz)
    axs[1].set_ylabel('Detection\nprobability', fontsize=fsz)
    for iax, ax in enumerate(axs[1:]):
        if iax+1 == no_stalta:
            continue
        ax.set_ylim([0., 1.])
    
    str_title = event
    if '_' in event:
        str_title = event.split('_')[0]
    
    
    axs[0].set_title(str_title, pad=20, fontsize=fsz)
    axs[0].text(0.5, 1.02, ' satellite ' + satellite + ' - station '+ station, fontsize=fsz-3., ha='center', va='bottom', transform = axs[0].transAxes)
    axs[0].get_xaxis().set_visible(False)
    if analytical_detections.size > 0 or sta_lta_detections.size > 0:
        axs[1].get_xaxis().set_visible(False)
    axs[0].legend()
    
    ## Shrink time bounds
    ## Only shift arrival time if this is not the first window
    tproba = shift_window + probas['time'].values/3600
    tproba[0] = begin_time/3600.
        
    for ax in axs:
        minbounds = tproba[0]
        # If there is an arrival that is before the range of RF probabilities, we stretch the xaxis to include it in the plot
        #if true_arrival > -1:
        #    minbounds = min(minbounds, true_arrival/3600.)
        ax.set_xlim([minbounds, tproba[-1]])
    
    ## Plot scale
    ylim = axs[0].get_ylim()
    scale_location = [(tproba[0], 0), (tproba[0]+window/3600., 0)]
    trans = (axs[0].transData + axs[0].transAxes.inverted()).transform(scale_location)
    axs[0].plot(trans[:, 0]-trans[:, 0].max()-0.001, [1.05, 1.05], '|-', 
                color='black', clip_on=False, transform = axs[0].transAxes)
    axs[0].text(trans[:, 0].min(), 1.02, 'window length', fontsize=fsz-2., ha='left', va='bottom', transform = axs[0].transAxes)
    
    ## Plot overlap
    scale_location = [(tproba[0], 0), (tproba[0] + window/3600.-shift_window, 0)]
    trans = (axs[0].transData + axs[0].transAxes.inverted()).transform(scale_location)
    axs[0].plot(trans[:, 0]-trans[:, 0].max()-0.001, [1.19, 1.19], '|-', 
                color='steelblue', clip_on=False, transform = axs[0].transAxes)
    axs[0].text(trans[:, 0].min(), 1.16, 'time shift', color='steelblue', fontsize=fsz-2., ha='left', va='bottom', transform = axs[0].transAxes)
    
    ## Figure dimensions
    fig.subplots_adjust(left=0.1, right=0.95, top=0.85, hspace=0.16, bottom=0.15)
    fig.align_ylabels(axs)
    """
    axins = inset_axes(axs[0], width="1%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[0].transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sc, cax=axins, extend='both')
    cbar.ax.set_ylabel('Detection probability', rotation=270, labelpad=16)
    """
    
    ## Label for paper
    if add_label:
        axs[0].text(-0.05, 1.05, add_label, ha='right', va='bottom', transform=axs[0].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=20., fontweight='bold')
    
    ## Save figure
    name_figure = str(event) + '_' + str(satellite) + '_' + str(station)
    comp_analytic = ''
    if analytical_detections.size > 0:
        comp_analytic = '_withanal'
    fig.savefig(options['DIR_FIGURES'] + 'prediction_detection_'+name_figure+comp_analytic+'.pdf')
    
    plt.close('all')

def compute_arrival_time_multiple_stations(tec_data_param, probas, window, nb_for_class=5, nb_for_end_class=3):

    detections = pd.DataFrame()
    grouped_data = probas.groupby(['event', 'satellite', 'station'])
    for group, proba in grouped_data:
        
        event, satellite, station = group
        params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                        & (tec_data_param['satellite'] == satellite) 
                        & (tec_data_param['event'] == event), : ]
        
        true_arrival = -1
        if params.size > 0:
            true_arrival = params.iloc[0]['arrival-time']

        detection = \
            compute_arrival_time(proba, window, 
                                 nb_for_class=nb_for_class, 
                                 nb_for_end_class=nb_for_end_class)
        detection['true-arrival-time'] = true_arrival

        detections = detections.append( detection )

    return detections

def compute_arrival_time(probas, window, nb_for_class=5, nb_for_end_class=3):

    """
    Retrieve all wavepackets corresponding to arrivals based on the minimum number of points for an arrival
    """

    arrival_time_, proba = -1., -1.
    probas_ = probas.copy()
    
    if probas_.shape[0] > 0:
        
        probas_.sort_values(by='time', inplace=True)
        probas_ = probas_.loc[probas_['class'] == 1, :]
        probas_ = probas_.reset_index(drop=True)
        if probas_.size > 0:

            dt = np.diff(probas.time).max()
            current_class = 0
            probas_       = probas_.assign(arrival_class = current_class)
            for iproba, proba_onetime in probas_.iloc[1:].iterrows(): 
                gap = proba_onetime['time'] - probas_.iloc[iproba-1]['time']
                if (gap > nb_for_end_class*dt+0.1) \
                    or (gap > dt+0.1 and probas_.loc[(probas_['arrival_class'] == current_class) \
                                                    & (probas_.index<iproba), :].shape[0] < nb_for_class): 
                    current_class = probas_.iloc[iproba-1]['arrival_class']+1
                    probas_.loc[probas_.index>=iproba, 'arrival_class'] = probas_.iloc[iproba-1]['arrival_class']+1;
                
            ## get the first element of the largest group
            probas_['count_class'] = probas_.groupby('arrival_class').transform('count')['class']
            
            probas_ = probas_.loc[probas_['count_class'] >= nb_for_class, :]
            
        """
        if probas_.size > 0:
            arrival_time_ = probas_.iloc[0]['time']
            proba = probas_.iloc[0]['proba']
                
        return arrival_time_ + window*0.5, proba
        """
    
    return probas_

def process_timeseries_with_forest(time_end, est, tec_data, tec_data_param, event, satellite, station, 
                                   columns_in, options, plot_probas=False, type='features', standard_sampling= 30.,
                                   standard_sampling_for_picker=30., adaptative_sampling=False,
                                   figsize=(), add_label='', determine_elapsed_time=False,
                                   est_picker=None, use_STA_LTA_for_picking=False,
                                   return_all_waveforms_used=False,
                                   nb_picks=5, activate_LTA_STA=False, 
                                   time_STA=60., time_LTA=300,
                                   STA_LTA_threshold_in=1.5, STA_LTA_threshold_out=0.1, add_inset=False,
                                   zscore_threshold = 50.) -> dict:

    """
    Compute detection probabilities for a given satellite and a given event 
    """

    class_to_no = {'arrival': 1, 'noise': 0}
    
    if determine_elapsed_time:
        time_elapsed = pd.DataFrame()

    ## Plot proba for an example
    #example_index = data.loc[data.index == id, :].iloc[0]
    #event, satellite, station = example_index['event'], example_index['satellite'], example_index['station']
    waveform = tec_data.loc[ (tec_data['station'] == station) & (tec_data['satellite'] == satellite) & (tec_data['event'] == event), : ]
    waveform = waveform.loc[waveform['time_s'] <= time_end, :] # Only selecte waveforms up to the end time requested
    sampling = waveform.iloc[0]['sampling']
    
    ## Remove outliers
    unknown = 'vTEC'
    std = waveform[unknown].std()
    if std == 0.:
        observation = {
            'probas': pd.DataFrame(),
            'detections': pd.DataFrame()
        }
        return observation
        
    outliers = waveform.loc[abs(waveform[unknown] - waveform[unknown].mean()/std) > zscore_threshold, :]
    waveform = waveform.loc[~waveform.index.isin(outliers.index)]
    
    if waveform.size == 0:
        observation = {
            'probas': pd.DataFrame(),
            'detections': pd.DataFrame()
        }
        return observation
    
    ## Get time windows
    times  = waveform['time_s'].values[:]
    window = read_data.get_window(sampling, options['window'])
    
    factor_overlap = options['factor_overlap']
    freq_max = options['freq_max']
    if adaptative_sampling:
        factor_overlap = window/np.round(sampling)
        freq_max = 1./standard_sampling
    
    time_boundaries      = np.arange(times[0], times[-1]-window, window/options['factor_overlap'])
    iloc_time_boundaries = [np.argmin(abs(time - times)) for time in time_boundaries]
    #size_subset          = np.argmin( abs((times-times[0]) - window) ) + 1
    size_subset = np.arange(0., window+sampling, sampling).size
    
    ## Compute parameters for each time step
    probas  = pd.DataFrame()
    all_waveforms_used = pd.DataFrame()
    for i0 in iloc_time_boundaries:
    
        iend = i0 + size_subset - 1
        #print('----------------------------')
        #print(station, satellite, times[iend]-times[i0])
        if times[i0:iend+1].size < size_subset or abs(times[iend]-times[i0] - window) > 1.: 
            #print('Removed')
            continue
    
        if determine_elapsed_time:
            time_start = time.time()
    
        #print('Verified')
        
        ## Extract waveform over the right subset
        #tr = read_data.create_Trace(times[:iend], waveform['vTEC'].values[:iend], detrend=False)
        #tr.differentiate()
        #tr.trim(tr.stats.starttime + tr.times()[i0], tr.stats.starttime + tr.times()[-1])
        tr, i0_, iend_ = read_data.pre_process_waveform(times, waveform['vTEC'].values, 
                                                      i0, iend, window, detrend=True, 
                                                      bandpass=[options['freq_min'], options['freq_max']],
                                                      standard_sampling=standard_sampling)
        
        #tr_ = compute_params_waveform.create_Trace(times[:], waveform['vTEC'].values, detrend=False, bandpass=[5e-4, options['freq_max']], differentiate=True); tr = compute_params_waveform.create_Trace(times[:iend+1], waveform['vTEC'].values[:iend+1], detrend=False, bandpass=[5e-4, options['freq_max']], differentiate=True)
        #tr_, _, _ = read_data.pre_process_waveform(times, waveform['vTEC'].values, i0, iend, window, detrend=False, bandpass=[1e-4, options['freq_max']], standard_sampling=standard_sampling)
        #plt.plot(times, waveform['vTEC'].values); plt.axvline(times[i0]); plt.axvline(times[iend]); plt.show()
        
        ## Compute time-domain features
        type_data = '' # dummy value not used
        features = read_data.extract_features_based_on_input_type(tr, type_data, type, options)

        if determine_elapsed_time:
            time_feature = time.time()

        ## Only select relevant columns 
        data_in = features[columns_in].copy()
        
        ## Quality check
        if not data_in.dropna(axis=1).size == data_in.size:
           continue
        
        if options['type_ML'] == 'CNN':
            data_in = reshape_dataframe_spectrograms(data_in, ['spectro'])
        
        proba   = est.predict_proba(data_in.values)
        
        class_predicted = class_to_no[est.predict(data_in.values)[0]]
        loc_dict = {
            'time': times[i0],
            'proba': proba[0, 0], 
            'class': class_predicted,
            'predicted-time': -1,
            }
            
        if determine_elapsed_time:
            time_classification = time.time()
            
        if return_all_waveforms_used:
            this_waveform = pd.DataFrame(data=[tr.data])
            this_waveform['time'] = times[i0]
            
            all_waveforms_used = all_waveforms_used.append( this_waveform )
            
        ## If a phase picker is provided, we use it to estimate the arrival time on the current window
        if class_predicted == 1:
            
            ## Downsample trace for picker if needed
            #tr_ = tr.copy()
            tr_, i0_, iend_ = read_data.pre_process_waveform(times, waveform['vTEC'].values, 
                                                      i0, iend, window, detrend=False, 
                                                      bandpass=[options['freq_min'], options['freq_max']],
                                                      standard_sampling=standard_sampling)
            read_data.downsample_trace(tr_, standard_sampling_for_picker)
            vTEC = np.array([tr_.data])
            vTEC /= abs(vTEC).max()
            
            if use_STA_LTA_for_picking:
            
                df  = tr.stats.sampling_rate
                cft = classic_sta_lta(tr.data, int(time_STA * df), int(time_LTA * df))
                detections = detector.recursively_find_wavetrains_STA_LTA(tr.times(), cft, STA_LTA_threshold_in, STA_LTA_threshold_out)
                if detections.loc[detections.arrival_class>-1, 'time'].size > 0:
                    loc_dict['predicted-time'] = times[i0] + detections.loc[detections.arrival_class>-1, 'time'].iloc[0]    
            
            elif not est_picker == None:
                
                if activate_LTA_STA:
                    df  = tr_.stats.sampling_rate
                    cft = classic_sta_lta(tr_.data, int(time_STA * df), int(time_LTA * df))
                    time_cft_max = tr_.times()[cft.argmax()]
                    cft_max = cft.max()
                    cft = np.array([[time_cft_max, cft_max]])
                    vTEC = np.concatenate((vTEC, cft), axis=1)
                    
                loc_dict['predicted-time'] = times[i0] + window*0.5 + est_picker.predict(vTEC)[0]
                
        if determine_elapsed_time:
            time_prediction = time.time()
            
        #if class_predicted == 1:
        if False:
            
            vTEC = np.array([tr.data])
            vTEC /= abs(vTEC).max()
            
            predicted_time = times[i0] + window*0.5 + est_picker.predict(vTEC)[0]
            
            """
            tr_ = obspy.Trace()
            tr_.data        = waveform['vTEC'].values[i0:iend+1]
            tr_.stats.delta = abs( times[1] - times[0] )
            tr_.filter("bandpass", freqmin=options['freq_min'], freqmax=options['freq_max'], zerophase=True)
            tr_.differentiate()
            tr_.detrend("polynomial", order=1)
            
            plt.plot(tr.times(), tr.data); plt.plot(tr_.times(), tr_.data); plt.show()
            features = read_data.extract_features_based_on_input_type(tr_, type_data, type, options); data_in = features[columns_in].copy(); proba   = est.predict_proba(data_in.values)
            """
            
            tr_, _, _ = read_data.pre_process_waveform(times, waveform['vTEC'].values, 0, waveform['vTEC'].values.size-1, 100000., detrend=True, bandpass=[options['freq_min'], options['freq_max']],standard_sampling=standard_sampling)
            tr_.trim(starttime=tr_.stats.starttime + times[i0]-times[0], endtime=tr_.stats.starttime + times[iend]-times[0])
        
            fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
            axs[0].plot(waveform['time_s'].values, waveform['vTEC'].values); 
            axs[0].axvline(waveform['time_s'].values[i0], color='red')
            axs[0].axvline(waveform['time_s'].values[iend], color='red')
            axs[0].axvline(predicted_time, color='green')
            axs[1].plot(waveform['time_s'].values[i0] + tr.times(), tr.data)
            axs[1].plot(waveform['time_s'].values[i0] + tr_.times(), tr_.data)
            axs[1].axvline(predicted_time, color='green')
            plt.show()
            bp()
        
        probas = probas.append( [loc_dict] )
        
        if determine_elapsed_time:
            detections = compute_arrival_time(probas, window, nb_for_class=options['nb_for_class'], 
                                      nb_for_end_class=options['nb_for_end_class'])
            time_validation = time.time()
            loc_time = {
                'time': times[i0],
                'feature': time_feature - time_start,
                'classification': time_classification - time_feature,
                'time-picking': time_prediction - time_classification,
                'validation': time_validation - time_prediction
            }
            time_elapsed = time_elapsed.append( [loc_time] )
    
    #probas = pd.DataFrame(probas)
    
    ## Determine arrival 
    #arrival_time_, proba_ = compute_arrival_time(probas, window, nb_for_class=options['nb_for_class'])
    detections = compute_arrival_time(probas, window, nb_for_class=options['nb_for_class'], 
                                      nb_for_end_class=options['nb_for_end_class'])
    
    ## Plot time series along with detection probabilities
    if plot_probas:
        true_arrival = -1
        if tec_data_param.size > 0:
            true_arrival = tec_data_param.epoch
            #true_arrival = tec_data_param.epoch
        print('Plotting')
        plot_processed_timeseries(event, satellite, station, waveform, 
                                  probas, detections, window, options,
                                  true_arrival=true_arrival, figsize=figsize,
                                  add_label=add_label, nb_picks=nb_picks,
                                  add_inset=add_inset)
        
        #from importlib import reload; import train_est; reload(train_est)
        #train_est.plot_processed_timeseries(event, satellite, station, waveform, probas, detections, window, options,true_arrival=true_arrival,add_label=add_label, nb_picks=nb_picks,add_inset=add_inset, figsize=(10, 4))
        
    observation = {
            #'arrival-time': arrival_time_,
            #'proba-max': proba_,
            'detections': detections,
            'probas': probas
    }
    
    if return_all_waveforms_used:
        observation['all_waveforms_used'] = all_waveforms_used
    
    if determine_elapsed_time:
        return observation, time_elapsed
    else:
        return observation
        
def plot_cum_distribution(data_all, label_columns, options):

    nb_cols = 4
    nb_rows = int(np.ceil(data_all.keys().size / nb_cols))
    fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols, sharey=True)

    for iunknown, unknown in enumerate(data_all.keys()):
    
        i, j = iunknown//nb_cols, np.mod(iunknown, nb_cols)

        data_cumul  = data_all[unknown].values
        sorted_data = np.sort(data_cumul)  # Or data.sort(), if data can be modified
        
        bins = np.arange(sorted_data.size)
        axs[i, j].step(sorted_data, bins/bins.max())
        axs[i, j].grid()
        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])
        axs[i, j].text(0.01, 0.99, label_columns[unknown], fontsize=10, ha='left', va='top', transform = axs[i, j].transAxes)
        axs[i, j].set_xlim([sorted_data.min(), sorted_data.max()])
        axs[i, j].set_ylim([0., 1.])
            
    ## Remove unused subplots
    for iunknown in range(data_all.keys().size, nb_rows*nb_cols):
        i, j = iunknown//nb_cols, np.mod(iunknown, nb_cols)
        fig.delaxes(axs[i, j])
    
    #axs[-1, 0].set_ylabel('Cum. distribution')
    
    fig.subplots_adjust(top=0.97, bottom=0.03, hspace=0.05)
    
    fig.savefig(options['DIR_FIGURES'] + 'cum_distributions_'+read_data.get_str_options(options)+'.pdf')

def plot_RF_importance(est, input_columns, options):
        
    """
    Plot relative importance (Gini coefficient) of input features
    """
    
    oob_score = est.oob_score_
    
    importances = {}
    for name, importance in zip(input_columns, est.feature_importances_): 
        importances[name] = importance;
    std = np.std([tree.feature_importances_ for tree in est.estimators_], axis=0)
    importances_tab = est.feature_importances_
    indices = np.argsort(importances_tab)[::-1]
    keys    = np.array(input_columns)[indices]
        
    fig, axs = plt.subplots(nrows=1, ncols=1)#, gridspec_kw=gridspec_kw)
    plt.title("Feature importances with accuracy "+ str(round(oob_score,2)))
    plt.bar(range(len(input_columns)), importances_tab[indices], color="tab:blue", yerr=std[indices], align="center")
    plt.xticks(range(len(input_columns)), keys, rotation='vertical')
    plt.xlim([-1, len(input_columns)])
   
    fig.subplots_adjust(bottom=0.25)
    
    fig.savefig(options['DIR_FIGURES'] + 'RF_importance_ML_'+read_data.get_str_options(options)+'.pdf')
    plt.close('all')
   
def build_label_encoder(output):

    """
    Encode categorical input features
    """

    le = preprocessing.LabelEncoder()
    le.fit(output.values)
    
    return le
    
def CNN_get_binary_class_from_proba(prediction, threshold=0.5):

    """
    Return a binary class from a list of predicted probabilited bases on an input threshold
    """

    return (prediction > threshold).astype(int)
    
def get_classes_encoded(est, data_test_in, out_test, type_ML, iclass = 0):

    """
    Return the encoded predicted and true output from the test dataset
    """

    data_test = data_test_in.copy()
    if type_ML == 'CNN':
        data_test = reshape_dataframe_spectrograms(data_test, ['spectro'])

    out_pred_proba = est.predict_proba(data_test)[:, iclass]
    
    if type_ML == 'CNN':
        out_pred_class = CNN_get_binary_class_from_proba(out_pred_proba, threshold=0.5)
    else:
        out_pred_class = est.predict(data_test)[:]
        
    out_test_class = out_test.copy()
    classes = np.unique(out_test.values)
    ## Only encode outputs if needed, i.e., if RF that uses categorical variables
    if not type_ML == 'forest':
        le = build_label_encoder(out_test)
        out_test_class = le.transform(out_test_class.values[:, iclass])
        out_pred_class = le.transform(out_pred_class)
        classes = np.arange(0, len(le.classes_))
    
    return out_pred_proba, out_pred_class, out_test_class, classes
    
def get_fpr_tpr(est, data_test, out_test, type_ML, data, iclass = 0):
    
    """
    Get false and true positive rates
    """
    
    out_pred, _, out_test_, _ = \
        get_classes_encoded(est, data_test, out_test, type_ML, iclass = 0)
    
    fpr, tpr, _ = roc_curve(out_test_, out_pred, pos_label=iclass)
    
    return fpr, tpr
   
def plot_results_optimization(scores, options, nb_cols=2, fsz=15., fsz_labels=13., metrics_to_plot=['R2', 'MAE-training', 'MAE-test', 'MSE-test']):
    
    """
    Plot results (MAR, MSE, R2) of RF optimization
    """
    
    nb_rows = len(metrics_to_plot) // nb_cols
    fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=(10,4))
    
    shortname_to_label = {
        'oob_score': 'Out-Of-Bag score',
        'noise-precision': 'Precision noise',
        'arrival-precision': 'Precision arrival',
        'noise-recall': 'Recall noise',
        'arrival-recall': 'Recall arrival',
        'accuracy': 'Accuracy',
    }

    irow = 0
    icol = -1
    alphabet = string.ascii_lowercase
    for imetric, metric in enumerate(metrics_to_plot):
        
        icol += 1
        if icol == nb_cols:
            icol  = 0
            irow += 1
        
        scores_plot = scores.copy()
        scores_plot[shortname_to_label[metric]] = scores[metric]
        scores_plot = scores_plot.pivot("max_depth", "nb_trees", shortname_to_label[metric])
        sns.heatmap(scores_plot, annot=True, cbar=False, ax=axs[irow, icol], robust=True, cmap='rocket', center=scores[metric].quantile(0.1), zorder=1)
        axs[irow, icol].set_title(shortname_to_label[metric])
        axs[irow, icol].text(-0.29, 1.05, alphabet[imetric] + ')', ha='right', va='bottom', transform=axs[irow, icol].transAxes, 
                            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=15., fontweight='bold')
        
        args = {}
        if irow+1 == nb_rows and icol == 0:
            axs[irow, icol].set_ylabel('Max. tree depth', fontsize=fsz-2.)
            axs[irow, icol].set_xlabel('Nb of trees', fontsize=fsz-2.)
        else:
            axs[irow, icol].set_yticklabels([])
            axs[irow, icol].set_ylabel('')
            axs[irow, icol].set_xticklabels([])
            axs[irow, icol].set_xlabel('')
        
        ymax = scores_plot.idxmax().idxmax()
        ypos = scores_plot.columns.get_loc(ymax)
        xmax = scores_plot.idxmax().max()
        xpos = scores_plot.index.get_loc(xmax)
        
        y, x = np.where(scores_plot.values == scores_plot.values.max())
        ypos = x[0]
        xpos = y[0]
        
        print(metric, ymax, ypos, xmax, xpos, scores_plot[ymax].max())
        axs[irow, icol].add_patch(Rectangle((ypos, xpos),1,1, fill=False, edgecolor='blue', lw=3, zorder=10))
        
    for irow in range(nb_rows):
        for icol in range(nb_cols):
            ax = axs[irow, icol]
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fsz_labels)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsz_labels)
        
    fig.subplots_adjust(bottom=0.15, left=0.09, right=0.95)
        
    fig.savefig(options['DIR_FIGURES'] + 'optimization_classifier.pdf')

def optimize_RF(data, data_train, out_train, data_test, out_test, seed, split_type,
                l_nb_trees=[1000], l_max_depth=[100], 
                oob_score=True, bootstrap=True,
                class_weight={'noise': 1, 'arrival': 1},
                hidden_layer_sizes=(64, 64, 64), learning_rate_init=0.05, 
                early_stopping=True, max_iter=10000, two_steps=True):
    
    """
    Optimize RF tree depth and nb of tress
    """
    
    scores = pd.DataFrame()
    for nb_trees in l_nb_trees:
        for max_depth in l_max_depth:
            est = create_classifier_and_train(data, data_train, out_train, data_test, out_test,
                                              split_type, seed, type_ML = 'forest', 
                                              nb_trees=nb_trees, max_depth=max_depth,
                                              oob_score=oob_score, bootstrap=bootstrap,
                                              SMOTE=True, class_weight=class_weight,
                                              hidden_layer_sizes=hidden_layer_sizes, 
                                              learning_rate_init=learning_rate_init, 
                                              early_stopping=early_stopping, max_iter=max_iter, two_steps=two_steps)
                                           
            #preds_training = est.oob_decision_function_[:,0]>=0.5
            #preds_test     = est.predict(data_test)
            report_dict = compute_performance_one_model(est, data)
            report = detector.convert_report_to_dataframe(report_dict)
            report['nb_trees']  = nb_trees
            report['max_depth'] = max_depth
            report['oob_score'] = est.oob_score_
            
            scores = scores.append( report )
            #from sklearn import model_selection
            #kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
            #scoring = 'neg_mean_squared_error'
            #results = model_selection.cross_val_score(est, data_picker[input_columns].values, data_picker[output_columns].values[:,0], cv=kfold, scoring=scoring)
            #results_ = model_selection.cross_val_score(est, data_test.values, out_test.values[:,0], cv=kfold, scoring=scoring)
    
    return scores
   
def plot_confusion(est, data_test, out_test, options, type_ML = 'forest', ax=None, cbar=False):

    """
    Compute and plot confusion matrix
    """
    
    #fpr, tpr = get_fpr_tpr(est, data_test, out_test, test_MLP, data)
    _, out_pred_, out_test_, classes_ = \
        get_classes_encoded(est, data_test, out_test, type_ML, iclass = 0)
        
    conf_mat  = confusion_matrix(out_test_, out_pred_, labels=classes_, normalize='true')
    row_, col_ = ['arrival', 'noise'], ['arrival', 'noise']
    df_cm = pd.DataFrame(conf_mat, index = row_, columns = col_)
    
    ## Plot heat map using seaborn
    new_figure = False
    if ax == None:
        new_figure = True
        fig, ax = plt.subplots(nrows=1, ncols=1)
    sb.heatmap(df_cm, ax=ax, annot=True, cbar=cbar)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if new_figure:
        fig.savefig(options['DIR_FIGURES'] + 'confusion_matrix_' + type_ML + '.pdf')
        plt.close('all')
  
def add_zoom_roc(axs, xlim, ylim, fpr, tpr, thresholds, fpr_neg, tpr_neg, thresholds_neg, vmin, vmax, cmap, subsample=1):

    """
    Plot zoom on ROC curve
    """
    
    axins = axs.inset_axes([0.4, 0.03, 0.57, 0.57])
    axins.scatter(fpr[::subsample], tpr[::subsample], c=thresholds[::subsample], 
                  marker='^', vmin=vmin, vmax=vmax, label='Arrival', s=20, cmap=cmap); 
    #axins.scatter(fpr_neg[::subsample], tpr_neg[::subsample], c=thresholds_neg[::subsample], 
     #             marker='o', vmin=vmin, vmax=vmax, label='Noise', s=20, cmap=cmap); 
    axins.plot([0., 1.], [0., 1.], color='black', linestyle=':')
    xlim = [0., 0.2]
    ylim = [0.8, 1.]
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.set_xticklabels('')
    axins.set_xticks([])
    axins.set_yticklabels('')
    axins.set_yticks([])
    axins.legend(frameon=True)

    axs.indicate_inset_zoom(axins, edgecolor="black")

def plot_roc_curve(est, data_test_in, out_test, options, xlim=[0., 1.], ylim=[0., 1.], 
                   type_ML = 'forest', ax=None, vmin=0.2, vmax=1.):

    """
    Plot ROC curve
    """

    ## Encode outputs
    le = preprocessing.LabelEncoder()
    le.fit(out_test.values)
    out_test_ = le.transform(out_test.values)

    ## Decision function
    data_test = data_test_in.copy()
    if type_ML == 'CNN':
        data_test = reshape_dataframe_spectrograms(data_test, ['spectro'])

    out_pred_proba = est.predict_proba(data_test)
    
    ## ROC curve computation
    fpr, tpr, thresholds = roc_curve(out_test_, out_pred_proba[:,1], pos_label=1)
    fpr_neg, tpr_neg, thresholds_neg = roc_curve(out_test_, out_pred_proba[:,0], pos_label=0)
    
    AUC_arrival = auc(fpr, tpr)
    AUC_noise   = auc(fpr_neg, tpr_neg)
    
    ## Plot ROC curve
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    #c0, c1 = cmap[1], cmap[-1]
    new_figure = False
    if ax == None:
        new_figure = True
        fig, ax = plt.subplots(nrows=1, ncols=1)
    sc = ax.scatter(fpr, tpr, c=thresholds, marker='^', vmin=vmin, vmax=vmax, cmap=cmap); 
    #sc = ax.scatter(fpr_neg, tpr_neg, c=thresholds_neg, marker='o', vmin=vmin, vmax=vmax, cmap=cmap); 
    ax.plot([0., 1.], [0., 1.], color='black', linestyle=':')
    ax.grid()
    ax.set_xlim([0., 1.])
    ax.set_ylim([0., 1.])
    ax.set_xlabel('Fall-out')
    ax.set_ylabel('Recall')
    ax.set_title('AUC arrival: ' + str(round(AUC_arrival,2)))# + ' - noise: ' + str(round(AUC_noise,2)), fontsize=10.)
    
    axins_ = inset_axes(ax, width="2%", height="100%", loc='lower left', 
                        bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins_.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sc, cax=axins_, extend='both')
    cbar.ax.set_ylabel('Probability threshold', rotation=270, labelpad=12)
    
    add_zoom_roc(ax, xlim, ylim, fpr, tpr, thresholds, fpr_neg, tpr_neg, thresholds_neg, vmin, vmax, cmap)
    
    if new_figure:
        fig.subplots_adjust(right=0.85)
        fig.savefig(options['DIR_FIGURES'] + 'roc_curve_' + type_ML + '.pdf')
        plt.close('all')
 
def plot_features_vs_features(data, input_columns, est, options, Nfeatures=3, axs=np.empty(0), features_to_remove = ['S14', 'S1']):

    """
    Plot most important input features (up to Nfeatures) against each other
    """
        
    try:
        seaborn_library = True
        import seaborn as sb
    except:
        seaborn_library = False 
            
    importances = {}
    ind = np.argsort(est.feature_importances_)[::-1]
    best_columns = np.array(input_columns)[ind[:]]
    best_scores  = est.feature_importances_[ind[:]]
    for iname, (name, importance) in enumerate(zip(best_columns, best_scores)): 
        if name in features_to_remove or len(importances) == Nfeatures:
            continue
        importances[name] = importance
    list_labels = [name for name in importances]
    
    ## Plot distribution
    if axs.size == 0:
        fig, axs = plt.subplots(nrows=Nfeatures, ncols=Nfeatures)
    
    cmap = sns.color_palette("rocket")
    c0, c1 = cmap[1], cmap[-1]
    list_colors = [c0, c1]
    
    print(list_labels)
    data.sort_values(by='type-data', inplace=True)
    for iunknown, unknown in enumerate(list_labels):
        for iunknown2, unknown2 in enumerate(list_labels):
            ax = axs[iunknown,iunknown2]
            
            if iunknown == iunknown2:
                bool_legend = (iunknown == 0)
                sbplot = sb.histplot(data, x=list_labels[iunknown], hue="type-data", element="poly", ax=ax, legend=bool_legend, palette=list_colors)
                if bool_legend:
                    ax.legend(handles=ax.legend_.legendHandles, labels=[t.get_text() for t in ax.legend_.texts],
                              title=ax.legend_.get_title().get_text(),
                              bbox_to_anchor=(0., 1.05), loc='lower left',
                              frameon=False, ncol=2)
                #print('dadas', data[list_labels[iunknown]])
                #print(np.quantile(data[list_labels[iunknown]], q=0.1), np.quantile(data[list_labels[iunknown]], q=0.8))
                qmin, qmax = 0., 0.9
                xlim = [min(np.quantile(data[list_labels[iunknown]], q=qmin), np.quantile(data[list_labels[iunknown2]], q=qmin)), 
                        max(np.quantile(data[list_labels[iunknown]], q=qmin), np.quantile(data[list_labels[iunknown2]], q=qmax))]
                ax.set_xlim(xlim)
                 
            else:
                sbplot = sb.scatterplot(data=data, x=unknown2, y=unknown, hue="type-data", style="type-data", legend=False, ax=ax, palette=list_colors)
            
            if axs.size > 0:
                sbplot.set(xticklabels=[])
                sbplot.set(yticklabels=[])
                
            ax.set_xlabel('')
            ax.set_ylabel('')
            #if unknown2 == list_labels[0]: 
            if iunknown == 0:
                ax.set_ylabel(unknown2)
            #if unknown == list_labels[-1]: 
            if iunknown2 == len(list_labels)-1:
                ax.set_xlabel(unknown)
    
    if axs.size == 0:
        fig.subplots_adjust(wspace=0.28, hspace=0.28, right=0.97, top=0.97)
        fig.savefig(options['DIR_FIGURES'] + 'distribution_best_features.pdf')        
        plt.close('all')

def remove_outliers(data, unknowns):

    """
    Remove outliers in data based on Z-score
    """
    
    ids_to_remove = []
    for unknown in unknowns:
        ids_to_remove += \
            find_outliers_zscore(data, unknown, zscore_threshold = unknowns[unknown])

    data = data.loc[~data.index.isin(ids_to_remove), :]
    
    return data

def find_outliers_zscore(data, unknown, zscore_threshold = 3.):
    
    """
    Find outliers id in data based on Z-score
    """
    #unknown = 'W7'
    #outlier = data.loc[data[unknown] > 100.*data[unknown].median(), :]
    
    std = data[unknown].std()
    ids_to_remove = []
    if abs(std) > 0.:
        outliers = data.loc[abs(data[unknown] - data[unknown].mean()/std) > zscore_threshold, :]
        ids_to_remove = outliers.index.tolist()
    
    """
    station, satellite, event = outlier.iloc[id].station, outlier.satellite.iloc[id], outlier.iloc[id].event
    waveform = read_data.get_one_entry(tec_data, station, satellite, event)
    id = -1
    plt.plot(waveform.UT, waveform.vTEC); plt.show()
    """
    
    return ids_to_remove
    
def find_all_combi_split(data, split, deviation=0.025):

    """
    Find all combination of events so that the number of waveform over the number total of waveforms in this combinaison if around the split value requested
    """

    pd_coefs = data.loc[data['type-data'] == 'arrival'].groupby('event').station.count().reset_index()
    coefs = pd_coefs.station.values/np.sum(pd_coefs.station.values)
    coefs_event = pd_coefs.event.values
    #nb_waveform_per_combi = []
    locs = []
    nb_event_per_combi = []
    for i in range(len(coefs), 0, -1):
        for seq in itertools.combinations(coefs, i):
            if np.sum(seq) >= split-deviation and np.sum(seq) <= split+deviation:
                loc = []
                for item in seq:
                    loc.append( coefs_event[np.where(coefs==item)[0]][0] )
                locs.append( loc )
                nb_event_per_combi.append( len(loc) )
                #result.append( np.sum(seq) )

    return locs, nb_event_per_combi#, nb_waveform_per_combi
    
def set_train_test(data, split=0.8, seed=1, split_by_event=False):

    """
    Returns Panda DataFrame with columns [time, wnd10hpa, vesp, type] and lines [time]
    """

    np.random.seed(seed)
    
    data['type'] = 'test'
    
    ## Split per event so that both the number of events and number of waveforms in training set correspond to the split value 
    if split_by_event:
        
        combi_events, nb_per_combi = find_all_combi_split(data, split, deviation=0.025)
        nb_total_event = data.loc[(data['type-data'] == 'arrival') & ~data['event_to_testing']].event.unique().size
        #nb_total_event = data.loc[(data['type-data'] == 'arrival')].event.unique().size
        nb_total_event_train = int(np.floor(nb_total_event * split))
        list_possible_combi = np.where(abs(np.array(nb_per_combi)-nb_total_event_train) == abs(np.array(nb_per_combi)-nb_total_event_train).min())[0]
        np.random.shuffle(list_possible_combi)
        combi_selected = combi_events[list_possible_combi[0]]
        
        data.loc[data.event.isin(combi_selected) & ~data['event_to_testing'] & ~data['extra_test'], 'type'] = 'train'
    
    ## Split across all waveforms
    else:
        ## Exclude waveforms from training set that do not contain an arrival label (but could still contain an arrival) or events for testing
        data_grouped_by_satellite = data.loc[~data['extra_test'] & ~data['event_to_testing'], :].groupby(['event', 'satellite'])
        #data_grouped = data.loc[~data['extra_test'], :].groupby('event')
        for (event, satellite), data_ in data_grouped_by_satellite:
        
            list_stations = data_['station'].unique()
            msk = np.random.rand(len(list_stations)) < split
            #idx_train = data_.loc[msk,:].index
            list_stations = list_stations[msk]
            #list_stations = data.loc[data.index.isin(idx_train), 'station'].unique()
            data.loc[(data.event == event) 
                    & (data.satellite == satellite) 
                    & (data.station.isin(list_stations)), 'type'] = 'train'
            #data.loc[(data.event == event) & (data.satellite == satellite) & (data.station.isin(list_stations))]
            #print(event, satellite, list_stations)
        bp()
    
    return data

def scale_inputs(data_in):

    """
    Returns Panda DataFrame with columns [time, wnd10hpa, vesp] and lines [time]
    """

    from sklearn.preprocessing import StandardScaler

    data = data_in.copy()

    ## Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    ## Update dataframe values
    #for iline in range(data.shape[0]):
    #        data.at[iline,'vesp'] = data_scaled[iline, :]

    return data

def remove_correlated_inputs(data_in, label_columns, type_corr='spearman', threshold=0.6, columns_to_remove=[]):

    """
    Remove dataframe columns corresponding to correlated inputs before training
    """
    
    inverse_label_columns = {}
    for key in label_columns: 
        inverse_label_columns[label_columns[key]] = key;
    
    ## Extract column names
    data = data_in.copy()
    if not columns_to_remove:
        #correlations, correlations_output = select_uncorrelated_input_features(data, type_corr=type_corr, threshold=threshold)
        #remove_raw = sorted(set(correlations.columns) - set(correlations_output.columns))
        
        """
        columns_to_remove  = ['ratio-median-max-env', 'energy-autocorr-1-3', 'energy-autocorr-3-4', 'energy-1-Nyf', 'energy-2-Nyf', 'energy-3-Nyf', 'energy-4-Nyf', 'mean-fft', 'max-env', 'var-norm-fft', 'FT4/FT6', 'gyration-radius', 'nb-peaks-amp-mean', 'nb-peaks-amp-max', 'nb-peak-max-freq', 'nb-peak-autocorr', 'FT4/FT5', 'nb-peaks-fft', 'skew-env', 'ratio-max-median-fft', 'nb-peak-central-freq', 'energy-freq-0.03-1.0', 'kurtosis-freq-0.03-1.0', 'W7/W8', 'ratio-max-mean-fft']
        remove_raw = []
        for key in columns_to_remove: 
            remove_raw.append( inverse_label_columns[key] );
        """    
        #remove_raw = ['W0', 'W1', 'W2', 'W3', 'W5', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13', 'W14', 'W15', 'S3', 'S4', 'S5', 'S6', 'S7', 'FT0', 'FT1', 'FT2', 'FT3', 'FT6']
        remove_raw = ['W11', 'W12', 'W13', 'S6', 'S7', 'S8', 'S11', 'S12', 'S14', 'S17', 'FT3']

    else:    
        remove_raw = []
        for key in columns_to_remove: 
            remove_raw.append( inverse_label_columns[key] );
        
    #data = data.loc[:, ~data.columns.isin(remove_raw)]
    
    return data

def find_only_noise_waveforms(x, tec_data_param):
    
    """
    Flag data that have been extracted from a noise-only waveform
    """
    
    event, satellite, station = x.event, x.satellite, x.station
    params = tec_data_param.loc[(tec_data_param['event'] == event) 
                                 & (tec_data_param['satellite'] == satellite) 
                                 & (tec_data_param['station'] == station), :]

    x['only_noise_waveform'] = False
    if params.size == 0:
        x['only_noise_waveform'] = True
    
    return x

def fix_unbalanced_classes(data_in, tec_data_param, factor_major_class=1, priority_only_noise_waveforms=True):
    
    """
    Fix unbalanced classes problem by randomly subsampling the largest class
    """
    
    data = data_in.copy()
    column = data.columns[0]
    size_arrival = data.loc[~data.event_to_testing].groupby('type-data')[column].count()['arrival'] 
    size_noise   = data.loc[~data.event_to_testing].groupby('type-data')[column].count()['noise']
    #size_arrival = data.groupby('type-data')[column].count()['arrival'] 
    #size_noise   = data.groupby('type-data')[column].count()['noise']
    
    #data['extra_test'] = False
    data_only_noise = pd.DataFrame()
    if size_noise > size_arrival:
        #data_arrival = data.loc[data['type-data'] == 'arrival', :]
        #data_noise   = data.loc[data['type-data'] == 'noise', :]
        data_arrival = data.loc[~data.event_to_testing].loc[data['type-data'] == 'arrival', :]
        data_noise   = data.loc[~data.event_to_testing].loc[data['type-data'] == 'noise', :]
        
        size_arrival = size_arrival * factor_major_class
        ## If priority to noise only waveforms, add them first and then the other waveforms if needed
        data_noise = data_noise.apply(find_only_noise_waveforms, args=[tec_data_param], axis=1)
        size_only_noise = data_noise.loc[data_noise['only_noise_waveform'], :].shape[0]
        size_not_only_noise = data_noise.loc[~data_noise['only_noise_waveform'], :].shape[0]
        
        ## Select in priority the noise-only waveforms
        if priority_only_noise_waveforms:
            if size_only_noise < size_arrival:
                list_noise_index  = data_noise.loc[~data_noise['only_noise_waveform'], :].index.tolist()
                np.random.shuffle(list_noise_index)
                data_noise = data_noise.loc[data_noise['only_noise_waveform'] 
                                                   | data_noise.index.isin(list_noise_index[:int(1.1*(size_arrival-size_only_noise))]), :]
                
            else:
                data_noise = data_noise.loc[data_noise['only_noise_waveform'], :]
        
            ## CAREFUl -> 25/12 extra randomization
            list_noise_index  = data_noise.index.tolist()
            np.random.shuffle(list_noise_index)
            #data_noise = data_noise.loc[data_noise.index.isin(list_noise_index[:int(size_arrival)]), :]
            #bp()
        
        ## Select in priority the noise collected around arrivals
        else:
        
            ## If we do not have enough noise samples from waveforms that contain arrivals
            if size_not_only_noise < size_arrival:
                list_noise_index  = data_noise.loc[data_noise['only_noise_waveform'], :].index.tolist()
                np.random.shuffle(list_noise_index)
                data_only_noise = data_noise.loc[~data_noise.index.isin(list_noise_index[:int(1.1*(size_arrival-size_not_only_noise))]), :]
                data_noise = data_noise.loc[~data_noise['only_noise_waveform'] 
                                                   | data_noise.index.isin(list_noise_index[:int(1.1*(size_arrival-size_not_only_noise))]), :]
                
            else:
                data_only_noise = data_noise.loc[data_noise['only_noise_waveform'], :]
                data_noise = data_noise.loc[~data_noise['only_noise_waveform'], :]
                #data_noise = data_noise.loc[data_noise['only_noise_waveform'], :]
                
            """
            data_noise_ = data.loc[~data.index.isin(data_noise.index), :]
            size_sup = size_arrival - data_noise_.shape[0]
            if size_sup > 0:
                data_noise_ = data_noise_.append( data_noise.iloc[:size_sup] )
            data_noise = data_noise_
            """
            
        ## Remove extra noise waveforms to balance dataset
        count_arrival_per_event = data_arrival.groupby('event').station.count().reset_index()
        count_arrival_per_event.station = count_arrival_per_event.station.astype(int)
        count_noise_per_event   = data_noise.groupby('event').station.count().reset_index()
        count_noise_per_event.station = count_noise_per_event.station.astype(int)
        diff_arrival_noise = count_arrival_per_event.copy()
        for ievent, one_event in count_noise_per_event.iterrows():
            event = one_event.event
            #print('@@', diff_arrival_noise.loc[diff_arrival_noise.event == event, 'station'], one_event.station)
            diff_arrival_noise.loc[diff_arrival_noise.event == event, 'station'] -= one_event.station
            #print('-->', diff_arrival_noise.loc[diff_arrival_noise.event == event, 'station'])
            
        data_noise_grouped = data_noise.groupby('event')
        data_noise['keep'] = False
        for event, data_noise_event in data_noise_grouped:
            count_event = count_arrival_per_event.loc[count_arrival_per_event.event==event].station.iloc[0]
            
            loc = data_noise_event.index.tolist()
            np.random.shuffle(loc)
            data_noise.loc[data_noise.index.isin(loc[:count_event]), 'keep'] = True
            #data_noise = data_noise.loc[data_noise.index.isin(loc[:size_arrival]), :]
        
        ## If there are noise window missing because less noise window available than arrival window for certain events
        ## It means that there are less noise windows than arrival windows for certain events 
        total_diff_arrival_noise = data_noise.loc[data_noise.keep].shape[0] - data_arrival.shape[0]
        if total_diff_arrival_noise < 0:
            loc_df = diff_arrival_noise.sort_values(by='station', ascending=False)
            loc_df = loc_df.loc[loc_df.station < 0]
            for ievent, diff_event in loc_df.iterrows():
            
                if total_diff_arrival_noise >= 0:
                    continue
                event = diff_event.event
                
                data_noise_event = data_noise.loc[(data_noise.event==event) & ~data_noise.keep]
                loc = data_noise_event.index.tolist()
                np.random.shuffle(loc)
                #data_noise.loc[data_noise.index.isin(loc[:abs(diff_event.station)]), 'keep'] = True
                data_noise.loc[data_noise.index.isin(loc[:abs(total_diff_arrival_noise)]), 'keep'] = True
                #total_diff_arrival_noise -= diff_event.station
                total_diff_arrival_noise = data_noise.loc[data_noise.keep].shape[0] - data_arrival.shape[0]
                
        list_accepted_index = []
        list_accepted_index += data_noise.loc[data_noise.keep].index.tolist()
        """
        list_accepted_index = []
        loc = data_noise.index.tolist()
        np.random.shuffle(loc)
        data_noise = data_noise.loc[data_noise.index.isin(loc[:size_arrival]), :]
        list_accepted_index += data_noise.index.tolist()
        """
        list_accepted_index += data_arrival.index.tolist()
        
        ## Flag waveform to add to training dataset (i.e., balanced)
        data['event_to_testing'] = False
        data.loc[~data.index.isin(list_accepted_index), 'event_to_testing'] = True
        
        ## Add noise-only waveforms as extra_test to remove them from main testing dataset
        data['extra_test'] = False
        data.loc[data.index.isin(data_only_noise.index) & ~data.index.isin(list_accepted_index), 'extra_test'] = True
        #data.loc[~data.index.isin(list_accepted_index), 'extra_test'] = True
        
    return data

def split_data_before_training(data, input_columns, output_columns, seed,   
                               split_type, split=0.7, twosteps=False, split_by_event=False):
    
    """
    Split dataset between training and testing dataset for a given split
    """
    #data_all = data.loc[:, input_columns]
    #out_all  = data.loc[:, output_columns]
    
    ## Setup random values
    np.random.seed(seed)
    
    if split_type == 'simple_split' or split_type == 'oob':
    
        ## Split train test
        set_train_test(data, split=split, seed=seed, split_by_event=split_by_event)
        
        ## Select train/test data
        data_train = data.loc[data['type'] == 'train', input_columns]
        out_train  = data.loc[data['type'] == 'train', output_columns]
        data_test = data.loc[data['type'] == 'test', input_columns]
        out_test  = data.loc[data['type'] == 'test', output_columns]
            
    elif split_type == 'kfold':
    
        sys.exit('Not implemented')
        from sklearn.model_selection import StratifiedKFold
        
        Nsplits = 5
        cv = StratifiedKFold(n_splits=Nsplits, random_state=seed)
                
    else:
        sys.exit('Split type not recognized!')
        
    return data, data_train, out_train, data_test, out_test

## TODO: RFECV was tested by hand. Implement it in a better way
def find_recursive_unknowns(est, data_train, out_train):

    from sklearn.feature_selection import RFECV
    rfe = RFECV(estimator=est)
    rfe.fit(data_train.values, out_train.values[:,0])
    bp()
    best_columns = data_train.loc[:, ~data_train.columns.isin( data_train.columns[rfe.support_].tolist() )].columns

    return rfe, best_columns

def create_one_classifier(seed, split_type, CNN_input_shape, 
                          type_ML='forest', nb_trees=1000, max_depth=None, 
                          oob_score=True, bootstrap=True, class_weight={'noise': 1, 'arrival': 1}, 
                          hidden_layer_sizes=(64, 64, 64), learning_rate_init=0.05, early_stopping=True, max_iter=10000):
    
    """
    Create a ML classifier (RF, MLP, or CNN)
    """
    
    if type_ML == 'MLP':
        print("Training MLPRegressor...")
        MLP = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                           learning_rate_init=learning_rate_init, 
                           early_stopping=early_stopping, 
                           max_iter=max_iter, random_state=seed)
        est = make_pipeline(StandardScaler(), MLP)
        
    elif type_ML == 'forest':
        
        print("Training Random Forest...")
        ## Prepare the classifier
        if split_type == 'oob':
            est = ExtraTreesClassifier(n_estimators=nb_trees, random_state=seed, 
                                       oob_score=oob_score, bootstrap=bootstrap, 
                                       class_weight=class_weight, max_depth=max_depth)
        else:
            est = ExtraTreesClassifier(n_estimators=nb_trees, random_state=seed)
    
    elif type_ML == 'CNN':
    
        print("Training CNN...")
        est = create_CNN(input_shape=CNN_input_shape, Nclasses=2, loss='binary_crossentropy')
        
    return est

def two_steps_training(est, data):
    
    ## Find inputs/outputs
    input_columns  = [key for key in data_without_info_columns(data)]
    output_columns = ['type-data']
    
    ## Find initial training and testing data
    data_train = data.loc[data['type'] == 'train', input_columns]
    out_train  = data.loc[data['type'] == 'train', output_columns]
    data_test = data.loc[(data['type'] == 'test') & ~data['extra_test'], input_columns]
    out_test  = data.loc[(data['type'] == 'test') & ~data['extra_test'], output_columns]
    
    est_baseline = clone(est)
    est_baseline.fit(data_train.values, out_train.values[:,0])
    
    ## Train a first estimator to find false positives
    est_step_one = clone(est)
    est_step_one.fit(data_train.values, out_train.values[:,0])
    out_pred  = est_step_one.predict(data_test)
    out_pred_train = est_step_one.predict(data_train)
    
    ## Encode outputs
    le = preprocessing.LabelEncoder()
    le.fit(out_test.values)
    out_test_encoded = le.transform(out_test.values)
    out_pred_encoded = le.transform(out_pred)
    out_train_encoded = le.transform(out_train.values)
    out_pred_train_encoded = le.transform(out_pred_train)

    ## Find false positives and false negatives
    ## Arrival = 0; Noise = 1
    idx_FP = np.where(out_test_encoded-out_pred_encoded > 0)[0]
    idx_FP = data_test.iloc[idx_FP].index.tolist()
    #idx_FP_train = np.where(out_train_encoded-out_pred_train_encoded > 0)[0]
    #idx_FP_train = data_train.iloc[idx_FP_train].index.tolist()
    #idx_FP += idx_FP_train
    idx_FN = np.where(out_test_encoded-out_pred_encoded < 0)[0]
    idx_FN = data_test.iloc[idx_FN].index.tolist()
    #idx_FN_train = np.where(out_train_encoded-out_pred_train_encoded < 0)[0]
    #idx_FN_train = data_train.iloc[idx_FN_train].index.tolist()
    #idx_FN += idx_FN_train
    
    ## Find correctly identifies phases in training dataset
    idx_TN = np.where((out_train_encoded-out_pred_train_encoded == 0) & (out_train_encoded == 1))[0]
    idx_TN = data_train.iloc[idx_TN].index.tolist()
    idx_TP = np.where((out_train_encoded-out_pred_train_encoded == 0) & (out_train_encoded == 0))[0]
    idx_TP = data_train.iloc[idx_TP].index.tolist()
    
    ## Find indexes of noise waveforms to remove from training dataset
    nb_arrivals = data.loc[(data['type-data'] == 'arrival'), :].shape[0]
    nb_training_arrivals = data.loc[(data['type-data'] == 'arrival') & (data['type'] == 'train'), :].shape[0]
    #idx_test_to_remove = data.loc[(data['type-data'] == 'noise') & (data['type'] == 'train'), :].index.tolist()
    #np.random.shuffle(idx_test_to_remove)
    idx_test_to_remove = idx_TN + idx_TP
    idx_test_to_remove = idx_test_to_remove[:min(nb_arrivals, len(idx_FP) + len(idx_FN))]
    idx_to_keep  = data.loc[(~data.index.isin(idx_test_to_remove)) & (data['type'] == 'train'), :].index.tolist()
    to_keep_loc = idx_FP + idx_FN
    np.random.shuffle(to_keep_loc)
    if nb_training_arrivals-len(idx_to_keep) > 0:
        idx_to_keep += to_keep_loc[:nb_training_arrivals-len(idx_to_keep)]
    
    data_new = data.copy()
    data_new['type'] = 'test'
    data_new.loc[data_new.index.isin(idx_to_keep), 'type']  = 'train'
    
    ## Reload training and testing data
    data_train_new = data_new.loc[data_new['type'] == 'train', input_columns]
    out_train_new = data_new.loc[data_new['type'] == 'train', output_columns]
    #data_test_new = data_new.loc[data_new['type'] == 'test', input_columns]
    #out_test_new  = data_new.loc[data_new['type'] == 'test', output_columns]
    
    ## Fit final estimator
    est.fit(data_train_new.values, out_train_new.values[:,0])
    
    ## Update data with new train/test distribution
    data.loc[:, 'type'] = data_new.type
    """
    est = ExtraTreesClassifier(n_estimators=1000, random_state=1, oob_score=True, bootstrap=True)
    out_pred_new  = est_step_one.predict(data_test_new); out_test_encoded_new = le.transform(out_test_new.values); out_pred_encoded_new = le.transform(out_pred_new)
    
    
    idx_FP2 = np.where(out_test_encoded_new-out_pred_encoded_new > 0)[0]
    idx_FP2 = data_test_new.iloc[idx_FP2].index.tolist()
    idx_FN2 = np.where(out_test_encoded_new-out_pred_encoded_new < 0)[0]
    
    est_step_one = ExtraTreesClassifier(n_estimators=1000, random_state=1, oob_score=True, bootstrap=True)
    est_step_one.fit(data_train.values, out_train.values[:,0])
    """
    
    return est

def create_classifier_and_train(data, data_train, out_train, data_test, out_test, 
                                split_type, seed, type_ML = 'forest', nb_trees=1000,
                                max_depth=None, oob_score=True, bootstrap=True,
                                SMOTE=True, class_weight={'noise': 1, 'arrival': 1},
                                hidden_layer_sizes=(64, 64, 64), learning_rate_init=0.05, 
                                early_stopping=True, max_iter=10000, two_steps=True):

    """
    Import right classifier library, and train
    """
    
    ## TODO: FIX KFOLD
    CNN_input_shape = None
    if type_ML == 'CNN':
        CNN_input_shape = data_train.iloc[0]['spectro'].shape + (1,)
    est = create_one_classifier(seed, split_type, CNN_input_shape, 
                          type_ML=type_ML, nb_trees=nb_trees, max_depth=max_depth, 
                          oob_score=oob_score, bootstrap=bootstrap, class_weight=class_weight, 
                          hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, 
                          early_stopping=early_stopping, max_iter=max_iter)

    ## Recursively remove input features
    #rfe, best_columns = find_recursive_unknowns(est, data_train, out_train)
    
    ## Train RF or MLP
    if split_type == 'kfold':
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (train, test) in enumerate(cv.split(data_all, out_all)): 
            data_train = data_all.loc[train]
            out_train  = out_all.loc[train]
            est.fit(data_train.values, out_train.values)
            
            data_test = data_all.loc[test]
            out_test  = out_all.loc[test]
            out_pred = est.predict(data_test)
            out_pred_proba = est.predict_proba(data_test)[:, 1]
                
    else:
        if type_ML == 'MLP':
            le = build_label_encoder(out_train)
            out_train = le.transform(out_train.values)
            est.fit(data_train.values, out_train)
        elif type_ML == 'forest':
            if two_steps:
                est = two_steps_training(est, data)
            else:
                est.fit(data_train.values, out_train.values[:,0])
            
        elif type_ML == 'CNN':
            train_CNN(est, data_train, out_train, data_test, out_test, epochs=3)
            
    return est

def reshape_dataframe_spectrograms(data_all, input_columns, stack=False):

    """
    Return a numpy float array from a dataframe of 2d arrays
    """

    X = data_all[input_columns[0]].values
    size_matrix = X[0].shape[0] * X[0].shape[1]
    X = np.stack(X, axis=0)
    if stack:
        X = X.reshape(X.shape[0], size_matrix)
    else:
        X = X.reshape(X.shape+(1,))
    
    return X

def compute_PCA(data_all, input_columns, output_columns, n_components = 2, max_elmt=500):

    """
    Compute PCA for an input feature dataset
    """

    ## Collect inputs and outputs, and scale
    X = data_all[input_columns].values[:max_elmt]
    
    ## if only one input column, it means that it is a spectrogram
    if len(input_columns) == 1: 
        X = reshape_dataframe_spectrograms(data_all, input_columns, stack=True)
    
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    y = data_all[output_columns].values[:max_elmt,0]
    
    ## Compute PCA up to n_components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, y

def correct_event_name(x):
    
    """
    Remove sampling from event name
    """
    
    x['event_name'] = x['event'].split('_')[0]
    return x

def plot_clustering(data_all_in, input_columns, output_columns, options, n_components = 2, perplexity=50, max_elmt=500):
    
    """
    Plot results of clustering analysis using PCA and TSNE. 
    """
    
    data_all = data_all_in.copy()
    data_all = data_all.apply(correct_event_name, axis=1)
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
    plot_PCA_analysis(axs[:,0], data_all, input_columns, output_columns, n_components = n_components, max_elmt=max_elmt)
    axs[0, 0].text(-0.1, 1.05, 'a)', ha='right', va='bottom', transform=axs[0, 0].transAxes, 
            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=15., fontweight='bold')
    axs[1, 0].text(-0.1, 1.05, 'c)', ha='right', va='bottom', transform=axs[1, 0].transAxes, 
            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=15., fontweight='bold')    
    
    plot_TSNE_analysis(axs[:,1], data_all, input_columns, output_columns, n_components = n_components, perplexity=perplexity, max_elmt=max_elmt)
    axs[0, 1].text(-0.1, 1.05, 'b)', ha='right', va='bottom', transform=axs[0, 1].transAxes, 
            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=15., fontweight='bold')
    axs[1, 1].text(-0.1, 1.05, 'd)', ha='right', va='bottom', transform=axs[1, 1].transAxes, 
            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=15., fontweight='bold')
    
    fig.subplots_adjust(wspace=0.3, right=0.8)
        
    plt.savefig(options['DIR_FIGURES'] + 'clustering_analysis_'+read_data.get_str_options(options)+'.pdf')
    
def plot_PCA_analysis(axs, data_all, input_columns, output_columns, n_components = 2, max_elmt=500, fontsize=15.):
    
    """
    Plot PCA of input features to show high-dimensional data in 2d/3d
    """
    
    X_pca, y = compute_PCA(data_all, input_columns, output_columns, 
                        n_components=n_components, max_elmt=max_elmt)

    labels = ['noise']
    colors = [sns.color_palette()[-1]]
    
    selected_data = data_all.iloc[:max_elmt]
    list_events = selected_data.event.unique().tolist()
    colors_classes = [sns.color_palette()[1], sns.color_palette()[-1]]
    labels_classes = labels + ['arrival']
    colors_events = sns.color_palette("rocket", n_colors=len(list_events))
    labels_events = labels + list_events
    
    ## Plot noise and events for 2 PCA components
    for color, label in zip(colors_classes, labels_classes):
        y_ = (y == label)
        axs[0].scatter(X_pca[y_, 0], X_pca[y_, 1],
                    color=color, lw=2, label=label)

   ## Plot only events with different colours for 2 PCA components
    for color, label in zip(colors_events, labels_events):
        y_ = (y == label)
        if not label == 'noise':
            y_ = (selected_data.event == label).values
        else:
            continue
        axs[1].scatter(X_pca[y_, 0], X_pca[y_, 1],
                    color=color, lw=2, label=label)

    axs[-1].set_xlabel('PC1', fontsize=fontsize)
    axs[-1].set_ylabel('PC2', fontsize=fontsize)
    axs[0].set_title('Principal\nComponent Analysis (PCA)', fontsize=fontsize)
    #ax.legend(loc="best", shadow=False, scatterpoints=1)

def plot_TSNE_analysis(axs, data_all, input_columns, output_columns, n_components = 2, perplexity=50, max_elmt=500, fontsize=15.):
    
    """
    Plot TSNE of input features to show high-dimensional data in 2d/3d
    """
    
    ## Collect inputs and outputs, and scale
    X = data_all[input_columns].values[:max_elmt]
    
    ## if only one input column, it means that it is a spectrogram
    if len(input_columns) == 1: 
        X = reshape_dataframe_spectrograms(data_all, input_columns, stack=True)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data_all[output_columns].values[:max_elmt,0]
    
    ## Compute PCA up to n_components
    pca = TSNE(perplexity=perplexity, n_components=n_components)
    X_pca = pca.fit_transform(X)

    selected_data = data_all.iloc[:max_elmt]
    list_events = selected_data.event_name.unique().tolist()
    labels_classes = ['noise', 'arrival']
    colors_classes = [sns.color_palette()[1], sns.color_palette()[-1]]
    labels_events  = list_events
    colors_events = sns.color_palette("rocket", n_colors=len(list_events))
    
    ## Plot noise and events for 2 PCA components
    for color, label in zip(colors_classes, labels_classes):
        y_ = (y == label)
        axs[0].scatter(X_pca[y_, 0], X_pca[y_, 1],
                    color=color, lw=2, label=label)
    
    ## Plot only events for 2 PCA components
    for color, label in zip(colors_events, labels_events):
        y_ = (y == label)
        if not label == 'noise':
            y_ = (selected_data.event_name == label).values
        else:
            continue
        axs[1].scatter(X_pca[y_, 0], X_pca[y_, 1],
                    color=color, lw=2, label=label)
    
    axs[-1].set_xlabel('TC1', fontsize=fontsize)
    axs[-1].set_ylabel('TC2', fontsize=fontsize)
    axs[0].set_title('T-Distributed Stochastic\nNeighbor Embedding (TSNE)', fontsize=fontsize)
    axs[-1].legend(loc="lower left", shadow=False, scatterpoints=1, bbox_to_anchor=(1,0), 
                   bbox_transform=axs[-1].transAxes)
    axs[0].legend(loc="lower left", shadow=False, scatterpoints=1, bbox_to_anchor=(1,0), 
                   bbox_transform=axs[0].transAxes)

def compute_performance_one_model(est, data):

    """
    Compute performance metrics using sklearn classification_report routine 
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """

    ## Choose input and output data
    input_columns  = [key for key in data_without_info_columns(data)]
    output_columns = ['type-data']
    
    ## Extract test data
    data_test = data.loc[data['type'] == 'test', input_columns]
    out_test  = data.loc[data['type'] == 'test', output_columns]
    out_pred  = est.predict(data_test)
    
    ## Encode outputs
    le = preprocessing.LabelEncoder()
    le.fit(out_test.values)
    out_test_encoded = le.transform(out_test.values)
    out_pred_encoded = le.transform(out_pred)

    ## Compute accuracy metrics
    target_names = ['noise', 'arrival']
    report = classification_report(out_test_encoded, out_pred_encoded, 
                                   target_names=target_names, output_dict=True)
    
    return report

def data_without_info_columns(data):
    
    """
    Return waveform dataframe without information columns not used for training
    """

    labels = ['type-data', 'event', 'event_corrected', 'satellite', 'station', 
              'type', 'sampling', 'only_for_testing', 'arrival-time', 
              'extra_test', 'event_to_testing', 'snr']
    
    return data.loc[:, ~data.columns.isin(labels)]

def return_data_for_ML(data):
    """
    return waveform data without testing-only (i.e., without param file) entries
    """
    data_for_ML = data.copy()
    if 'only_for_testing' in data.columns:
        data_for_ML = data_for_ML.loc[~data_for_ML['only_for_testing'],:]
    
    return data_for_ML

def find_new_distribution(data_for_ML, max_proporption_one_class=0.3):

    """
    Find the number of elements to keep in each class to get balanced event inputs
    """
    
    counts_per_event = data_for_ML.groupby(['event_corrected', 'type-data'])['station'].count().reset_index()
    counts_per_event['event'] = counts_per_event['event_corrected']
    counts_per_event['count'] = counts_per_event['station']
    counts_arrival = counts_per_event.loc[counts_per_event['type-data']=='arrival',:]
    
    counts_arrival['proportion'] = counts_arrival['count']/counts_arrival['count'].sum()
    while counts_arrival['proportion'].max() > max_proporption_one_class:
        id_max = counts_arrival['count'].idxmax()
        counts_arrival.loc[counts_arrival.index==id_max, 'count'] -= 1
        counts_arrival['proportion'] = counts_arrival['count']/counts_arrival['count'].sum()
        
    return counts_arrival

def fix_unbalanced_events(data_for_ML, max_proporption_one_class=0.3, exclude_events_from_training=[]):

    """
    Remove input waveforms for events that have a proportion > max_proporption_one_class in the whole dataset
    """
    
    data_for_ML.reset_index(drop=True, inplace=True)
    #data_for_ML['flag-keep'] = True
    data_for_ML['event_to_testing'] = False
    
    data_for_ML['event_corrected'] = 'N/A'
    grouped_data_for_ML = data_for_ML.groupby('event')
    for event, one_event in grouped_data_for_ML:
        name = event.split('_')[0]
        data_for_ML.loc[data_for_ML.index.isin(one_event.index), 'event_corrected'] = name
        
    ## First remove events from training set that have been flagged
    data_for_ML.loc[data_for_ML.event.isin(exclude_events_from_training), 'event_to_testing'] = True
    
    ## Second fix unbalanced events
    if data_for_ML.event.unique().size > 2:

        counts_arrival = find_new_distribution(data_for_ML, max_proporption_one_class=max_proporption_one_class)
        counts_arrival = counts_arrival.loc[~(counts_arrival['count'] == counts_arrival['station']), :]
        
        for ievent, event_row in counts_arrival.iterrows():
            new_count = event_row['count']
            event = event_row.event
            l_index = data_for_ML.loc[(data_for_ML.event_corrected == event) & (data_for_ML['type-data'] == 'arrival'), :].index.tolist()
            np.random.shuffle(l_index)
            data_for_ML.loc[data_for_ML.index.isin(l_index[new_count:]), 'event_to_testing'] = True
            #data_for_ML.loc[data_for_ML.index.isin(l_index[new_count:]), 'flag-keep'] = False
    #return data_for_ML.loc[data_for_ML['flag-keep'], ~(data_for_ML.columns.isin(['flag-keep']))]
    
def plot_input_distribution(data_cols, options, col_wrap=6, height=1., aspect=1.):

    """
    Plot input distribution for each input feature
    """

    collapsed_cols = pd.DataFrame()
    for col in data_cols.columns.tolist(): 
        loc_df=data_cols[[col]]; 
        loc_df.columns=['val']; 
        loc_df['index']=col; 
        collapsed_cols=collapsed_cols.append(loc_df);
    grid = sns.FacetGrid(collapsed_cols, col="index", hue='index', 
                         col_wrap=col_wrap, height=height, aspect=aspect, 
                         sharex=False, sharey=False, palette='rocket'); 
    grid.set(xticks=[], yticks=[],xlabel = '', ylabel = '');
    grid.map(sns.kdeplot, "val", fill=True)
    grid.set_titles("{col_name}")
    
    plt.savefig(options['DIR_FIGURES'] + 'input_distribution_'+read_data.get_str_options(options)+'.pdf')

def preprocess_dataset(data_in, tec_data_param, options, seed, 
                       split_type = 'oob', split=0.9, columns_to_remove=[], 
                       type='features', threshold=0.7, unknowns_for_outliers = {'W7': 3, 'W9': 2, 'W10': 3, 'W11': 3},
                        n_components = 2, perplexity=50, max_elmt=1500, twosteps=True, 
                        priority_only_noise_waveforms=True, exclude_events_from_training=[],
                        split_by_event=False, max_proporption_one_class=0.3):

    """
    Preprocess waveform dataset: 
        1) remove correlated inputs
        2) Fix unbalanced classes
        3) Split between training and testing datasets
        4) Plot input-feature distributions and clusters
    """
    
    ## Find long name for each input parameter
    label_columns = compute_params_waveform.setup_name_parameters(options)
    
    ## Store a local copy for modification
    data = data_in.copy()
    data.reset_index(inplace=True, drop=True)
    
    ## Only reprocess data if no loading request
    if not 'data' in options['load'].keys():
    
        ## Remove data from same event
        events_to_process = ['Tohoku', 'Sanriku']
        for event in events_to_process:
            stations_tohoku_1s = data.loc[(data.event == event + '_1s'), 'station'].unique()
            data = data.loc[~( (data.event == event + '_30s') & data.station.isin(stations_tohoku_1s) )]
        
        ## Remove outliers
        data = remove_outliers(data, unknowns_for_outliers)
        
        ## Remove nan values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(axis='columns', inplace=True)
        
        ## Remove correlated inputs
        if type == 'features':
            data = remove_correlated_inputs(data, label_columns, type_corr='spearman', 
                                        threshold=threshold, columns_to_remove=columns_to_remove)
        
        # Remove testing-only data
        #data_for_ML = return_data_for_ML(data)
        #data_for_ML = data
        
        ## Fix unbalanced number of waveforms per events
        #data_for_ML = fix_unbalanced_events(data)
        fix_unbalanced_events(data, exclude_events_from_training=[], max_proporption_one_class=max_proporption_one_class)
        
        ## Fix unbalanced classes
        #data_for_ML = fix_unbalanced_classes(data_for_ML, tec_data_param, factor_major_class=1, 
        #                                     priority_only_noise_waveforms=priority_only_noise_waveforms)
        data_for_ML = fix_unbalanced_classes(data, tec_data_param, factor_major_class=1, 
                                             priority_only_noise_waveforms=priority_only_noise_waveforms)
        
    else:
        data_for_ML = data_in
        
    ## Choose input and output data
    input_columns  = [key for key in data_without_info_columns(data_for_ML)]
    output_columns = ['type-data']
    
    ## Split between test and training sets
    data, data_train, out_train, data_test, out_test = \
        split_data_before_training(data_for_ML, input_columns, output_columns, seed, 
                                   split_type, split=split, twosteps=twosteps,
                                   split_by_event=split_by_event)
    
    #bp()
    
    if type == 'features':
    
        #unknowns_for_outliers = {'W7': 3, 'W9': 1, 'W10': 3, 'W11': 3}
        #data_ = remove_outliers(data_for_ML, unknowns_for_outliers)
        ## Plot input distribution
        plot_input_distribution(data_for_ML[input_columns], options)
        ## Plot cumulative distributions for each parameter
        plot_cum_distribution(data_for_ML[input_columns], label_columns, options)
        ## Plot correlations
        plot_statistics(data_for_ML, input_columns, options)
    
    ## Plot clusters
    plot_clustering(data_for_ML, input_columns, output_columns, options, n_components=n_components, perplexity=perplexity, max_elmt=max_elmt)
    
    return data, data_train, out_train, data_test, out_test, input_columns, output_columns

def train_machine(features_pd, tec_data_param, options, seed=123, columns_to_remove=[], 
                  plot_performance=True, plot_best_features=True,
                  split_type = 'oob', split=0.9, unknowns_for_outliers = {'W7': 3., 'FT2': 2000.},
                  threshold=0.7, n_components = 2, perplexity=50, max_elmt=1500,
                  priority_only_noise_waveforms=True, save_estimator=True, two_steps=True,
                  exclude_events_from_training=[], split_by_event=False, max_proporption_one_class=0.3):

    """
    Train a classifier over a given dataset:
        1) preprocess dataset
        2) create classifier
        3) train classifier
        4) plot results
    """
    
    ## Load existing preprocessed data
    data = features_pd
    if 'data' in options['load'].keys():    
        data = load_data_forest(options)

    ## Preprocess data
    type_input = options['type_input']
    type_ML    = options['type_ML']

    ## Prevent tensorflow to use CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    ## Preprocess dataset
    data, data_train, out_train, data_test, out_test, input_columns, output_columns = \
        preprocess_dataset(data, tec_data_param, options, seed, split_type = split_type, split=split,
                           columns_to_remove=columns_to_remove, type=type_input,
                           threshold=threshold, unknowns_for_outliers=unknowns_for_outliers,
                           n_components=n_components, perplexity=perplexity, max_elmt=max_elmt,
                           priority_only_noise_waveforms=priority_only_noise_waveforms,
                           exclude_events_from_training=exclude_events_from_training,
                           split_by_event=split_by_event, max_proporption_one_class=max_proporption_one_class)
        
    if not 'data' in options['load'].keys():    
        ## Save data
        name_f = options['DIR_FIGURES'] + 'data_' + read_data.get_str_options(options) + '.pkl'
        with open(name_f, 'wb') as f:
            pickle.dump(data, f)
        
    ## Load existing estimator
    if 'est' in options['load'].keys():
        est = load_est_forest(options)
    
    ## Train classifier    
    else:    
        est = create_classifier_and_train(data, data_train, out_train, data_test, out_test, 
                                          split_type, seed, type_ML = type_ML,
                                          class_weight=options['class_weight'], two_steps=two_steps)
        
        """
        X, y = compute_PCA(data, input_columns, output_columns, n_components = 2, max_elmt=1500)
        unknown = 'FT2'
        data['zscore'] = abs(data[unknown] - data[unknown].mean()/data[unknown].std())
        data_ = remove_outliers(data, ['FT2'], zscore_threshold = 3.)    
        plot_clustering(data_, input_columns, output_columns, options, n_components=2, perplexity=50, max_elmt=1500)
        """
        
        ## Save classifier
        if save_estimator:
            name_f = options['DIR_FIGURES'] + type_ML + '_est_' + read_data.get_str_options(options) + '.pkl'
            if type_ML == 'CNN':
                est.save(options['DIR_FIGURES']+ type_ML + '_est_' + read_data.get_str_options(options))
            else:
                with open(name_f, 'wb') as f:
                    pickle.dump(est, f)
        #joblib.dump(est, options['DIR_FIGURES'] + 'random_est.joblib')
        
        ## Plot performances on testing dataset
        if plot_performance:
            if type_ML == 'forest':
                plot_RF_importance(est, input_columns, options)
            
            """
            data_test_all = data.loc[(data['type'] == 'test') & ~data.event_to_testing & ~data.extra_test & (data['type-data'] == 'arrival'), :]; l_idx = data_test_all.index.tolist(); np.random.shuffle(l_idx); data_test_all = data_test_all.loc[data_test_all.index.isin(l_idx[:data_train.shape[0]//8])]
            data_noise = data.loc[(data['type'] == 'test') & ~data.extra_test & (data['type-data'] == 'noise'), :]; l_idx = data_noise.index.tolist(); np.random.shuffle(l_idx); data_test_all = data_test_all.append( data_noise.loc[data_noise.index.isin(l_idx[:data_test_all.shape[0]])] )
            data_test = data_test_all[input_columns]; out_test  = data_test_all[output_columns]
            """
            
            ## Select right testing dataset
            #data_test=data.loc[(data.snr==-1) & (data['type'] == 'test') & ~data.event_to_testing & ~data.extra_test, input_columns]; out_test=data.loc[(data.snr==-1) & (data['type'] == 'test') & ~data.event_to_testing & ~data.extra_test, output_columns]
            data_test=data.loc[(data['type'] == 'test') & ~data.event_to_testing & ~data.extra_test, input_columns]; out_test=data.loc[(data['type'] == 'test') & ~data.event_to_testing & ~data.extra_test, output_columns]
            #data.loc[(data['type-data'] == 'arrival') & ~data.event_to_testing & ~data.extra_test]
            
            plot_confusion(est, data_test, out_test, options, type_ML=type_ML)
            plot_roc_curve(est, data_test, out_test, options, xlim=[0., 1.], ylim=[0.8, 1.], type_ML=type_ML)
        
        ## Plot (most important) features against features
        if type_ML == 'forest' and plot_best_features:
            data_for_ML = return_data_for_ML(data)
            plot_features_vs_features(data_for_ML, input_columns, est, options, Nfeatures = 3)
        
        """
        import utils_paper; from importlib import reload; reload(utils_paper)
        data_=data.loc[~data.extra_test, :]
        bp()
        FN, FP, TN, TP = utils_paper.get_FTP_FTN(data_, est)
        #FP_=FP.loc[FP.type=='train']; FN_=FN.loc[FN.type=='train']
        bp()
        lw = utils_paper.show_FP_FN(tec_data, FP, FN, 720., options, type_input='F', nb_waveforms=40, seed=0, plot_per_col=10, axs=[])
        """
        
    ## Optimize RF   
    """
    l_nb_trees  = np.arange(200, 1300, 200).tolist()
    l_max_depth = np.arange(10, 120, 20).tolist()
    scores = optimize_RF(data, data_train, out_train, data_test, out_test, seed, split_type,
                l_nb_trees=l_nb_trees, l_max_depth=l_max_depth, 
                oob_score=True, bootstrap=True,
                class_weight={'noise': 1, 'arrival': 1},
                hidden_layer_sizes=(64, 64, 64), learning_rate_init=0.05, 
                early_stopping=True, max_iter=10000)
    bp()
    #scores = optimize_RF(data, data_train, out_train, data_test, out_test, seed, split_type,l_nb_trees=l_nb_trees, l_max_depth=l_max_depth, oob_score=True, bootstrap=True,class_weight={'noise': 1, 'arrival': 1},hidden_layer_sizes=(64, 64, 64), learning_rate_init=0.05, early_stopping=True, max_iter=10000)
        
    scores_all = pd.DataFrame()
    remove_cols = ['max_depth', 'nb_trees', 'macro avg-support', 'arrival-support','weighted avg-support', 'noise-support']
    for col in scores.loc[:, ~scores.columns.isin(remove_cols)].columns: score_ = scores[['max_depth', 'nb_trees']]; score_['metric']=col; score_['val'] = scores[col]; scores_all = scores_all.append( score_ );
    g = sns.FacetGrid(scores_all, col="metric", col_wrap=4)
    g.map_dataframe(test_module.draw_heatmap, "max_depth", "nb_trees", 'val')
    """
    
    return est, data

def load_data_forest(options):

    """
    Load pickled training data
    """

    name_f = options['load']['data']
    with open(name_f, 'rb') as f:
        data = pickle.load(f)
    
    return data

def load_est_forest(options):

    """
    Load pickled ML estimator
    """

    if options['type_ML'] == 'CNN':
        from tensorflow import keras
        est = keras.models.load_model(options['load']['est'])
        
    else:
        est = joblib.load(options['load']['est'])

    return est