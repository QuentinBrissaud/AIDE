#!/usr/bin/env python3
import numpy as np
from pdb import set_trace as bp
import pandas as pd
import os
import obspy
import pickle
import joblib
import time
            
import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import ListedColormap

from obspy.core.utcdatetime import UTCDateTime

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from obspy.signal.trigger import classic_sta_lta

import compute_params_waveform, read_data, train_est, utils_paper

def extract_timeseries_from_data(tec_data, tec_data_param, data, window=720.):

    data_picker = pd.DataFrame()

    grouped_data = data.loc[data['type-data'] == 'arrival', :].groupby(['event', 'satellite', 'station'])
    for group, input in grouped_data:
        
        time_begin_window  = input['arrival-time'].iloc[0]
        time_center_window = time_begin_window + window*0.5
        
        event, satellite, station = group
        params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                        & (tec_data_param['satellite'] == satellite) 
                        & (tec_data_param['event'] == event), : ].iloc[0]

        if time_begin_window > params['arrival-time']:
            continue

        waveform = read_data.get_one_entry(tec_data, station, satellite, event)
        waveform = waveform.loc[(waveform.time_s >= time_begin_window) 
                                & (waveform.time_s <= time_begin_window + window), :]

        one_data_picker = waveform.vTEC.to_dict()
        one_data_picker['shift_from_center'] = time_center_window - params['arrival-time']
        
        data_picker = data_picker.append( [one_data_picker] )
        
    return data_picker
    
def find_random_window_around_arrival(arrival_time, times, window, nb_picks=3, 
                                      max_deviation=0.5, min_distance_time=90.):

    """
    Create random time picks based on window size and max_deviation
    """

    min_time = max(arrival_time - window*0.5 - window*0.5*max_deviation, times.min())
    max_time = min(arrival_time - window*0.5 + window*0.5*max_deviation, times.max())
    available_picks = np.arange(min_time, max_time+min_distance_time, min_distance_time)
    available_picks = available_picks.tolist()
    
    nb_max_picks = min(nb_picks, len(available_picks))
    picks = np.random.choice(available_picks, nb_max_picks, replace=False)
    
    return picks
    
#train_wave_picker.extract_timeseries_from_tec_data_param(tec_data, tec_data_param, bandpass, options, sampling=30., window=720., nb_picks=3, max_deviation=0.5, min_distance_time=30.)
def extract_timeseries_from_tec_data_param(tec_data, tec_data_param, bandpass, options, 
                                           sampling=30., window=720., nb_picks=3, 
                                           max_deviation=0.5, min_distance_time=30.,
                                           seed=1, activate_LTA_STA=False, time_STA=60.,
                                           time_LTA=300.):

    """
    Extract and preprocess vTEC data chunks for training
    """

    ## Either load previously-generate picked data
    if 'features-picker' in options['load']:
        data_picker = pd.read_csv(options['load']['features-picker'], sep=',', header=[0])
    
    ## ... or create new picks
    else:
    
        print('Computing picker features with:')
        print('- nb_picks:', nb_picks)
        print('- sampling:', sampling)
        print('- window:', window)
        print('- max_deviation:', max_deviation)
        print('- min_distance_time:', min_distance_time)

        np.random.seed(seed)
        standard_time_vector = np.arange(0., window+sampling, sampling)

        data_picker = pd.DataFrame()

        grouped_data = tec_data_param.groupby(['event', 'satellite', 'station'])
        for group, input in grouped_data:
            
            print('Processing:', group)
            
            event, satellite, station = group
            params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                            & (tec_data_param['satellite'] == satellite) 
                            & (tec_data_param['event'] == event), : ].iloc[0]
            arrival_time = params['arrival-time']
        
            time_begin_window  = input['arrival-time'].iloc[0]
            time_center_window = time_begin_window + window*0.5

            waveform = read_data.get_one_entry(tec_data, station, satellite, event)
            if waveform.size == 0:
                print('Remove:', group)
                continue
            
            ## We only leave this loop if we found at least one waveform
            at_least_one_done = False
            nb_iter = 0
            while not at_least_one_done and nb_iter < 10:
                picks = find_random_window_around_arrival(arrival_time, waveform.time_s.values, 
                                                          window, nb_picks=nb_picks, 
                                                          max_deviation=max_deviation, 
                                                          min_distance_time=min_distance_time)
                nb_iter += 1
                for pick in picks:
                    time_center_window = pick + window/2.
                    waveform_snippet   = waveform.loc[(waveform.time_s >= pick)
                                                      & (waveform.time_s <= window + pick + 5*sampling), :]
                                                      
                    
                    tr = compute_params_waveform.create_Trace(waveform_snippet.time_s.values,
                                                              waveform_snippet.vTEC.values, 
                                                              detrend=False, bandpass=bandpass, 
                                                              differentiate=True)
                    read_data.downsample_trace(tr, sampling)
                    tr.data = tr.data[:standard_time_vector.size]
                    tr.normalize() 
                    
                    ## Quality check on size on input vTEC
                    if tr.data.size < standard_time_vector.size:
                        print('Remove:', tr.data.size, standard_time_vector.size)
                        continue
                    
                    ## We found one waveform, we can later leave the loop
                    at_least_one_done = True
                    
                    input = tr.data
                    ## Get STA/LTA predictions
                    if activate_LTA_STA:
                        ## Compute STA/LTA cft coefficient
                        df  = tr.stats.sampling_rate
                        cft = classic_sta_lta(tr.data, int(time_STA * df), int(time_LTA * df))
                        time_cft_max = tr.times()[cft.argmax()]
                        cft_max = cft.max()
                        cft = np.array([time_cft_max, cft_max])
                        input = np.concatenate((input, cft))
                    
                    """
                    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False)
                    axs[0].plot(waveform_snippet.time_s.values, waveform_snippet.vTEC.values)
                    axs[0].axvline(params['arrival-time'], color='tab:red')
                    axs[1].plot(tr.times()-tr.times()[-1]/2., cft)
                    axs[1].axvline(params['arrival-time'] - time_center_window, color='tab:red')
                    axs[2].plot(tr.times()-tr.times()[-1]/2., tr.data)
                    axs[2].axvline(params['arrival-time'] - time_center_window, color='tab:red')
                    plt.show()
                    """
                    
                    one_data_picker = pd.Series(data=input)
                    one_data_picker['shift_from_center'] = params['arrival-time'] - time_center_window
                    one_data_picker['event'] = event
                    one_data_picker['satellite'] = satellite
                    one_data_picker['station'] = station
                    one_data_picker['time'] = pick
                    
                    data_picker = data_picker.append( [one_data_picker] )
                
        data_picker.reset_index(drop=True, inplace=True)
        
        if options['save_features_picker']:
            data_picker.to_csv(options['DIR_DATA'] + 'features_picker_w'+str(window)+'_d'+str(max_deviation)+'_STA'+str(activate_LTA_STA)+'.csv', sep=',', header=True, index=False)
        
    return data_picker
    
def preprocess_data(data_picker, split=0.7, split_type='oob', seed=1, balanced_classes=True):

    """
    Preprocess data: split between test and training datasets
    """
    
    output_columns = ['shift_from_center']
    extra_cols     = ['event', 'satellite', 'station', 'type', 'time']
    input_columns  = data_picker.loc[:, ~data_picker.columns.isin(output_columns + extra_cols)].columns.tolist()
    
    data_picker = data_picker.dropna(axis=0)
    data_picker['extra_test'] = False
    
    """
    bp()
    from importlib import reload
    reload(train_est)
    count, data_picker_ = train_est.fix_unbalanced_events(data_picker, max_proporption_one_class=0.3)
    """
    if balanced_classes:
        data_picker['type-data'] = 'arrival'
        train_est.fix_unbalanced_events(data_picker, max_proporption_one_class=0.3)
        data_picker = data_picker.loc[:, ~(data_picker.columns.isin(['type-data']))]
    
    data_picker, data_train, out_train, data_test, out_test = \
            train_est.split_data_before_training(data_picker, input_columns, output_columns, 
                                                 seed, split_type, split=split)
                         
    return data_picker, data_train, out_train, data_test, out_test

def create_1dCNN(size_waveform, CNN_layers, dense_layers, loss_metric="mse", metrics=['mse'], activation='relu'):

    ## import ML librairies
    #from tensorflow.keras.models import Sequential
    #from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D
    #from tensorflow.keras.layers import Activation
    #from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import TimeDistributed
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.layers.experimental import preprocessing
    
    ## define two sets of inputs
    inputA = Input(shape=size_waveform)
    
    ## Data preprocessing
    normalizer_amp = preprocessing.Normalization()
    normalizer_amp.adapt(self.dataset.encoded_input)
    
    ## the first branch operates on the first input
    x = normalizer_amp(inputA)
    for size in CNN_layers:
        x = Conv1D(size[0], size[1], activation=activation)(x)
        #x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

    ## apply a FC layer and then a regression prediction on the combined outputs
    for size in dense_layers:
        x = Dense(size, activation=activation)(x)
        
    ## Output layer
    x = Dense(1, activation=activation_result)(x)

    ## Optimizer
    opt = Adam(lr=learning_rate)#, decay=decay)
    
    ## Compile model
    est = Model(inputs=inputA, outputs=x)
    est.compile(loss=loss_metric, optimizer=opt, metrics=metrics)
    
    return est

def train_1d_CNN(est, data_train, out_train, data_test, out_test, epochs=120, batch_size=16, verbose=True):

    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
            
    ## Fittin and stopping conditions
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=1e-8, cooldown=3)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=12)
    history = est.fit(
        x = data_train, 
        y = out_train,
        validation_data = (data_test, out_test),
        callbacks = [earlystop, reduce_lr],
        epochs = epochs,
        batch_size = batch_size,        
        verbose = verbose
    )
    
    return history

def train_arrival_picker(data_train, out_train, data_test, out_test, seed, type_ML = 'forest', nb_trees=1000,
                         oob_score=True, bootstrap=True, max_depth=100):
    
    """
    Create ML regressor and train it
    """
    
    np.random.seed(seed)
    if type_ML == 'forest':
        est = ExtraTreesRegressor(n_estimators=nb_trees, random_state=seed, oob_score=oob_score, 
                                  bootstrap=bootstrap, n_jobs=-1, max_depth=max_depth)
        est.fit(data_train.values, out_train.values[:,0])
        
    elif type_ML == 'CNN':
        CNN_layers   = [(32, 5), (64, 5)]
        dense_layers = [(32), (32)]
        est = create_1dCNN(data_train.shape[1], CNN_layers, dense_layers, activation='relu')
        history = train_1d_CNN(est, data_train, out_train, data_test, out_test, epochs=120, batch_size=16, verbose=True)

    elif type_ML == 'MLP':
        est = MLPRegressor(hidden_layer_sizes=(20, 80, 20), learning_rate_init=0.001, early_stopping=True, max_iter=10000, random_state=seed)
        #est = make_pipeline(StandardScaler(), MLP)
        est.fit(data_train.values, out_train.values[:,0])
    
    return est
    
def load_picker(options):

    """
    Load pickled picker estimator
    """

    est = joblib.load(options['load']['est-picker'])
            
    return est

def optimize_RF(data_train, out_train, data_test, out_test, seed, 
                l_nb_trees=[1000], l_max_depth=[100], 
                oob_score=True, bootstrap=True):
    
    """
    Optimize RF tree depth and nb of tress
    """
    
    scores = pd.DataFrame()
    for nb_trees in l_nb_trees:
        for max_depth in l_max_depth:
            est = train_arrival_picker(data_train, out_train, data_test, out_test,
                                       seed, type_ML = 'forest', nb_trees=nb_trees,
                                       oob_score=oob_score, bootstrap=bootstrap, max_depth=max_depth)
            preds_training = est.oob_prediction_
            preds_test     = est.predict(data_test)
            error_dict = {
                'est': est,
                'nb_trees': nb_trees,
                'max_depth': max_depth,
                'R2': est.oob_score_,
                'MSE-training': np.mean((preds_training-out_train.values[:,0])**2),
                'MSE-test': np.mean((preds_test-out_test.values[:,0])**2),
                'MAE-training': np.mean(abs(preds_training-out_train.values[:,0])),
                'MAE-test': np.mean(abs(preds_test-out_test.values[:,0])),
            }
            scores = scores.append( [error_dict] )
            #from sklearn import model_selection
            #kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
            #scoring = 'neg_mean_squared_error'
            #results = model_selection.cross_val_score(est, data_picker[input_columns].values, data_picker[output_columns].values[:,0], cv=kfold, scoring=scoring)
            #results_ = model_selection.cross_val_score(est, data_test.values, out_test.values[:,0], cv=kfold, scoring=scoring)
    
    return scores
    
def plot_error_distribution(data_picker, options, xlim=[-400., 400.], ylim=[-300., 300.], 
                            name_ext='', fontsize=15., create_figures=True):

    """
    Plot arrival-time picker error vs shift from center for test and train datasets
    """

    color = sns.color_palette("rocket")[1]
    
    jp_train = sns.jointplot(data=data_picker.loc[data_picker.type=='train', :], x="shift_from_center", y="error", 
                       kind="kde", fill=True, color=color)
    jp_train.ax_joint.set_xlim(xlim)
    jp_train.ax_joint.set_ylim(ylim)
    jp_train.ax_joint.set_yticklabels([])
    print(jp_train.ax_joint.get_xticks(), jp_train.ax_joint.get_yticks())
    if create_figures:
        jp_train.ax_joint.set_xlabel('Time shit from central time (s)', fontsize=fontsize)
        jp_train.ax_joint.set_ylabel('Error (s)', fontsize=fontsize)
        jp_train.fig.savefig(options['DIR_FIGURES'] + 'picker_error_distribution_train'+name_ext+'.pdf')
    else:
        jp_train.ax_joint.set_xlabel('')
        jp_train.ax_joint.set_ylabel('')
        jp_train.ax_joint.text(0.05, 0.95, 'Train', fontsize=fontsize, 
                        ha='left', va='top', transform = jp_train.ax_joint.transAxes)
        jp_train.ax_joint.text(-0.1, 1.05, 'd)', ha='right', va='bottom', 
            transform=jp_train.ax_joint.transAxes, bbox=dict(facecolor='w', edgecolor='w', pad=0.1), 
            fontsize=15., fontweight='bold')
            
    jp_test = sns.jointplot(data=data_picker.loc[data_picker.type=='test', :], x="shift_from_center", y="error",
                       kind="kde", fill=True, color=color)
    jp_test.ax_joint.set_xlabel('Time shit from central time (s)', fontsize=fontsize)
    jp_test.ax_joint.set_ylabel('Error (s)', fontsize=fontsize)
    jp_test.ax_joint.set_xlim(xlim)
    jp_test.ax_joint.set_ylim(ylim)
    print(jp_test.ax_joint.get_xticks(), jp_test.ax_joint.get_yticks())
    jp_test.ax_joint.set_yticklabels(jp_test.ax_joint.get_yticks(), size = fontsize-2.)
    jp_test.ax_joint.set_xticklabels(jp_test.ax_joint.get_xticks()[::2], size = fontsize-2.)
    jp_train.ax_joint.set_xticklabels(jp_train.ax_joint.get_xticks()[::2], size = fontsize-2.)
    if create_figures:
        jp_test.fig.savefig(options['DIR_FIGURES'] + 'picker_error_distribution_test'+name_ext+'.pdf')
    else:
        jp_test.ax_joint.text(0.05, 0.95, 'Test', fontsize=fontsize, 
                        ha='left', va='top', transform = jp_test.ax_joint.transAxes)
        jp_test.ax_joint.text(-0.1, 1.05, 'c)', ha='right', va='bottom', 
            transform=jp_test.ax_joint.transAxes, bbox=dict(facecolor='w', edgecolor='w', pad=0.1), 
            fontsize=15., fontweight='bold')
        
    return jp_test, jp_train
    
def compute_error_distribution(est, data_picker):

    """
    Compute the arrival time error (s) vs shift from center data
    """

    output_columns = ['shift_from_center']
    extra_cols     = ['event', 'satellite', 'station', 'type', 'extra_test', 'event_to_testing', 'event_corrected', 'time']
    input_columns  = data_picker.loc[:, ~data_picker.columns.isin(output_columns + extra_cols)].columns.tolist()
    out_  = data_picker[output_columns]
    data_ = data_picker[input_columns]
    
    ## Compute error over the whole dataset
    preds = est.predict(data_.values)
    data_picker['error'] = out_.values[:,0] - preds
    
#train_wave_picker.create_arrival_picker(tec_data, tec_data_param, bandpass, options, sampling=30., window=720., nb_picks=3, max_deviation=0.5, min_distance_time=30., split=0.7, seed=1, nb_trees=1000, type_ML = 'forest')
def create_arrival_picker(tec_data, tec_data_param, bandpass, options,
                          sampling=30., window=720., nb_picks=3, 
                          max_deviation=0.5, min_distance_time=30.,
                          split=0.8, seed=1, nb_trees=1000, max_depth=100, 
                          type_ML = 'forest', balanced_classes=False, save_est=True,
                          plot_error_distribution=True, activate_LTA_STA=False, 
                          time_STA=60., time_LTA=300):

    """
    Create a picker from scratch:
        1) generate picks or load them
        2) split in training and testing datasets
        3) Create and train regressor
    """

    data_picker = extract_timeseries_from_tec_data_param(tec_data, tec_data_param, bandpass, options, 
                                                         sampling=sampling, window=window, nb_picks=nb_picks, 
                                                         max_deviation=max_deviation, 
                                                         min_distance_time=min_distance_time,
                                                         activate_LTA_STA=activate_LTA_STA, 
                                                         time_STA=time_STA, time_LTA=time_LTA)

    ## Split between training vs testing
    #data_picker, data_train, out_train, data_test, out_test = preprocess_data(data_picker, split=split, seed=seed)
    data_picker, data_train, out_train, data_test, out_test = \
        preprocess_data(data_picker, split=split, seed=seed, balanced_classes=balanced_classes)

    ## Create picker and train
    est = train_arrival_picker(data_train, out_train, data_test, out_test, 
                               seed, type_ML = type_ML, nb_trees=nb_trees, max_depth=max_depth)
    bp()
    
    ## Performance summary for paper
    compute_error_distribution(est, data_picker)
    reports = pd.read_csv(options['DIR_FIGURES'] + 'reports_picker.csv', header=[0], sep=',')
    utils_paper.plot_performance_picker(data_picker, reports, options, metric='RMSE', 
                            xlim=[-400., 400.], ylim=[-300., 300.],
                            name_ext='', fontsize=15.)
    bp()
    
    name_ext = '_w'+str(window)+'_d'+str(max_deviation)+'_b'+str(balanced_classes)+'_STA'+str(activate_LTA_STA)
        
    if plot_error_distribution:
        _, _ = plot_error_distribution(data_picker, options, xlim=[-400., 400.], ylim=[-300., 300.], name_ext=name_ext)
    
    if save_est:
        joblib.dump(est, options['DIR_FIGURES'] + 'random_est_picker'+name_ext+'.joblib')
    
    #import test_module
    #scores = test_module.optimize_RF(data_train, out_train, data_test, out_test, seed, l_nb_trees=[100, 200, 500, 1000], l_max_depth=[10, 50, 100], oob_score=True, bootstrap=True)
    #scores_plot = scores.pivot("max_depth", "nb_trees", "R2")
    
    """
    est = ExtraTreesRegressor(n_estimators=1000, random_state=seed, oob_score=True, bootstrap=True, n_jobs=-1, max_depth=None)
    output_columns = ['shift_from_center']
    extra_cols     = ['event', 'satellite', 'station', 'type', 'extra_test', 'time']
    input_columns  = data_picker.loc[:, ~data_picker.columns.isin(output_columns + extra_cols)].columns.tolist()
    data_ = data_picker.groupby(['event', 'satellite', 'station']).head(4)
    out_  = data_picker[output_columns]
    data_ = data_picker[input_columns]
    est.fit(data_.values, out_.values[:,0])
    est.fit(data_train.values, out_train.values[:,0])
    
    
    est_  = MLPRegressor(hidden_layer_sizes=(50, 100, 50), learning_rate_init=0.01, early_stopping=True, max_iter=10000, random_state=seed)
    est_.fit(data_train.values, out_train.values[:,0])
    preds = est_.predict(data_test.values)
    error_ = out_test.values[:,0] - preds
    data_plot = data_picker.copy()
    data_plot['error'] = error
    sns.histplot(data_plot, x="error", hue="event", stat="density", element="poly")
    
    joblib.dump(est_, options['DIR_FIGURES'] + 'MLP_est_picker.joblib')
    """
    
    return est, data_picker