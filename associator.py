#!/usr/bin/env python3
import numpy as np
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import pandas as pd
import os
import obspy
import ast
import pickle
import itertools
import seaborn as sns
from matplotlib.collections import QuadMesh
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import string
import time as tm
import copy

from scipy import signal
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.tf_misfit import cwt, plot_tfr, plot_tf_gofs
from sklearn.metrics import roc_curve
from obspy.signal.trigger import classic_sta_lta
from obspy.geodetics.base import degrees2kilometers

import compute_params_waveform, read_data, train_est, train_wave_picker, localization, utils_paper

def add_lat_lon_to_detections(tec_data, detections, window, tec_data_hion=pd.DataFrame(), hion_dict={}, take_true_location=False):
    
    """
    Add ionospheric point coordinate to a given detection list
    """
    
    detections['lat'] = -200
    detections['lon'] = -200
    detections['time-begin-waveform'] = -1
    
    grouped_data = detections.groupby(['event', 'satellite', 'station', 'arrival_class'])
    for group, detection in grouped_data:
    
        #detection = detection.iloc[0]
        event, satellite, station, arrival_class = group
        
        try:
            station = "{:04d}".format(station)
        except:
            pass
            
        found_hion = False
        if tec_data_hion.size > 0:
            if satellite not in hion_dict:
                print('Information about which Hion to use is not provided for satellite ', satellite)
                waveform = pd.DataFrame()
                
            else:
                waveform = \
                    tec_data_hion.loc[(tec_data_hion.event == event) 
                                & (tec_data_hion.satellite == satellite)
                                & (tec_data_hion.station == station)
                                & (tec_data_hion.Hion == hion_dict[satellite]), :]
                
        #waveform = tec_data.loc[(tec_data.event == event) & (tec_data.satellite == satellite) & (tec_data.station == station), :]
        
        #event, satellite, station = 'Tohoku_1s', 'G05', '0030' 
        #tec_data_hion.loc[(tec_data_hion.event == event) & (tec_data_hion.satellite == satellite)& (tec_data_hion.station == station)& (tec_data_hion.Hion == hion_dict[satellite]), :]
        
            if waveform.size == 0:
                print('Can not correct position for ', event, satellite, station)
            else:
                found_hion = True
        
        if not found_hion:
            waveform = \
                tec_data.loc[(tec_data.event == event) & (tec_data.satellite == satellite) & (tec_data.station == station), :]
        
        val_position = detection.iloc[0]['time-corrected']
        if detection.iloc[0]['true-arrival-time'] > -1 and take_true_location:
            print('use_true_location')
            val_position = detection.iloc[0]['true-arrival-time']
            
        waveform_ = waveform.loc[(waveform.time_s <= val_position)]
        time_begin = waveform.iloc[0]['time_s']
        if waveform_.size > 0:
            waveform_ = waveform_.iloc[-1]
            lat = waveform_['lat']
            lon = waveform_['lon']
        else:
            print('bug', group, val_position, waveform.time_s.min(), waveform.time_s.max())
            bp()
            
        detections.loc[detections.index.isin(detection.index), 'lat'] = lat
        detections.loc[detections.index.isin(detection.index), 'lon'] = lon
        detections.loc[detections.index.isin(detection.index), 'time-begin-waveform'] = time_begin
            
        """
        for idetection, detection_ in detection.iterrows():
        
            waveform_ = waveform.loc[(waveform.time_s <= detection_['time-corrected'])]#detection_['time']+window)]
            if waveform_.size > 0:
                waveform_ = waveform_.iloc[-1]
                lat = waveform_['lat']
                lon = waveform_['lon']
                
            detections.loc[detections.index == detection_.name, 'lat'] = lat
            detections.loc[detections.index == detection_.name, 'lon'] = lon
        """
        
def associator(tec_data, detections, probas, options, offset=500., quantile_threshold=0.8, window=720., 
               sampling=30., max_radius_search=500., velocity_search_max=7., velocity_search_min=0.5,
               nb_pts_picker=10, determine_elapsed_time=True, tec_data_hion=pd.DataFrame(), 
               hion_dict={'G26': 180., 'G05': 180.}, save_associations=True, association_name='test'):
    
    """
    Association detected wavetrained based on min/max velocity conditions
    """
    
    if 'associations' in options['load']:
        
        associations_all_events = pd.read_csv(options['load']['associations'], sep=',', header=[0])
        associations_time_steps_all_events = pd.read_csv(options['load']['associations_time_steps'], sep=',', header=[0])
        
    else:
    
        #print(hion_dict)
        add_lat_lon_to_detections(tec_data, detections, window, tec_data_hion=tec_data_hion, hion_dict=hion_dict)
        
        #detections.to_csv('Iquique_full_detections.csv', sep=',', header=True, index=False)
        #detections = pd.read_csv('Tohoku_1s_full_detections_H250.csv', sep=',', header=[0])
        detections.to_csv(association_name + '_withlatlon.csv', sep=',', header=True, index=False)
        #bp()
        #probas.to_csv('Tohoku_1s_full_probas.csv', sep=',', header=True, index=False)
        #detections = pd.read_csv('Tohoku_1s_full_detections.csv', sep=',', header=[0])
        #probas = pd.read_csv('Tohoku_1s_full_probas.csv', sep=',', header=[0])
        #detections = detections.loc[detections.event=='Tohoku_1s']
        detections['association_no'] = -1
        #detections['event'] = 'Iquique_30s'
        #detections['event'] = 'Tohoku_1s'

        ## Time thresholds
        threshold_min = max_radius_search / velocity_search_max
        threshold_max = max_radius_search / velocity_search_min
        
        ## Initialize association dataframes
        associations_all_events = pd.DataFrame()
        associations_time_steps_all_events = pd.DataFrame()
            
        ## Loop over each event
        grouped_data = detections.groupby(['event'])
        for event, detections_event in grouped_data:
        
            waveforms = \
                tec_data.loc[(tec_data.event == event), :]
            
            print(event)
            time_min = waveforms.time_s.min()
            time_max = waveforms.time_s.max()
            times = np.arange(time_min, time_max+sampling, sampling)
            
            #associations = {}
            #associations_reverse = {}
            associations = pd.DataFrame()
            associations_time_steps = pd.DataFrame()
            associations_copy = pd.DataFrame()
            association_no = 0
            
            ## If we want to determine the computational cost of an association
            if determine_elapsed_time:
                time_elapsed = pd.DataFrame()
                time_all = pd.DataFrame()
            
            ## Loop over all available times to mimic a near real time scenario
            for time in times:
            
                ## Only select new detection up to current time
                detections_in_time = detections_event.loc[(detections_event.time+window<=time+0.5)
                                                          & (detections_event['predicted-time'] > -1)]
                #probas_in_time = probas_event.loc[(probas_event.time+window<=time+0.5)
                #                                  & (probas_event['predicted-time'] > -1)]
                
                #detections_in_time = \
                #    train_est.compute_arrival_time(probas_in_time, window, nb_for_class=options['nb_for_class'], 
                #                                   nb_for_end_class=options['nb_for_end_class'])
                
                detections_in_time['count_class'] = \
                    detections_in_time.groupby(['satellite', 'station', 'arrival_class'])['lat'].transform('count')
                
                ## Skip empty detection lists
                if detections_in_time.size == 0:
                    continue
                    
                utils_paper.correct_arrival_times(detections_in_time, offset, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
                
                if determine_elapsed_time:
                    time_start = tm.time()
                
                ## Find new detections at current time
                detections_in_time_new = pd.DataFrame(columns=detections_in_time.columns)
                list_new_detections = \
                    detections_in_time.loc[(detections_in_time.time+window>=time-sampling-0.5)].groupby(['satellite', 'station', 'arrival_class'])
                    
                ## Add all detections belonging to the same arrival class/station/satellite
                for group_new_detect, new_detect in list_new_detections:
                    satellite_new, station_new, arrival_class_new = group_new_detect
                    detections_in_time_new = \
                        detections_in_time_new.append( detections_in_time.loc[(detections_in_time.satellite == satellite_new) 
                                                                            & (detections_in_time.station == station_new) 
                                                                            & (detections_in_time.arrival_class == arrival_class_new)] )
                          
                print('Time: ', time)                   
                #print('detections_in_time: ', detections_in_time_new.shape[0])
                
                if determine_elapsed_time:
                    t1 = tm.time()
                    loc_dict = {'type': 't1', 'cost': t1-time_start}
                    #time_all = time_all.append( [loc_dict] )
                    #print('time t1', t1-time_start)
                
                ## Loop over all station/satellites
                grouped_detections = detections_in_time_new.groupby(['satellite', 'station', 'arrival_class'])
                for group, detection in grouped_detections:
                
                    if determine_elapsed_time:
                        t1_ = tm.time()
                
                    satellite, station, arrival_class = group
                    one_detection = detection.iloc[0]
                    
                    """
                    if np.max(abs(detection['time-corrected'].values-detection['true-arrival-time'].values)) > 1000. \
                        and detection['true-arrival-time'].values.min() > -1 \
                        and detection.shape[0] > 3:
                            bp()
                    """
                    
                    ## If a detection has already been associated, we get its association number
                    list_association_id = associations.index.tolist()
                    id_assoc = -1
                    if one_detection.name in list_association_id:
                        id_assoc = associations.loc[associations.index == one_detection.name, 'association_no'].iloc[0]
                    
                    ## Current coordinates
                    lat, lon = one_detection.lat, one_detection.lon
                    
                    ## Find other detections within search bounds
                    associated_detections = \
                        detections_in_time.loc[(abs(detections_in_time['time-corrected']-one_detection['time-corrected']) \
                                                    <= degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)/velocity_search_min))
                                            & (abs(detections_in_time['time-corrected']-one_detection['time-corrected']) 
                                                    >= degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)/velocity_search_max))
                                            & (degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)) <= max_radius_search)
                                            & ~(detections_in_time.station == one_detection.station)
                                            & ~(detections_in_time.satellite == one_detection.satellite)]
                                          
                    ## Remove associated detections that are already in same association class
                    if associated_detections.loc[associated_detections.index.isin(list_association_id)].size > 0 \
                        and id_assoc > 0:
                        already_in_same_class = associations.loc[associations.index.isin(associated_detections.index) & (associations.association_no == id_assoc)].index.tolist()
                        associated_detections = associated_detections.loc[~associated_detections.index.isin(already_in_same_class)]
                        #print('Remove same class', len(already_in_same_class))
                        
                        #associated_detections = \
                        #    associated_detections.loc[( (associated_detections.index.isin(list_association_id)
                        #                                    & associations.loc[associations.index == associated_detections.index].size > 0), 'association_no']) 
                        #                                | (~associated_detections.index.isin(list_association_id)) )]
                    
                    """
                    detections_in_time.loc[(abs(detections_in_time['time-corrected']-one_detection['time-corrected']) <= degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)/velocity_search_min)) & (abs(detections_in_time['time-corrected']-one_detection['time-corrected']) >= degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)/7.))]
                    
                    detections_in_time.loc[(abs(detections_in_time['time-corrected']-one_detection['time-corrected']) <= degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)/velocity_search_min)) & (abs(detections_in_time['time-corrected']-one_detection['time-corrected']) >= degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)/velocity_search_max)) & (degrees2kilometers(np.sqrt((detections_in_time['lat']-lat)**2 + (detections_in_time['lon']-lon)**2)) <= max_radius_search) & ~(detections_in_time.station == one_detection.station) & ~(detections_in_time.satellite == one_detection.satellite)]
                    """
                    #print(associated_detections)
                    #bp()
                    
                    # Label all detections from current class
                    #associations_reverse[association_no] = []
                    #for idetection_class, detection_class in detection.iterrows():
                    #    associations[idetection_class] = association_no
                    #    associations_reverse[association_no].append( idetection_class )
                    
                    ## Find detected windows in current arrival class that are already in association list
                    overlap_detection_associations = set(detection.index.tolist()) & set(list_association_id)
                    overlap_detection_associations = [*overlap_detection_associations,]
                    #if not one_detection.name in associations.index.tolist():
                    """
                    ## If none are found, we 
                    if not overlap_detection_associations:
                        detection['association_no'] = association_no
                        associations = associations.append( detection )
                    else:
                    """
                    
                    ## TODO: remove
                    #if associations.shape[0] > 0:
                    #    bp()
                    
                    ## Update association number if wavetrain not in association list yet
                    ## Otherwise use already existing one
                    new_association_no = association_no
                    if len(overlap_detection_associations) > 0:
                        new_association_no = \
                            associations.loc[associations.index == overlap_detection_associations[0], 'association_no'].iloc[0]
                        one_detection['association_no'] = new_association_no
                    detection['association_no'] = new_association_no
                    
                    ## Add detections that are not already in association list
                    associations = associations.append( detection.loc[~detection.index.isin(overlap_detection_associations)] )
                    
                    ## Correct picked arrival time with new detections
                    associations.loc[associations.index.isin(overlap_detection_associations), 'time-corrected'] = one_detection['time-corrected']
                    
                    ## Correct count class time with new detections
                    associations.loc[associations.index.isin(overlap_detection_associations), 'count_class'] = one_detection['count_class']
                    
                    if determine_elapsed_time:
                        t2 = tm.time()
                        #print('time t2', t2-t1_)
                        #loc_dict = {'type': 't2', 'cost': t1-t1_}
                        #time_all = time_all.append( [loc_dict] )
                    
                    #print('associated_detections:', associated_detections.shape[0])
                    
                    ## Loop over associated detections
                    found_already_associated = False
                    grouped_associations = associated_detections.groupby(['satellite', 'station'])
                    for group, association in grouped_associations:
                    
                        if determine_elapsed_time:
                            t2_ = tm.time()
                    
                        #print('try to associate ', (satellite, station, arrival_class), ' and ', group)
                    
                        associated_satellite, associated_station = group
                    
                        ## If multiple associations are possible we select the best one
                        ## We select the detection with the maxixmum number of detection windows so far
                        if association.arrival_class.unique().size > 1:
                            ## proba, count per arrival class 
                            association['proba_max'] = association.groupby('arrival_class')['proba'].transform('max')
                            association = association.loc[association.count_class == association.count_class.max()]
                            association = association.loc[association.proba_max == association.proba_max.max()] 
                        
                        ## Get indexes for each detection
                        associated_arrival_class = association.arrival_class.iloc[0]
                        list_iassociation = association.index.tolist()
                
                        ## TODO: remove
                        associations_save = associations.copy()
                                    
                        ## If a possibly associated detection has already been associated
                        #if iassociation in associations:
                        association_no_found = associations.loc[associations.index.isin(list_iassociation), 'association_no']
                        #if iassociation in associations.index.tolist():
                        if association_no_found.size > 0:
                        
                            ## If current association is not already associated to possible associated detection
                            #association_no_found = associations[iassociation]
                            #association_no_found = associations.loc[associations.index==iassociation, 'association_no'].iloc[0]
                            association_no_found = association_no_found.iloc[0]
                            if not association_no == association_no_found:
                            
                                if determine_elapsed_time:
                                    t4_start = tm.time()
                            
                                #found_already_associated = True
                                #list_to_add = associations_reverse[association_no].copy()
                                #associations_reverse[association_no_found] += list_to_add
                                
                                list_to_add = associations.loc[associations.association_no == association_no].index.tolist()
                                associations.loc[associations.index.isin(list_to_add), 'association_no'] = association_no_found
                                
                                #test = associations.groupby(['association_no', 'event', 'satellite', 'station', 'arrival_class']).first().reset_index()
                                #test = test.groupby(['association_no', 'event', 'satellite', 'station'])['arrival_class'].count().reset_index()
                                #if test['arrival_class'].max() > 1:
                                #    bp()
                                
                                if determine_elapsed_time:
                                    t40 = tm.time()
                                    #print('time', t3-t2_)
                                    #loc_dict = {'type': 't40', 'cost': t40-t4_start}
                                    #time_all = time_all.append( [loc_dict] )
                                
                                ## Check if among all detections associated to the newly found association no, there are multiple arrivals per station
                                ## Quality check on other stations -> we do not want two phases from the same station/satellite
                                associations_check_right_no = associations.loc[(associations.association_no == association_no_found)]
                                associations_check_other_station = \
                                    associations_check_right_no.loc[ ~( (associations_check_right_no.satellite == satellite)
                                                                        & (associations_check_right_no.station == station) )]
                                
                                #associations_check_other_station_ = associations.loc[(associations.association_no == association_no_found) & ~(associations.satellite == satellite) & ~(associations.station == station)]
                                associations_check_nb = associations_check_other_station.groupby(['satellite', 'station', 'arrival_class']).first().reset_index()
                                associations_check_nb['count_arrival_class'] = associations_check_nb.groupby(['satellite', 'station'])['proba'].transform('count')
                                associations_check_nb = associations_check_nb.loc[associations_check_nb.count_arrival_class>1]
                                associations_check_nb = associations_check_nb.groupby(['satellite', 'station'])
                                
                                if determine_elapsed_time:
                                    t41 = tm.time()
                                    #print('time', t3-t2_)
                                    #loc_dict = {'type': 't41', 'cost': t41-t40}
                                    #time_all = time_all.append( [loc_dict] )
                                
                                for group_check, association_check_nb in associations_check_nb:
                                
                                    association_check_other_stations = \
                                        associations_check_other_station.loc[(associations_check_other_station.satellite == group_check[0])
                                                                            & (associations_check_other_station.station == group_check[1])]
                                    
                                    #association_check_other_stations = associations_check_other_station.loc[(associations_check_other_station.satellite == group_check[0]) & (associations_check_other_station.station == group_check[1])]
                                    #association_check_other_stations['proba_max'] = association_check_other_stations.groupby('arrival_class')['proba'].transform('max')
                                    #association_check_other_stations = association_check_other_stations.loc[association_check_other_stations.count_class == association_check_other_stations.count_class.min()]
                                    #association_check_other_stations = association_check_other_stations.loc[association_check_other_stations.proba_max == association_check_other_stations.proba_max.min()]
                                    
                                    #test = association_check_other_stations.groupby(['association_no', 'event', 'satellite', 'station', 'arrival_class']).first().reset_index()
                                    #test = test.groupby(['association_no', 'event', 'satellite', 'station'])['arrival_class'].count().reset_index()
                                    #print('Move other phase to another class', test['arrival_class'].max())
                                    
                                    #if association_check_other_stations.arrival_class.unique().size > 2:
                                    #    bp()
                                    
                                    ## proba, count per arrival class 
                                    association_check_other_stations['proba_max'] = association_check_other_stations.groupby('arrival_class')['proba'].transform('max')
                                    association_check_other_stations = \
                                        association_check_other_stations.loc[association_check_other_stations.count_class == association_check_other_stations.count_class.min()]
                                    association_check_other_stations = \
                                        association_check_other_stations.loc[association_check_other_stations.proba_max == association_check_other_stations.proba_max.min()]
                                    
                                    #print('Move another phase to another class')
                                    #station_test   = association_check_other_stations.station.iloc[0]
                                    #satellite_test = association_check_other_stations.satellite.iloc[0]
                                    #association_test = associations.loc[(associations.station == station_test) & (associations.satellite == satellite_test) & (associations.association_no==association_no)]
                                    
                                    association_no_moved = association_no
                                    #if association_test.size > 0:
                                    #    association_no_moved = associations.association_no.max()+1
                                    #    bp()
                                    associations.loc[associations.index.isin(association_check_other_stations.index), 'association_no'] = association_no_moved
                                
                                if determine_elapsed_time:
                                    t4 = tm.time()
                                    #print('time', t3-t2_)
                                    #loc_dict = {'type': 't4_start', 'cost': t4-t41}
                                    #time_all = time_all.append( [loc_dict] )
                                
                                """
                                test = associations.groupby(['association_no', 'event', 'satellite', 'station', 'arrival_class']).first().reset_index()
                                test = test.groupby(['association_no', 'event', 'satellite', 'station'])['arrival_class'].count().reset_index()
                                if test['arrival_class'].max() > 1:
                                    bp()
                                """
                                
                                ## Quality check on current station -> we do not want two phases from the same station/satellite
                                associations_check = \
                                    associations_check_right_no.loc[(associations_check_right_no.satellite == satellite)
                                                                    & (associations_check_right_no.station == station)]
                                #associations_check = associations.loc[(associations.association_no == association_no_found) & (associations.satellite == satellite) & (associations.station == station)]
                                if associations_check.arrival_class.unique().size > 1:
                                
                                    #if associations_check.arrival_class.unique().size > 2:
                                    #    bp()
                                    
                                    #test = associations_check.groupby(['association_no', 'event', 'satellite', 'station', 'arrival_class']).first().reset_index()
                                    #test = test.groupby(['association_no', 'event', 'satellite', 'station'])['arrival_class'].count().reset_index()
                                    #print('Move one of the phase to another class', test['arrival_class'].max())
                                
                                    ## proba, count per arrival class 
                                    associations_check['proba_max'] = associations_check.groupby('arrival_class')['proba'].transform('max')
                                    associations_check_min = associations_check.loc[associations_check.count_class == associations_check.count_class.min()]
                                    associations_check_min = associations_check_min.loc[associations_check_min.proba_max == associations_check_min.proba_max.min()]
                                    
                                    
                                    ## Move the "worse" arrival to another arrival class
                                    associations.loc[associations.index.isin(associations_check_min.index), 'association_no'] = association_no

                                    ## If the "worse" arrival is the current one, we have to flag for association_no update
                                    if one_detection.name in associations_check_min.index.tolist():
                                        association_no_found = association_no
                                
                                #test = associations.groupby(['association_no', 'event', 'satellite', 'station', 'arrival_class']).first().reset_index()
                                #test = test.groupby(['association_no', 'event', 'satellite', 'station'])['arrival_class'].count().reset_index()
                                #if test['arrival_class'].max() > 1:
                                #    #test.loc[test.arrival_class>1]
                                #    bp()
                                
                                ## Move all detections from current association to older one
                                #for current_assoc in list_to_add:
                                    #associations[current_assoc] = association_no_found
                                #    associations.loc[associations.index==current_assoc, 'association_no'] = association_no_found
                                    
                                ## Remove current association
                                #del associations_reverse[association_no]
                                association_no = association_no_found
                                
                                if determine_elapsed_time:
                                    t4_end = tm.time()
                                    
                                    #loc_dict = {'type': 't4', 'cost': t4_end-t4}
                                    #time_all = time_all.append( [loc_dict] )
                                    #print('time', t4-t3)
                                
                        
                        ## If a detection has not been associated and we are creating a new association
                        elif not one_detection.name in associations.index.tolist():
                            #associations[iassociation] = association_no
                            #associations_reverse[association_no].append( iassociation )
                            #associations.loc[associations.index==iassociation, 'association_no'] = association_no
                            
                            ## Check if station/satellite exists in current class
                            associations_test = associations.loc[(associations.satellite == associated_satellite)
                                                                & (associations.station == associated_station)
                                                                & (associations.association_no == association_no)]
                            
                            association['association_no'] = association_no
                            associations = associations.append( association )
                            
                            ## Move worst classes to other associations
                            if associations_test.size > 0:
                                associations_test = associations_test.append( association ) 
                                associations_test_max = associations_test.copy()
                                associations_test_max['proba_max'] = associations_test.groupby('arrival_class')['proba'].transform('max')
                                associations_test_max = associations_test_max.loc[associations_test_max.count_class == associations_test_max.count_class.max()]
                                associations_test_max = associations_test_max.loc[associations_test_max.proba_max == associations_test_max.proba_max.max()]
                                associations_test = associations_test.loc[~associations_test.index.isin(associations_test_max.index)]
                                associations_test = associations_test.groupby(['arrival_class'])
                                
                                max_no_association = associations.association_no.max()
                                for group_associations_test, association_test in associations_test:
                                    max_no_association += 1
                                    associations.loc[associations.index.isin(association_test.index), 'association_no'] = max_no_association
                            
                            #test = associations.groupby(['association_no', 'event', 'satellite', 'station', 'arrival_class']).first().reset_index()
                            #test = test.groupby(['association_no', 'event', 'satellite', 'station'])['arrival_class'].count().reset_index()
                            #if test['arrival_class'].max() > 1:
                            #    bp()
                        
                        if determine_elapsed_time:
                            t3 = tm.time()
                            #print('time t5', t3-t2)
                            #loc_dict = {'type': 't3', 'cost': t3-t2_}
                            #time_all = time_all.append( [loc_dict] )
                        
                    if determine_elapsed_time:
                        t5 = tm.time()
                        #print('time t5', t5-t2)
                        #loc_dict = {'type': 't5', 'cost': t5-t2}
                        #time_all = time_all.append( [loc_dict] )
                            
                    ## If we connected this detection to an already existing association map, we remove the newly created one
                    #if found_already_associated:
                    #    del associations_reverse[association_no]
                    
                    ## If this new association is saved we increment the association number for next possible association
                    #else:
                    #    association_no += 1
                    
                    #association_no = np.max([assoc for assoc in associations_reverse]) + 1
                    association_no = associations.association_no.max() + 1
                   
                """
                if associations.size > 0:
                    grouped_assoc = associations.groupby(['satellite', 'station', 'arrival_class'])
                    for group_assoc, assoc in grouped_assoc:
                        if assoc.association_no.unique().size > 1:
                            bp()
                """
                
                if determine_elapsed_time:
                    time_end = tm.time()
                    nb_detections = 0
                    if associations.size > 0:
                        new_detections_this_time = \
                            associations.groupby(['satellite', 'station', 'arrival_class']).first().reset_index()
                        nb_detections = new_detections_this_time.shape[0]
                    
                    nb_new_detections = 0
                    if detections_in_time_new.size > 0:
                        nb_new_detections = detections_in_time_new.groupby(['satellite', 'station', 'arrival_class']).first().shape[0]
                    
                    new_entry = {
                        'time': time, 
                        'cost': time_end - time_start, 
                        'nb_detections': nb_detections,
                        'detections_in_time_new': nb_new_detections,
                    }
                    time_elapsed = time_elapsed.append( [new_entry] )
                
                ## We append the new associations table if different
                if not associations.equals(associations_copy):
                    
                    ## Correct arrival times in associated arrivals
                    #utils_paper.correct_arrival_times(associations, offset, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
                    associations_copy = associations.copy()
                    associations_copy['time_association'] = time
                    associations_time_steps = associations_time_steps.append( associations_copy )
    
            ## Save event-specific association lists to global association list
            associations_all_events = associations_all_events.append( associations )
            associations_time_steps_all_events = associations_time_steps_all_events.append( associations_time_steps )
    
        if save_associations:
            associations_all_events.to_csv(association_name + '_associations.csv', header=True, index=False)
            associations_time_steps_all_events.to_csv(association_name + '_associations_time_steps.csv', header=True, index=False)   
    
    """  
    ## Plot computational cost
    time_elapsed.to_csv('time_elapsed.csv', header=True, index=False)
    selected_waveform = associations.loc[associations.index==619].iloc[0]
    event, satellite, station = selected_waveform.event, selected_waveform.satellite, selected_waveform.station
    try:
        station = "{:04d}".format(station)
    except:
        pass
    waveform = tec_data.loc[(tec_data['event'] == event) & (tec_data['satellite'] == satellite) & (tec_data['station'] == station), :]
    utils_paper.plot_time_cost_associations(time_elapsed, waveform, window, options, fontsize=15., time_max=15.)
    """
              
    return associations_all_events, associations_time_steps_all_events