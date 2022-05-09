#!/usr/bin/env python3

def get_ML_options():
    
    options = {}
    options['save_features'] = True
    options['check_new_features'] = False
    options['save_features_picker'] = True
    options['load_tec_data'] = True
    options['save_tec_data'] = True
    options['check_new_tec_data'] = False
    options['use_picker'] = True
    options['plot_correlations'] = False
    
    return options

def get_training_parameters():

    options = {}
    options['add_gaussian_noise'] = True
    options['augment_noise_snr_min'] = 1.
    options['augment_noise_snr_max'] = 5.
    options['nb_windows_artificial_noise'] = 0
    options['nb_noise_windows_noarrival'] = 5
    options['nb_noise_windows']   = 4
    options['nb_arrival_windows'] = 4
    options['shift_detection_after_max'] = 500. # factor (x diff-t) 
    options['min_overlap_label'] = 0.7 # minimum overlap between perturbed waveforms extracted from arrival
    options['noise_pick_shift']  = 1000. # Delay from arrival time to pick noise
    options['default_duration']  = 150. # (s) default duration of an arrival if none is provided
    options['min_deviation']     = 90. # (s) minimum time between random picks of noise and arrival windows
    options['class_weight'] = 'balanced'
    options['type_input'] = 'features'
    options['type_ML']    = 'forest'

    return options

def get_detector_parameters():

    options = {}
    options['nb_for_class']     = 3
    options['nb_for_end_class'] = 4
    
    return options

def get_preprocessing_parameters():

    options = {}
    options['list_freq_bans'] = [ [0.001, 0.005], [0.005, 0.015] ]
    options['freq_min'] = 1e-4
    options['freq_max'] = 1./30.
    options['window'] = { # in s
            40.: 720.,
            -1.: 720.,
        }
    options['factor_overlap'] = options['window'][40.]/30.
    
    return options

def get_signal_duration():

    options = {}
    options['signal_duration'] = {
        'Tohoku_1s': 700.,
        'Tohoku_30s': 700.,
        'Sanriku_1s': 140.,
        'Sanriku_30s': 140.,
        'Kii_30s': 425.,
        'Kaikoura_1s': 350.,
        'Sumatra_1_15s': 300.,
        'Sumatra_2_15s': 300.,
        'Fiordland_30s': 300.,
        'Tokachi_30s': 310.,
        'Chuetsu_30s': 200.,
        'Macquarie_30s': 400.,
        'Illapel_30s': 600.,
        'Iquique_30s': 700.,
    }
    
    return options