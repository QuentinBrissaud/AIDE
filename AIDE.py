#!/usr/bin/env python3
from pdb import set_trace as bp
import pandas as pd

import detector, read_data, train_est, train_wave_picker, utils_paper, associator, constants

class AIDE():
    
    def __init__(self, dir_dict, training_parameters={}, detector_parameters={}, preprocessing_parameters={}, signal_duration={}, ML_options={}):
        self.options = {}
        
        self.options.update( dir_dict )
        
        ## Load default options or user defined ones
        if not training_parameters:
            training_parameters = constants.get_training_parameters()
        self.options.update( training_parameters )
        
        if not detector_parameters:
            detector_parameters = constants.get_detector_parameters()
        self.options.update( detector_parameters )
        
        if not preprocessing_parameters:
            preprocessing_parameters = constants.get_preprocessing_parameters()
        self.options.update( preprocessing_parameters )
        
        if not signal_duration:
            signal_duration = constants.get_signal_duration()
        self.options.update( signal_duration )
        
        if not ML_options:
            ML_options = constants.get_ML_options()
        self.options.update( ML_options )
        
        self.optimization = {}
        
        ## Initialize variables
        self.tec_data = pd.DataFrame()
        self.tec_data_param = pd.DataFrame()
        self.est = None
        self.data = pd.DataFrame()
        self.est_picker = None
        self.options['load'] = {}
        self.detections = pd.DataFrame()
        self.associations = pd.DataFrame()
        self.associations_time_steps = pd.DataFrame()
        
    def load_data(self, load_dict):
    
        self.options['load'].update( load_dict )
        self.tec_data, self.tec_data_param = read_data.read_data_folders(self.options)
        
    def load_RF_detector(self, load_dict):
    
        self.options['load'].update( load_dict )
        self.est = train_est.load_est_forest(self.options)
        self.data = train_est.load_data_forest(self.options)
        
    def load_RF_picker(self, load_dict):
    
        self.options['load'].update( load_dict )
        self.est_picker = train_wave_picker.load_picker(self.options)
        
    def run_detections(self, load_dict, detection_parameters, add_inset=False, adaptative_sampling=False, standard_sampling=30., standard_sampling_for_picker=30., nb_picks=5, plot_probas=True, focus_on_arrival=False, focus_arrival_shift=1000., nb_CPU=20):
    
        if self.tec_data.shape[0] == 0:
            print('Load vTEC data before performing detection')
            return
            
        if self.est == None:
            print('Load RF detector before performing detection')
            return
    
        if self.est_picker == None:
            print('Load RF picker before performing detection')
            return
    
        self.detection_parameters = detection_parameters
        self.options['load'].update( load_dict )
        self.detections, self.probas, _ = \
            detector.compute_arrival_times_RF(self.est, self.data, self.tec_data, self.tec_data_param, detection_parameters, 
                                                      self.options, focus_on_arrival=focus_on_arrival, focus_arrival_shift=focus_arrival_shift,
                                                      est_picker=self.est_picker, return_all_waveforms_used=False,
                                                      plot_probas=plot_probas, stop_at_each_iter=False, nb_picks=nb_picks,
                                                      add_label='', adaptative_sampling=adaptative_sampling, standard_sampling=standard_sampling,
                                                      standard_sampling_for_picker=standard_sampling_for_picker, nb_CPU=nb_CPU, add_inset=add_inset)
                                                      
    def run_association(self, load_dict, nb_pts_picker=5, quantile_threshold=0.8, offset=500., window=720., sampling=30., max_radius_search=500., velocity_search_max=50., velocity_search_min=0.65, save_associations=True, association_name='Tohoku_250km_afterreview_detrend'):
    
        if self.detections.shape[0] == 0:
            print('Run detector before performing association')
            return
    
        self.options['load'].update( load_dict )
        tec_data_hion = pd.DataFrame()
        dummy_value = 500.
        utils_paper.correct_arrival_times(self.detections, dummy_value, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
        first_detections = utils_paper.create_arrival_time_plot(self.detections, self.options, offset=dummy_value, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
        
        self.associations, self.associations_time_steps = \
            associator.associator(self.tec_data, self.detections, self.probas, self.options, offset=offset, quantile_threshold=quantile_threshold, 
                       window=window, sampling=sampling, max_radius_search=max_radius_search, 
                       velocity_search_max=velocity_search_max, velocity_search_min=velocity_search_min, tec_data_hion=tec_data_hion, hion_dict={}, save_associations=save_associations, association_name=association_name)
        
        associator.add_lat_lon_to_detections(self.tec_data, self.associations, window, tec_data_hion=tec_data_hion, hion_dict={})
        
##########################
if __name__ == '__main__':
    
    ## Directories
    main_dir = '/staff/quentin/Documents/Projects/ML_TEC/'
    dir_dict = {
        'DIR_FIGURES': main_dir + 'figures/',
        'DIR_DATA': main_dir + 'data/'
    }
    
    ## TODO: add data loader and test new detections and new associations + create list module requirements
    
    ## Loading options
    load_dict = {
        'est': main_dir + 'figures/model_arr4_noise4_snr1_5.0_pFalse/forest_est_s500.0_m0.7_n1000.0_w720.0.pkl',
        'data': main_dir + 'figures/model_arr4_noise4_snr1_5.0_pFalse/data_s500.0_m0.7_n1000.0_w720.0.pkl',
        'features': main_dir + 'figures/model_arr4_noise4_snr1_5.0_pFalse/features_features_m0.7_w720.0.csv',
        'features-picker': main_dir + 'data/features_picker_w720.0_d0.7.csv',
        'est-picker': main_dir + 'figures/random_est_picker_w720.0_d0.7_bTrue.joblib',
        #'detections': main_dir + 'figures/detected_arrivals_Iquique_afterreview_detrend.csv',
        #'probas': main_dir + 'figures/probas_all_waveforms_Iquique_afterreview_detrend.csv',
        #'associations': main_dir + 'Iquique_afterreview_associations.csv',
        #'associations_time_steps': main_dir + 'Iquique_afterreview_associations_time_steps.csv',
        }
    
    ## Create detection model
    one_model = AIDE(dir_dict)
    one_model.load_data(load_dict)
    one_model.load_RF_detector(load_dict)
    one_model.load_RF_picker(load_dict)
    
    ## Create detection list to process
    detection_network = {
        'name': 'test',
        'events': ['Tohoku_1s'],
        'satellites': ['G26'],
        'stations': one_model.tec_data.loc[one_model.tec_data['event'].isin( ['Tohoku_1s'] ) & one_model.tec_data['satellite'].isin( ['G26'] ), 'station'].unique().tolist()[:5],
        'time_end': one_model.tec_data.loc[one_model.tec_data['event'].isin( ['Tohoku_1s'] ), 'time_s'].max()
    }
    
    detection_options = { 
        'nb_picks': 5, 
        'plot_probas': True, 
        'focus_on_arrival': False, 
        'focus_arrival_shift': 1000., 
        'nb_CPU': 20
    }
    one_model.run_detections(load_dict, detection_network, **detection_options)
    
    ## Create association list
    association_options = {
        'max_radius_search': 500., 
        'velocity_search_max': 50., 
        'velocity_search_min': 0.65, 
        'save_associations': True, 
        'association_name': 'test'
    }
    one_model.run_association(load_dict, **association_options)
    
    first_detections = utils_paper.create_arrival_time_plot(one_model.detections, one_model.options, offset=500., nb_pts_picker=5, quantile_threshold=0.8)
    utils_paper.plot_image_iono(one_model.tec_data, one_model.tec_data_param, first_detections, one_model.options, associations=one_model.associations, add_fault=False, add_inset_fault=False, unknown='slip', rotation=25., vmin=6., vmax=11., offset_source_lat=8., offset_source_lon=8., first_label='d', hion_dict={}, add_new_waveform={}, add_new_waveform_class={}, ext_name='_test')
    bp()