#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
import string
import pandas as pd
from pdb import set_trace as bp
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from obspy.core.utcdatetime import UTCDateTime
from matplotlib import animation
from functools import partial
from sklearn.datasets import make_blobs
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import os
import matplotlib.ticker as ticker

from sklearn import preprocessing
from obspy.geodetics.base import gps2dist_azimuth

from multiprocessing import get_context
from functools import partial

from adjustText import adjust_text
import obspy

import detector, train_est, compute_params_waveform, read_data, train_wave_picker

def plot_list_detetions(folder, associations, probas, tec_data, ev_coord, ev_time, 
                        figsize=(4,20), max_waveform=10, vmin=0.6, vmax=1., list_stat_sat=[], 
                        label_fig='', fontsize_label_fig=14):
    # utils_paper.plot_section(associations, tec_data, ev_coord)
    
    """
    Plot a stack of vTEC records from the same event along with a map
    """
    
    #first_detections = associations.groupby(['satellite','station','arrival_class']).first().reset_index()
    
    first_detections = associations.loc[associations.groupby(['satellite', 'station'])['proba'].idxmax()]
    first_detections['combi'] = first_detections['satellite']+'-'+first_detections['station']
    first_detections = first_detections.loc[first_detections.combi.isin(list_stat_sat)]
    
    nb_entries = first_detections.groupby(['satellite', 'station']).first().reset_index().shape[0]
    first_detections['dist'] = first_detections.apply(lambda x: gps2dist_azimuth(ev_coord[0], ev_coord[1], x.lat, x.lon)[0], axis=1)
    first_detections = first_detections.sort_values(by='dist')
    first_detections['order'] = -1
    #for igroup, (group, one_detect) in enumerate(first_detections.groupby(['satellite', 'station'])):
    #    first_detections.loc[first_detections.index.isin(one_detect.index), 'order'] = igroup
    first_detections['order'] = range(nb_entries)
    first_detections['order'] = first_detections.groupby(['satellite','station'])['order'].transform('min')
    UT_event = (ev_time - UTCDateTime(ev_time.year, ev_time.month, ev_time.day))/60.
    first_detections['time_UT'] = first_detections['time-corrected']/60. - UT_event
    first_detections = first_detections.loc[first_detections['order'] < max_waveform]
    
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    #cmap.set_alpha(0.4)
    cmap.set_under(alpha=0.)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(min(max_waveform, first_detections.combi.unique().size), 1)
    axs = []
    order_group = []
    ll_test = []
    #fig, axs = plt.subplots(nrows=nb_entries, ncols=2, sharex=True)
    for i, (group, data) in enumerate(first_detections.groupby(['satellite','station'])):
        satellite, station = group
        
        ## If only station/sat subset needed we skip all other entries
        #if list_stat_sat:
        #    if not (satellite, station) in list_stat_sat:
        #        continue
        
        print(satellite, station)
        
        igroup = data.order.iloc[0]
        order_group.append( igroup )
        dist = data.dist.iloc[0]/1e3
        loc_data = tec_data.loc[(tec_data.satellite == satellite) & (tec_data.station == station)]
        if i == 0:
            ax = fig.add_subplot(gs[igroup])
        else:
            ax = fig.add_subplot(gs[igroup], sharex=axs[0])
        axs.append(ax)
        
        """
        tr.data = loc_data.vTEC.values
        dt = (loc_data.UT.iloc[1]-loc_data.UT.iloc[0])*3600.
        tr.stats.delta = dt
        tr.stats.starttime = UTCDateTime(ev_time.year, ev_time.month, ev_time.day) + int(loc_data.UT.min()*3600.)
        tr.filter('highpass', freq=1/7200., zerophase=True)
        tr.detrend('linear')
        
        """
        
        one_association = associations.loc[(associations.satellite == satellite) & (associations.station == station)]
        one_proba = probas.loc[(probas.satellite == satellite) & (probas.station == station)].drop_duplicates()
        
        from scipy import interpolate
        tr = obspy.Trace()
        dt = loc_data.iloc[0].sampling
        f = interpolate.interp1d(loc_data.UT.values*3600., loc_data.vTEC.values, fill_value='extrapolate')
        times = np.arange(loc_data.UT.min()*3600., loc_data.UT.max()*3600.+dt, dt)
        vTEC = f(times)
        tr.data = vTEC
        tr.stats.delta = dt
        tr.stats.starttime = UTCDateTime(ev_time.year, ev_time.month, ev_time.day) + int(loc_data.UT.min()*3600.)
        tr.filter('highpass', freq=1/7200., zerophase=True)
        tr.detrend('linear')
        f = interpolate.interp1d(tr.times()+loc_data.UT.min()*3600., tr.data, fill_value='extrapolate')
        #vTEC = f(one_proba.time+720.)
        
        #print(tr.times()[-1]/3600.+loc_data.UT.min(), one_proba.iloc[-1].time/3600.)
        #if abs(tr.times()[-1]+loc_data.UT.min()*3600. - one_proba.iloc[-1].time) > 5:
        #    continue
        
        print('-->', igroup, satellite, station)
        ll_test.append((satellite, station))
        
        #from scipy import interpolate
        #f = interpolate.interp1d(tr.times()+loc_data.UT.min()*3600., tr.data)
        #print('xxxx')
        #print(tr.times()+loc_data.UT.min()*3600.)
        #print(one_proba.time.values)
        
        #print(one_proba.time.values, tr.times()+loc_data.UT.min()*3600.)
        #sc = ax.scatter(one_proba.time/3600., f(one_proba.time.values), c=one_proba.proba, vmin=0.5, vmax=0.9, cmap=cmap)
        #ydata = loc_data.loc[loc_data.UT.round(5).isin((one_proba.time/3600.).round(5))]
        #fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        sc = ax.scatter((one_proba.time+720.)/3600., f(one_proba.time+720.), c=one_proba.proba, vmin=vmin, vmax=vmax, cmap=cmap)
        #ax.plot(one_proba.time/3600., f(one_proba.time), linewidth=2.)
        ax.plot(loc_data.UT.values.min()+tr.times()/3600., tr.data, linewidth=2.)
        
        # TODO: remove
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        sc = ax.scatter((one_proba.time+720.), f(one_proba.time+720.), c=one_proba.proba, vmin=0.5, vmax=0.9, cmap=cmap)
        ax.plot(one_proba.time, f(one_proba.time), linewidth=2.)
        ax.plot(loc_data.UT.values.min()*3600.+tr.times(), tr.data, color='red')
        plt.show()
        """
        
        ax.text(-0.02, 0.5, '{satellite}-{station}'.format(satellite=satellite, station=station), ha='right', va='center', transform=ax.transAxes)
        ax.text(1.02, 0.5, '{dist:.1f}'.format(dist=dist), ha='left', va='center', transform=ax.transAxes)
        ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        #ax.axvline(data.iloc[0]['time-corrected']/3600., linestyle=':')
        
        #for irow, row in data.iterrows():
        #    ax.axvline(row['time-corrected']/3600.)
        
        #ax.axvline(UT_event/60., color='tab:red')

        #if station == 'lesv' and satellite == 'R12':
        #    bp()

    for iax in range(1,len(order_group)):
        ax = axs[order_group.index(iax)]
        frame = ax.spines["top"]
        frame.set_visible(False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=False)
        
    for iax in range(1,len(order_group)-1):
        ax = axs[order_group.index(iax)]
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
    for iax in range(len(order_group)-1):
        ax = axs[order_group.index(iax)]
        frame = ax.spines["bottom"]
        frame.set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax = axs[order_group.index(0)]
    axins = inset_axes(ax, width="50%", height="5%", loc='lower left', 
                        bbox_to_anchor=(0.25, 1.1, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False, labelrotation=90.)
    cbar = plt.colorbar(sc, cax=axins, extend='both', orientation="horizontal")  
    cbar.ax.xaxis.set_ticks_position('top') 
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.tick_top()
    cbar.ax.set_xlabel('Detection probability', labelpad=1) 

    axs[order_group.index(0)].text(-0.02, 1.05, label_fig, ha='right', va='bottom', transform=axs[order_group.index(0)].transAxes, fontsize=fontsize_label_fig, fontweight='bold')
    axs[order_group.index(len(order_group)-1)].set_xlabel('Time (UT)')
    fig.subplots_adjust(hspace=0.01, left=0.2, top=0.85, right=0.87)

    plt.savefig(folder+'waveforms_probas_{folder}.pdf'.format(folder=folder.split('/')[-2]))
    
    #print(ll_test)

def plot_section(folder, associations, tec_data, ev_coord, ev_time, vmin=5., vmax=10.):
    # utils_paper.plot_section(associations, tec_data, ev_coord)
    
    """
    Plot a stack of vTEC records from the same event along with a map
    """
    
    #first_detections = associations.groupby(['satellite','station','arrival_class']).first().reset_index()
    
    first_detections = associations.loc[associations.groupby(['satellite', 'station'])['proba'].idxmax()]
    
    nb_entries = first_detections.groupby(['satellite', 'station']).first().reset_index().shape[0]
    first_detections['dist'] = first_detections.apply(lambda x: gps2dist_azimuth(ev_coord[0], ev_coord[1], x.lat, x.lon)[0], axis=1)
    first_detections = first_detections.sort_values(by='dist')
    first_detections['order'] = -1
    #for igroup, (group, one_detect) in enumerate(first_detections.groupby(['satellite', 'station'])):
    #    first_detections.loc[first_detections.index.isin(one_detect.index), 'order'] = igroup
    first_detections['order'] = range(nb_entries)
    first_detections['order'] = first_detections.groupby(['satellite','station'])['order'].transform('min')
    UT_event = (ev_time - UTCDateTime(ev_time.year, ev_time.month, ev_time.day))/60.
    first_detections['time_UT'] = first_detections['time-corrected']/60. - UT_event
    print(first_detections)
    
    fig = plt.figure()
    offset=2
    gs = fig.add_gridspec(nb_entries*2+offset, 1)
    axs = []
    order_group = []
    #fig, axs = plt.subplots(nrows=nb_entries, ncols=2, sharex=True)
    for i, (group, data) in enumerate(first_detections.groupby(['satellite','station'])):
        satellite, station = group
        igroup = data.order.iloc[0]
        order_group.append( igroup )
        dist = data.dist.iloc[0]/1e3
        loc_data = tec_data.loc[(tec_data.satellite == satellite) & (tec_data.station == station)]
        if i == 0:
            ax = fig.add_subplot(gs[igroup])
        else:
            ax = fig.add_subplot(gs[igroup], sharex=axs[0])
        axs.append(ax)
        
        tr = obspy.Trace()
        tr.data = loc_data.vTEC.values
        tr.stats.delta = 30.
        tr.stats.starttime = UTCDateTime(ev_time.year, ev_time.month, ev_time.day) + int(loc_data.UT.min()*3600.)
        tr.filter('highpass', freq=1/7200., zerophase=True)
        tr.detrend('linear')
        
        ax.plot(tr.times()/3600.+loc_data.UT.min(), tr.data)
        ax.text(-0.02, 0.5, '{satellite}-{station}'.format(satellite=satellite, station=station), ha='right', va='center', transform=ax.transAxes)
        ax.text(1.02, 0.5, '{dist:.1f}'.format(dist=dist), ha='left', va='center', transform=ax.transAxes)
        ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        for irow, row in data.iterrows():
            ax.axvline(row['time-corrected']/3600.)
        
        ax.axvline(UT_event/60., color='tab:red')
        
    print(order_group, len(axs))
        
    for iax in range(1,len(order_group)):
        ax = axs[order_group.index(iax)]
        frame = ax.spines["top"]
        frame.set_visible(False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=False)
        
    for iax in range(1,len(order_group)-1):
        ax = axs[order_group.index(iax)]
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
    for iax in range(len(order_group)-1):
        ax = axs[order_group.index(iax)]
        frame = ax.spines["bottom"]
        frame.set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    colors = ["black" for i in range(tec_data.satellite.unique().size)]
    only_one = sns.color_palette(colors)    
    
    cmap = sns.color_palette("rocket", as_cmap=True)
    axs[order_group.index(len(order_group)-1)].set_xlabel('Time (UT)')
    
    ## Map
    tec_data['combi'] = tec_data['satellite'] + '-' + tec_data['station']
    ax = fig.add_subplot(gs[nb_entries+offset:])
    sns.scatterplot(data=tec_data, x='lon', y='lat', hue='combi', ax=ax, legend=False, s=1, linewidth=0)
    sc = ax.scatter(first_detections['lon'], first_detections['lat'], marker='o', c=first_detections['time_UT'], s=100, cmap=cmap, vmin=vmin, vmax=vmax)
    #sns.scatterplot(data=first_detections, x='lon', y='lat', hue='time_UT', palette='rocket', style='satellite', s=150, ax=ax, legend=False, vmin=vmin, vmax=vmax)
    ax.scatter(ev_coord[1], ev_coord[0], marker='x', color='red', s=150)

    axins = inset_axes(ax, width="4%", height="100%", loc='lower left', 
                        bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = plt.colorbar(sc, cax=axins, extend='both')
    cbar.ax.set_ylabel('Arrival time (min)', rotation=270, labelpad=12) 
    

    fig.subplots_adjust(hspace=0.01, right=0.87)
    plt.savefig(folder + 'arrivals_section.pdf')
        
def plot_map(ax, list_events):

    m = Basemap(projection='mill', lon_0=0, llcrnrlat=-75,urcrnrlat=75,\
                llcrnrlon=-180,urcrnrlon=180, ax=ax)
    m.drawmapboundary(fill_color='lightcyan')
    m.fillcontinents(color='blanchedalmond', lake_color='lightcyan')
    m.drawcoastlines()
    m.drawcountries()
    
    lat_ticks = np.linspace(-60., 60., 5)
    lon_ticks = np.linspace(-140., 140., 5)
    #lats = m.drawparallels(lat_ticks, labels=[True,False,False,False], zorder=5)
    #lons = m.drawmeridians(lon_ticks, labels=[False,False,True,False], zorder=5)
    
    texts, scatters = [], []
    for event in list_events:
        lat, lon = list_events[event]
        scatters.append( m.scatter(lon, lat, marker='*', s=200, latlon=True, edgecolors='black', clip_on=False, color='yellow', zorder=10) )
        x, y = m(lon, lat + 2.)
        texts.append( ax.text(x, y, event, horizontalalignment='center', verticalalignment='center', 
                             bbox=dict(facecolor='w', edgecolor='w', pad=1., alpha=0.75), color='black', zorder=15) )

    adjust_text(texts, add_objects=scatters, only_move={'points':'xy', 'texts':'xy'}, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

def fixed_aspect_ratio(ax, ratio):
    '''
    Set a fixed aspect ratio on matplotlib plots 
    regardless of axis units
    '''
    xvals,yvals = ax.get_xlim(),ax.get_ylim()

    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    ax.set_aspect(ratio*(xrange/yrange), adjustable='box')

def plot_map_events_and_waveforms(tec_data, tec_data_param, options, 
                                  list_waveforms=[], offset_from_arrival=1000., fontsize=15.):

    list_events = {
        'Tohoku': (38.3, 142.37),
        'Sanriku': (38.44, 142.84),
        'Kaikoura': (-42.757, 173.077),
        'Kii': (33.1, 136.6),
        'Sumatra1': (2.35, 92.86),
        'Sumatra2': (0.90, 92.31),
        'Macquarie': (-49.91, 161.25),
        'Fiordland': (-45.75, 166.58),
        'Tokachi': (41.78, 143.90),
        'Chuetsu': (37.54, 138.45),
        'Illapel': (-31.57, -71.61),
        'Iquique': (-19.61, -70.77),
    }
    
    fig = plt.figure(figsize=(6,4))
    #height = 3
    height = 10
    #width  = 6
    width  = 8
    size_waveforms = 2
    w_waveforms = 2
    h_waveforms = 2
    h_map = 6
    offset_waveform = 1
    offset_h_waveform = 0
    plot_per_col = 3
    gs = fig.add_gridspec(height, width)

    #ax = fig.add_subplot(gs[:, :width-size_waveforms])
    ax = fig.add_subplot(gs[:h_map, :])
    plot_map(ax, list_events)
    ax.text(-0.02, 1.05, 'a)', ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize, fontweight='bold')
    
    alphabet = string.ascii_lowercase
    axs_waveform = []
    nb_rows = 0
    for iax, data in enumerate(list_waveforms[:]):
    
        arrival_time = -1
        if len(data) == 3:
            event, satellite, station = data
        else:
            event, satellite, station, arrival_time = data
            
        iax_plot = iax - nb_rows*plot_per_col
        if iax_plot == plot_per_col:
            offset_h_waveform += h_waveforms
            nb_rows += 1
            iax_plot = iax - nb_rows*plot_per_col
    
        
        #axs_waveform.append( fig.add_subplot(gs[iax+offset_waveform, width-size_waveforms:]) )
        axs_waveform.append( fig.add_subplot(gs[h_map+offset_h_waveform:h_map+h_waveforms+offset_h_waveform, 
                                               iax_plot*w_waveforms+offset_waveform:iax_plot*w_waveforms+offset_waveform+w_waveforms]) )
        
        waveform = read_data.get_one_entry(tec_data, station, satellite, event)
        if arrival_time == -1:
            color = 'tab:green'
            params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                            & (tec_data_param['satellite'] == satellite) 
                            & (tec_data_param['event'] == event), : ].iloc[0]
            arrival_time = params['arrival-time']/3600.
            axs_waveform[-1].axvline(arrival_time, alpha=0.4, color='black')
        else:
            color = 'tab:red'
                         
        waveform = waveform.loc[(waveform.time_s >= arrival_time*3600. - offset_from_arrival) 
                                & (waveform.time_s <= arrival_time*3600. + offset_from_arrival), :]
        times = waveform['UT'].values
        vTEC  = waveform['vTEC'].values
        
        axs_waveform[-1].plot(times, vTEC, color=color, linewidth=1.25)
        axs_waveform[-1].set_xlim([times.min(), times.max()])
        #str_event = event.split('_')[0] + ' - ' + satellite + ' - ' + station
        #axs_waveform[-1].set_title(str_event, fontsize=11., pad=0.)
        
        axs_waveform[-1].text(-0.0, 1.015, alphabet[iax+1]+')', ha='right', va='bottom', transform=axs_waveform[-1].transAxes, fontsize=fontsize, fontweight='bold')
        
        #fixed_aspect_ratio(axs_waveform[-1], 0.5)
    
    ## Add time scale bar
    scalebar = AnchoredSizeBar(axs_waveform[nb_rows*plot_per_col].transData,
               5./60., '5 mn', 'lower left', 
               pad=0.1,
               color='black',
               frameon=False,
               size_vertical=0.05)
    axs_waveform[nb_rows*plot_per_col].add_artist(scalebar)

    ## Add arrival information
    axs_waveform[2].text(0.05, 0.95, 'arrival', ha='left', va='top', 
                         transform=axs_waveform[2].transAxes, fontsize=10.,
                         color='black', alpha=0.6)
    axs_waveform[-1].text(0.05, 0.05, 'noise', ha='left', va='bottom', 
                         transform=axs_waveform[-1].transAxes, fontsize=10.,
                         color='tab:red', alpha=0.6)
        
    plt.setp(axs_waveform, xticks=[], yticks=[])
    
    #fig.subplots_adjust(wspace=1., hspace=0.2)
    fig.subplots_adjust(wspace=0.15, hspace=1.1)
    
    plt.savefig(options['DIR_FIGURES'] + 'map_events.pdf')

def plot_blobs_association(ax):

    """
    Create fake 2d blobs for association step in ML scheme
    """

    # Generate sample data
    n_samples = 4000
    n_components = 3
    X, y_true = make_blobs(n_samples=n_samples,
                           centers=n_components,
                           cluster_std=0.60,
                           random_state=0)

    colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']
    for k, col in enumerate(colors):
        cluster_data = y_true == k
        ax.scatter(X[cluster_data, 0], X[cluster_data, 1],
                    c=col, marker='.', s=10)

def get_FTP_FTN(data, est):

    """
    Get a list of False/True positives and negatives over the testing dataset
    """

    ## Find inputs/outputs
    input_columns  = [key for key in train_est.data_without_info_columns(data)]
    output_columns = ['type-data']
    
    ## Find initial training and testing data
    data_train = data.loc[data['type'] == 'train', input_columns]
    out_train  = data.loc[data['type'] == 'train', output_columns]
    data_test = data.loc[data['type'] == 'test', input_columns]
    out_test  = data.loc[data['type'] == 'test', output_columns]
    
    ## Train a first estimator to find false positives
    out_pred  = est.predict(data_test)
    out_pred_train = est.predict(data_train)
    
    ## Encode outputs
    le = preprocessing.LabelEncoder()
    le.fit(out_test.values)
    out_test_encoded = le.transform(out_test.values)
    out_train_encoded = le.transform(out_train.values)
    out_pred_encoded = le.transform(out_pred)
    out_pred_train_encoded = le.transform(out_pred_train)

    ## Find false positives and false negatives
    ## Arrival = 0; Noise = 1
    idx_FP = np.where(out_test_encoded-out_pred_encoded > 0)[0]
    if idx_FP.size > 0:
        idx_FP = data_test.iloc[idx_FP].index.tolist()
    idx_FP_train = np.where(out_train_encoded-out_pred_train_encoded > 0)[0]
    if idx_FP_train.size > 0:
        idx_FP += data_train.iloc[idx_FP_train].index.tolist()
    
    idx_TP = np.where((out_test_encoded==0) & (out_pred_encoded == 0))[0]
    
    if idx_TP.size > 0:
        idx_TP = data_test.iloc[idx_TP].index.tolist()
    idx_TP_train = np.where((out_train_encoded == 0) & (out_pred_train_encoded == 0))[0]
    print(idx_TP_train)
    if idx_TP_train.size > 0:
        idx_TP += data_train.iloc[idx_TP_train].index.tolist()
    
    idx_FN = np.where(out_test_encoded-out_pred_encoded < 0)[0]
    if idx_FN.size > 0:
        idx_FN = data_test.iloc[idx_FN].index.tolist()
    idx_FN_train = np.where(out_train_encoded-out_pred_train_encoded < 0)[0]
    if idx_FN_train.size > 0:
        idx_FN += data_train.iloc[idx_FN_train].index.tolist()
    
    idx_TN = np.where((out_test_encoded==1) & (out_pred_encoded == 1))[0]
    if idx_TN.size > 0:
        idx_TN = data_test.iloc[idx_TN].index.tolist()
    idx_TN_train = np.where((out_train_encoded==1) &(out_pred_train_encoded == 1))[0]
    if idx_TN_train.size > 0:
        idx_TN += data_train.iloc[idx_TN_train].index.tolist()
    
    return data.loc[data.index.isin(idx_FN)], data.loc[data.index.isin(idx_FP)], data.loc[data.index.isin(idx_TN)], data.loc[data.index.isin(idx_TP)]

def show_FP_FN(tec_data, FP_in, FN_in, window, options, type_input='F', nb_waveforms=6, seed=0, plot_per_col=6, axs=[], all_test_dataset=False, exclude=[]):

    """
    Show a grid of waveforms corresponding to FP or FN
    """

    FP = FP_in.loc[~FP_in.event.isin(exclude)]
    FN = FN_in.loc[~FN_in.event.isin(exclude)]
    
    if not all_test_dataset:
        FP = FP.loc[~FP.extra_test]
        FN = FN.loc[~FN.extra_test]

    ## Add waveforms
    alphabet = string.ascii_lowercase
    axs_waveform = []
    np.random.seed(seed)
    
    ## Collect metadata for each FN/FP
    l_FP = np.arange(FP.shape[0])
    np.random.shuffle(l_FP)
    l_FN = np.arange(FN.shape[0])
    np.random.shuffle(l_FN)
    if type_input == 'T':
        l_FN = l_FN[:len(l_FP)//10]
        
    if l_FP.size > 0:
        FP = FP.iloc[l_FP]
    if l_FN.size > 0:
        FN = FN.iloc[l_FN]
    
    FN_new = FN.append( FP )
    if FN_new.shape[0] > 0:
        l_FN_new = np.arange(FN_new.shape[0])
        np.random.shuffle(l_FN_new)
        FN_new = FN_new.iloc[l_FN_new]
    
    list_waveforms = []
    for j in range(nb_waveforms):
        i = j#np.random.randint(0,FN.shape[0])
        
        if i >= FN_new.shape[0]:
            continue
        
        entry = FN_new.iloc[i].event, FN_new.iloc[i].satellite, FN_new.iloc[i].station, FN_new.iloc[i]['arrival-time']/3600., FN_new.iloc[i]['type-data']
        list_waveforms.append( entry )
            
        #i = np.random.randint(0,FP.shape[0])
        #entry = FN.iloc[i].event, FN.iloc[i].satellite, FN.iloc[i].station, FN.iloc[i]['arrival-time']/3600., 'arrival'
        #entry = FP.iloc[i].event, FP.iloc[i].satellite, FP.iloc[i].station, FP.iloc[i]['arrival-time']/3600., 'noise'
        #list_waveforms.append( entry )
    
    ## Setup figure
    nrows_total = int(np.ceil(len(list_waveforms)/plot_per_col))
    fig, axs = plt.subplots(nrows=nrows_total, ncols=plot_per_col, sharex=True, sharey=True)
    
    ## Plot FP/FN
    iax = 0
    max_amp = 0.
    nb_rows = 0
    for (event, satellite, station, t0, type) in list_waveforms[:]:
        
        waveform_orig = read_data.get_one_entry(tec_data, station, satellite, event)
        
        iax_plot = iax - nb_rows*plot_per_col
        if iax_plot == plot_per_col:
            nb_rows += 1
            iax_plot = iax - nb_rows*plot_per_col
    
        waveform = waveform_orig.loc[(waveform_orig.UT >= t0) & (waveform_orig.UT <= t0 + window/3600.), :]
        times = waveform['UT'].values
        vTEC  = waveform['vTEC'].values
        size_subset = np.argmin( abs( (times-times[0]) - window ) )
        i0   = 0#np.argmin( abs(times - t0) )
        iend = i0 + size_subset - 1
        
        tr_full, _, _ = \
            read_data.pre_process_waveform(times*3600., vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
        
        tr, _, _ = \
            read_data.pre_process_waveform(times*3600., vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
        max_amp = max(max_amp, abs(tr.data).max())
        color = 'tab:green' if type == 'arrival' else 'tab:red'
        
        print(event, satellite, station)
        
        if nrows_total > 1:
            ax = axs[nb_rows, iax_plot]
        else:
            ax = axs[iax_plot]
        
        ax.set_title(event.split('_')[0] + '\n' + satellite + '-' + station + '\n' + str(waveform['UT'].iloc[0]), fontsize=6., pad=0.)
        ax.plot(tr.times(), tr.data, color=color, linewidth=1.5)
        ax.set_xlim([tr.times().min(), tr.times().max()])
        #axs_waveform[-1].axvline(offset_time*3600., color='tab:red', alpha=0.5)
        #axs_waveform[-1].text(-0.0, 1.015, alphabet[iax+1]+')', ha='right', va='bottom', transform=axs_waveform[-1].transAxes, 
        #                      bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')

        iax += 1
    
    if nrows_total > 1:
        axs[0,0].set_ylim([-0.5*max_amp, 0.5*max_amp])
        plt.setp(axs[:,:], xticks=[], yticks=[])
    else:
        axs[0].set_ylim([-0.5*max_amp, 0.5*max_amp])
        plt.setp(axs[:], xticks=[], yticks=[])
    
    fig.subplots_adjust(hspace=0.5)
     
    #plt.setp(axs[:,1:], xticks=[], yticks=[])
    plt.savefig(options['DIR_FIGURES'] + 'waveforms_{type}P_{type}N.pdf'.format(type=type_input))

    return list_waveforms

def get_preprocessed_waveforms_for_augmentation_plot(tec_data, tec_data_param, options, nb_windows=3, event='Tohoku_1s', satellite='G26', station='0205', fsz=18., seed=0, min_shift=-100., max_shift=100.):

    import obspy

    np.random.seed(seed)

    ## Extract specific waveform data
    waveform = read_data.get_one_entry(tec_data, station, satellite, event)
    sampling = waveform['sampling'].iloc[0]
    window   = read_data.get_window(sampling, options['window'])
    times = waveform['time_s'].values
    vTEC  = waveform['vTEC'].values
    
    ## Select window parameters
    arrival_time = 5.9 # in UT
    size_subset = np.argmin( abs( (times-times[0]) - window ) )
    t0   = arrival_time - 0.5*window/3600.
    i0   = np.argmin( abs(times/3600. - arrival_time) )
    iend = i0 + size_subset - 1
    
    ## Random parameters
    shifts = np.random.uniform(min_shift, max_shift, nb_windows)
    shifts_idx = [np.sign(shift)*np.argmin( abs(times - times[0] - abs(shift)) ) for shift in shifts]
    snrs   = np.random.uniform(1., 5., nb_windows)
    snrs = [1., 2.5, 5.]
    
    ## Process waveform snippet
    st = obspy.Stream()
    st_perturb = obspy.Stream()
    for iax in range(nb_windows):
    
        tr, _, _ = \
            read_data.pre_process_waveform(times, vTEC, i0+int(shifts_idx[iax]), iend+int(shifts_idx[iax]), window, detrend=False, 
                                           bandpass=[options['freq_min'], options['freq_max']])
        st += tr.copy()
        
        p1 = np.var(tr.data)
        tr.data += np.sqrt(p1/snrs[iax])*np.random.randn(tr.data.size)
        st_perturb += tr.copy()
    
    ## Create plot
    fig, axs = plt.subplots(nrows=2, ncols=nb_windows, sharex=True, sharey=True)
    for itr, (tr, tr_perturb) in enumerate(zip(st, st_perturb)):
        axs[0, itr].plot(tr.times(), tr.data, color='tab:blue')
        axs[1, itr].plot(tr_perturb.times(), tr_perturb.data, color='tab:blue')
        axs[0, itr].set_title(str(shifts[itr]) + ' | ' + str(shifts_idx[itr]))
        axs[1, itr].set_title('{snr:.2f}'.format(snr=snrs[itr]))
        #axs[0, itr].axvline(-shifts[itr], color='red')
        
    axs[0, 0].set_xlim([tr.times().min(), tr.times().max()])
    plt.setp(axs, xticks=[], yticks=[])
    plt.savefig(options['DIR_FIGURES'] + 'data_augmentation.pdf')
        
    ## Plot global waveform
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    ax.axvspan(arrival_time*3600.-0.3*700., arrival_time*3600.+700.*1.3, facecolor='grey', alpha=0.18, zorder=1)
    ax.plot(times, vTEC)
    ax.axvline(arrival_time*3600., color='red')
    plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(options['DIR_FIGURES'] + 'data_augmentation_global.pdf')
    
def plot_scheme_methods(tec_data, tec_data_param, options, event='Tohoku_1s', satellite='G26', station='0205', FP=pd.DataFrame(), FN=pd.DataFrame(), fsz=18., seed=0):

    """
    Plot different stages of algorithm for 2nd figure paper
    """

    ## Extract specific waveform data
    waveform = read_data.get_one_entry(tec_data, station, satellite, event)
    sampling = waveform['sampling'].iloc[0]
    window   = read_data.get_window(sampling, options['window'])
    times = waveform['time_s'].values
    vTEC  = waveform['vTEC'].values
    
    ## Select window parameters
    arrival_time = 5.9 # in UT
    size_subset = np.argmin( abs( (times-times[0]) - window ) )
    t0   = arrival_time - 0.5*window/3600.
    i0   = np.argmin( abs(times/3600. - t0) )
    iend = i0 + size_subset - 1
    
    ## Process waveform snippet
    tr, _, _ = \
        read_data.pre_process_waveform(times, vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
    
    tr_m2, _, _ = \
        read_data.pre_process_waveform(times, vTEC, i0-2*size_subset, iend-2*size_subset, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
    
    tr_m1, _, _ = \
        read_data.pre_process_waveform(times, vTEC, i0-size_subset, iend-size_subset, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
    
    ## Extract features
    type_data = 'noise'
    type      = 'features'
    tduration = options['signal_duration'][event]
    params = tec_data_param.loc[ (tec_data_param['station'] == station) 
                    & (tec_data_param['satellite'] == satellite) 
                    & (tec_data_param['event'] == event), : ]
    if params.size > 0:
        params = params.iloc[0]
    features = read_data.extract_one_feature(times, waveform, params, tduration, i0, iend, window, 
                                             event, satellite, station, type_data, 
                                             options, type=type, add_gaussian_noise=False)
                                            # (times, waveform, params, tduration, i0_arrival_toadd, iend_arrival_toadd, 
                                            #           window, event, satellite, station, type_data, options, i0_noise_toadd=i0_noise_toadd, 
                                            #           iend_noise_toadd=iend_noise_toadd,snr=snr, type=type, add_gaussian_noise=options['add_gaussian_noise'])
    
    ## Compute spectrogram
    spectro_f, spectro = compute_params_waveform.compute_spectrogram(tr.times(), tr.data, options['freq_min'], options['freq_max'], w0=8, freq_time_factor=1)
    fft_f, fft_amp = compute_params_waveform.compute_fft(tr.times(), tr.data)
    
    ## Plot waveforms
    fig = plt.figure(figsize=(10,5))
    offset = 4
    N_cols = 4
    N_rows = 2
    depth  = 4
    gs  = fig.add_gridspec(depth*N_rows+1, N_cols*offset+1)
    #gs  = fig.add_gridspec(depth*N_rows+1+depth*N_rows+1, N_cols*offset+2)
    axs = []
    fsz_small = fsz - 2.
    
    # all vTEC waveform
    i = 0
    axs.append( fig.add_subplot(gs[:depth*N_rows//2, i:i+offset-1]) )
    axs[-1].plot(times/3600., vTEC, zorder=10)
    axs[-1].set_xlim([times.min()/3600., times.max()/3600.])
    axs[-1].axvspan(arrival_time, arrival_time+window/3600., facecolor='grey', alpha=0.18, zorder=1)
    axs[-1].set_title('Select window', fontsize=fsz)
    axs[-1].text(-0.05, 1.0, '1.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    
    axs[-1].text(-0.1, 1.13, 'a)', ha='right', va='bottom', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz+4., fontweight='bold')
    
    # waveform window
    i += offset
    axs.append( fig.add_subplot(gs[:depth*N_rows//2, i:i+offset-1]) )
    axs[-1].plot(tr.times(), tr.data)
    axs[-1].set_xlim([tr.times().min(), tr.times().max()])
    axs[-1].set_title('Preprocessing', fontsize=fsz)
    axs[-1].text(-0.05, 1.0, '2.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    
    # feature extraction
    x, y = 0.95, 0.87
    i += offset
    axs.append( fig.add_subplot(gs[:depth*N_rows//4, i:i+offset//2]) )
    axs[-1].plot(tr.times(), tr.data)
    axs[-1].set_xlim([tr.times().min(), tr.times().max()])
    axs[-1].text(-0.05, 1.0, '3.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    axs[-1].set_title('Extract features', x=1., fontsize=fsz)
    axs[-1].text(x, y, 'i.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz_small)
    
    axs.append( fig.add_subplot(gs[:depth*N_rows//4, i+offset//2:i+offset]) )
    axs[-1].plot(fft_f, fft_amp)
    axs[-1].set_xlim([fft_f.min(), fft_f.max()])
    axs[-1].text(x, y, 'ii.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz_small)
                 
    axs.append( fig.add_subplot(gs[depth*N_rows//4:depth*N_rows//2, i:i+offset//2]) )
    axs[-1].pcolormesh(tr.times(), spectro_f, spectro)
    axs[-1].set_xlim([tr.times().min(), tr.times().max()])
    axs[-1].text(x, y, 'iii.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz_small)
    
    # Classification
    i += offset+1
    axs.append( fig.add_subplot(gs[:depth*N_rows//4, i:i+offset-1]) )
    [s.set_visible(False) for s in axs[-1].spines.values()]
    axs[-1].set_title('Classification', fontsize=fsz)
    axs[-1].text(-0.05, 1.0, '4.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    
    # Plot picker
    #i += offset
    i = 0
    axs.append( fig.add_subplot(gs[depth*N_rows//2+1:depth*N_rows//2+depth*N_rows//4, i:i+offset-1]) )
    axs[-1].text(-0.05, 1.0, '5.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    axs[-1].set_title('Time picking', fontsize=fsz)
    [s.set_visible(False) for s in axs[-1].spines.values()]
    
    axs.append( fig.add_subplot(gs[depth*N_rows//2+depth*N_rows//4+1:depth*N_rows+1, i:i+offset-1]) )
    axs[-1].plot(tr.times()/3600.+t0, tr.data)
    axs[-1].set_xlim([t0, tr.times().max()/3600.+t0])
    axs[-1].axvline(arrival_time, alpha=0.5, color='tab:red')
    
    # Plot heuristic
    i += offset
    axs.append( fig.add_subplot(gs[depth*N_rows//2+1:depth*N_rows//2+depth*N_rows//4+1, i:i+offset-1]) )
    [s.set_visible(False) for s in axs[-1].spines.values()]
    axs[-1].text(-0.05, 1.0, '6.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    axs[-1].set_title('Validation', fontsize=fsz)
    
    axs.append( fig.add_subplot(gs[depth*N_rows//2+depth*N_rows//4+1:depth*N_rows+1, i:i+offset//2]) )
    axs[-1].plot(tr_m2.times(), tr_m2.data)
    axs[-1].set_xlim([tr_m2.times().min(), tr_m2.times().max()])
    axs[-1].text(0.5, 0.8, '$t^{n-2}$\nnoise', ha='center', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1))
    #i += offset//4
    axs.append( fig.add_subplot(gs[depth*N_rows//2+depth*N_rows//4+1:depth*N_rows+1, i+offset//2:i+offset]) )
    axs[-1].plot(tr_m1.times(), tr_m1.data)
    axs[-1].set_xlim([tr_m1.times().min(), tr_m1.times().max()])
    axs[-1].text(0.5, 0.8, '$t^{n-1}$\narrival', ha='center', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1))
    
    # plot association
    i += offset+1
    axs.append( fig.add_subplot(gs[depth*N_rows//2+1:depth*N_rows+1, i:i+offset-1]) )
    plot_blobs_association(axs[-1])
    #axs[-1].plot(times/3600., vTEC, zorder=10)
    #axs[-1].set_xlim([times.min()/3600., times.max()/3600.])
    #axs[-1].axvspan(arrival_time, arrival_time+window/3600., facecolor='grey', alpha=0.18, zorder=1)
    axs[-1].set_title('Association', fontsize=fsz)
    axs[-1].text(-0.05, 1.0, '7.', ha='right', va='top', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    
    """
    ## Add waveforms
    offset_from_arrival = 500.
    h_map = depth*N_rows+2
    h_waveforms = depth*N_rows//2
    offset_h_waveform = 0
    w_waveforms = offset - 1
    plot_per_col = 6
    alphabet = string.ascii_lowercase
    axs_waveform = []
    nb_rows = 0
    
    np.random.seed(seed)
    nb_waveforms = 6
    list_waveforms = []
    for j in range(nb_waveforms):
        i = j#np.random.randint(0,FN.shape[0])
        try:
            entry = FN.iloc[i].event, FN.iloc[i].satellite, FN.iloc[i].station, FN.iloc[i]['arrival-time']/3600., 'arrival'
        except:
            entry = FP.iloc[i].event, FP.iloc[i].satellite, FP.iloc[i].station, FP.iloc[i]['arrival-time']/3600., 'noise'
        list_waveforms.append( entry )
        i = np.random.randint(0,FP.shape[0])
        entry = FN.iloc[i].event, FN.iloc[i].satellite, FN.iloc[i].station, FN.iloc[i]['arrival-time']/3600., 'arrival'
        #entry = FP.iloc[i].event, FP.iloc[i].satellite, FP.iloc[i].station, FP.iloc[i]['arrival-time']/3600., 'noise'
        list_waveforms.append( entry )
    
    iax = 0
    max_amp = 0.
    for (event, satellite, station, t0, type) in list_waveforms[:]:
        
        waveform_orig = read_data.get_one_entry(tec_data, station, satellite, event)
        
        
        #params = tec_data_param.loc[ (tec_data_param['station'] == station) 
        #                & (tec_data_param['satellite'] == satellite) 
        #                & (tec_data_param['event'] == event), : ].iloc[0]
        #arrival_time = params['arrival-time']/3600.
        
        
        ## Process waveform snippet
        #t0s = [(1200./3600., 800./3600.), (300./3600., 100./3600.)]
        #for range_offset in t0s:
        
        #offset_time = random.uniform(range_offset[0], range_offset[1])
        #t0 = arrival_time - offset_time
    
        iax_plot = iax - nb_rows*plot_per_col
        if iax_plot == plot_per_col:
            offset_h_waveform += h_waveforms
            nb_rows += 1
            iax_plot = iax - nb_rows*plot_per_col
    
        axs_waveform.append( fig.add_subplot(gs[h_map+offset_h_waveform:h_map+h_waveforms+offset_h_waveform, 
                                               iax_plot*w_waveforms:iax_plot*w_waveforms+w_waveforms]) )
        
        waveform = waveform_orig.loc[(waveform_orig.UT >= t0) & (waveform_orig.UT <= t0 + window/3600.), :]
        times = waveform['UT'].values
        vTEC  = waveform['vTEC'].values
        size_subset = np.argmin( abs( (times-times[0]) - window ) )
        i0   = 0#np.argmin( abs(times - t0) )
        iend = i0 + size_subset - 1
        
        #print(waveform.shape, waveform_orig.UT.min(), waveform_orig.UT.max(), event, satellite, station, times[0], t0, t0 + window/3600., times[-1])
        tr_full, _, _ = \
            read_data.pre_process_waveform(times*3600., vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
        
        tr, _, _ = \
            read_data.pre_process_waveform(times*3600., vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
        max_amp = max(max_amp, abs(tr.data).max())
        color = 'tab:green' if type == 'arrival' else 'tab:red'
        
        axs_waveform[-1].set_title(event + '\n' + satellite + '-' + station + '\n' + str(waveform['UT'].iloc[0]))
        axs_waveform[-1].plot(tr.times(), tr.data, color=color, linewidth=1.5)
        axs_waveform[-1].set_xlim([tr.times().min(), tr.times().max()])
        #axs_waveform[-1].axvline(offset_time*3600., color='tab:red', alpha=0.5)
        #axs_waveform[-1].text(-0.0, 1.015, alphabet[iax+1]+')', ha='right', va='bottom', transform=axs_waveform[-1].transAxes, 
        #                      bbox=dict(facecolor='w', edgecolor='w', pad=0.1, alpha=0.5), fontsize=15., fontweight='bold')

        iax += 1
    
    for ax in axs_waveform:
        ax.set_ylim([-0.5*max_amp, 0.5*max_amp])
    """
    fig.subplots_adjust(bottom=0.05, left=0.05, right=0.99)
     
    plt.setp(axs, xticks=[], yticks=[])
    fig.savefig(options['DIR_FIGURES'] + 'scheme_ML.pdf')
    plt.close()
    
def plot_FPFN_for_scheme(tec_data, options, nb_waveforms=6, FP=pd.DataFrame(), FN=pd.DataFrame(), fsz=18., seed=0, window=720.):

    """
    Plot preprocess data for Figure 2 below ML architecture
    """
    
    ## Grid setup
    fig = plt.figure(figsize=(10,2.5))
    plot_per_col = 4
    h_waveforms = 1
    w_waveforms = 1
    nb_rows     = 2
    gs  = fig.add_gridspec(h_waveforms*nb_rows, plot_per_col*w_waveforms)
    alphabet = string.ascii_lowercase
    axs_waveform = []
    
    ## Collect waveform data
    np.random.seed(seed)
    list_waveforms = []
    for j in range(nb_waveforms):
        """
        i = j#np.random.randint(0,FN.shape[0])
        try:
            entry = FN.iloc[i].event, FN.iloc[i].satellite, FN.iloc[i].station, FN.iloc[i]['arrival-time']/3600., 'arrival'
        except:
            entry = FP.iloc[i].event, FP.iloc[i].satellite, FP.iloc[i].station, FP.iloc[i]['arrival-time']/3600., 'noise'
        #print(entry)
        list_waveforms.append( entry )
        """
        i = np.random.randint(0,min(FP.shape[0], FN.shape[0]))
        test = np.random.random()
        if test >= 0.5:
            entry = FN.iloc[i].event, FN.iloc[i].satellite, FN.iloc[i].station, FN.iloc[i]['arrival-time']/3600., 'arrival', FN.iloc[i].snr
        else:
            entry = FP.iloc[i].event, FP.iloc[i].satellite, FP.iloc[i].station, FP.iloc[i]['arrival-time']/3600., 'noise', FP.iloc[i].snr
        list_waveforms.append( entry )
    
    print(list_waveforms)
    
    ## Loop over all waveforms
    iax = 0
    max_amp = 0.
    nb_rows = 0
    offset_h_waveform = 0
    for (event, satellite, station, t0, type, snr) in list_waveforms[:]:
        
        waveform_orig = read_data.get_one_entry(tec_data, station, satellite, event)
    
        iax_plot = iax - nb_rows*plot_per_col
        if iax_plot == plot_per_col:
            offset_h_waveform += h_waveforms
            nb_rows += 1
            iax_plot = iax - nb_rows*plot_per_col
    
        axs_waveform.append( fig.add_subplot(gs[offset_h_waveform:h_waveforms+offset_h_waveform, 
                                               iax_plot*w_waveforms:iax_plot*w_waveforms+w_waveforms]) )
        
        waveform = waveform_orig.loc[(waveform_orig.UT >= t0) & (waveform_orig.UT <= t0 + window/3600.), :]
        times = waveform['UT'].values
        vTEC  = waveform['vTEC'].values
        size_subset = np.argmin( abs( (times-times[0]) - window ) )
        i0   = 0#np.argmin( abs(times - t0) )
        iend = i0 + size_subset - 1
        
        tr_full, _, _ = \
            read_data.pre_process_waveform(times*3600., vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
        
        tr, _, _ = \
            read_data.pre_process_waveform(times*3600., vTEC, i0, iend, window, detrend=False, 
                                       bandpass=[options['freq_min'], options['freq_max']])
        max_amp = max(max_amp, abs(tr.data).max())
        color = 'tab:green' if type == 'arrival' else 'tab:red'
        
        p1 = np.var(tr.data)
        tr.data += np.sqrt(p1/snr)*np.random.randn(tr.data.size)
        
        #axs_waveform[-1].set_title(event + '\n' + satellite + '-' + station)
        if iax == 0:
            axs_waveform[-1].plot(tr.times(), tr.data, color='tab:red', linewidth=1.4, label='False positive')
            axs_waveform[-1].plot(tr.times(), tr.data, color='tab:green', linewidth=1.4, label='False negative')
        axs_waveform[-1].plot(tr.times(), tr.data, color=color, linewidth=1.5)
        axs_waveform[-1].set_xlim([tr.times().min(), tr.times().max()])
        
        
        iax += 1
    
    axs_waveform[0].legend(loc='lower right', bbox_to_anchor=(3.4, 0.83),  frameon=False, ncol=2, fontsize=16.)
    axs_waveform[0].text(-0.05, 1.03, alphabet[4] + ')', ha='right', va='bottom', transform=axs_waveform[0].transAxes, fontsize=fsz+4., fontweight='bold')
        
    for ax in axs_waveform:
        ax.set_ylim([-1*max_amp, 1.*max_amp])
    
    fig.subplots_adjust(hspace=0.2)#, bottom=0.05, left=0.05, right=0.99)
    plt.setp(axs_waveform, xticks=[], yticks=[])
    
    plt.savefig(options['DIR_FIGURES'] + 'preprocessed_waveforms.pdf')
    
def build_performance_figure(est, data, optimization, options, reports=pd.DataFrame(), type_ML='forest',
                             features_to_remove = ['W10', 'FT14', 'FT13', 'S6', 'FT12', 'S11', 'FT16', 'S0', 'FT15', 'S1', 'S12']):

    """
    Fiure showing confusion matric, precision/recall for different timw dinows, ROC curve and best-feature distribution
    """

    ## Load reports if not provided
    if reports.size == 0:
        reports = pd.read_csv(options['DIR_FIGURES'] + 'reports.csv', sep=',', header=[0])

    ## Choose input and output data
    input_columns  = [key for key in train_est.data_without_info_columns(data)]
    output_columns = ['type-data']
    
    ## Extract test data
    data_test = data.loc[data['type'] == 'test', input_columns]
    out_test  = data.loc[data['type'] == 'test', output_columns]
    out_pred  = est.predict(data_test)

    #train_est.plot_clustering(data, input_columns, output_columns, options, n_components = 2, perplexity=50, max_elmt=1500)

    ## Setup figure
    #fig = plt.figure(figsize=(10,4))
    fig = plt.figure()
    width  = 6
    height = 6
    offset = 1
    gs = fig.add_gridspec(width+offset, height+offset)
    axs = []
    x, y = -0.12, 1.05
    fsz = 13.
    
    ## Confusion matrix and ROC curve
    axs.append( fig.add_subplot(gs[height//2+offset:, :width//2]) )
    train_est.plot_confusion(est, data_test, out_test, options, type_ML=type_ML, ax=axs[-1])
    axs[-1].text(x, y, 'c)', ha='right', va='bottom', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
                 
    axs.append( fig.add_subplot(gs[height//2+offset:, width//2+offset:]) )
    train_est.plot_roc_curve(est, data_test, out_test, options, xlim=[0., 1.], ylim=[0.8, 1.], type_ML=type_ML, ax=axs[-1], vmax=0.9)
    axs[-1].text(x, y, 'd)', ha='right', va='bottom', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
                 
         
    ## Comparison different window sizes
    axs.append( fig.add_subplot(gs[:height//2, :width//2]) )
    options_list = detector.build_option_list(optimization, options)
    detector.plot_metrics_windows(reports, options_list, ax=axs[-1])
    axs[-1].text(x, y, 'a)', ha='right', va='bottom', transform=axs[-1].transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
    
    ## Distribution features
    axs_features = np.zeros((3,3), dtype=object)
    for i in range(3):
        for j in range(3):
            axs_features[i, j] = fig.add_subplot(gs[j, width//2+i+offset]) 
    
    train_est.plot_features_vs_features(data, input_columns, est, options, Nfeatures = 3, axs=axs_features, features_to_remove=features_to_remove)
    axs_features[0,0].text(-0.45, 1.05, 'b)', ha='right', va='bottom', transform=axs_features[0,0].transAxes, 
                          bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz, fontweight='bold')
                 
    plt.savefig(options['DIR_FIGURES'] + 'performance_figure_paper.pdf')
  
def plot_full_figure_cost(event, satellite, station, tec_data, time_elapsed, time_elapsed_station, window, options, fontsize=15., time_max=10.):

    """
    Plot Figure paper showing cost per station and across network
    """

    ## Load waveform data
    #time_elapsed.to_csv('time_elapsed.csv', header=True, index=False)
    #selected_waveform = associations.loc[associations.index==index].iloc[0]
    #event, satellite, station = selected_waveform.event, selected_waveform.satellite, selected_waveform.station
    #try:
    #    station = "{:04d}".format(station)
    #except:
    #    pass
    waveform = tec_data.loc[(tec_data['event'] == event) & (tec_data['satellite'] == satellite) & (tec_data['station'] == station), :]
    
    ## Setup figure
    fig = plt.figure(figsize=(6,6))
    gs = fig.add_gridspec(5, 1)

    ## Association plots
    axs_association = []
    axs_association.append( fig.add_subplot(gs[0, :]) )
    for iax in range(2,5):
        axs_association.append( fig.add_subplot(gs[iax, :]) )

    ## Single station plot
    ax_station = [fig.add_subplot(gs[1, :])]

    plot_time_cost_associations(time_elapsed, waveform, window, options, fontsize=fontsize, time_max=time_max, axs=axs_association)
    plot_time_cost_per_station(time_elapsed_station, waveform, window, options, fontsize=fontsize, time_max=time_max, plot_waveform=False, axs=ax_station)
    ax_station[0].set_xlim(axs_association[0].get_xlim())
    ax_station[0].set_ylim([0., 1.])
    ax_station[0].set_xlabel('')
    plt.setp(ax_station, xticks=[])
    
    alphabet = ['a', 'c', 'd', 'e', 'b']
    for iax, ax in enumerate(axs_association + ax_station):
        ax.text(-0.1, 1., alphabet[iax] + ')', ha='right', va='bottom', 
                transform=ax.transAxes, fontsize=15., fontweight='bold')
  
    plt.savefig(options['DIR_FIGURES'] + 'cost_figure.pdf')
  
def plot_association_no_signal_day(tec_data, associations_time_steps_all, options, thresh_count=3, subsampling_time=10, fontsize=12., plot_time=False, file_name='no_signal_day_association.pdf'):

    ## Extract evolution of association classes
    list_times = associations_time_steps_all.time_association.unique()
    list_times = list_times[::subsampling_time]
    
    print('1111')
    
    associations_time_steps_ = associations_time_steps.groupby(['time_association','association_no','satellite','station','arrival_class']).first().reset_index()
    counts_assoc = associations_time_steps_.groupby(['time_association','association_no']).count().reset_index()
    counts_assoc = counts_assoc.loc[counts_assoc.time >= thresh_count]
    cmap = sns.color_palette("rocket", as_cmap=True, n_colors=len(counts_assoc.association_no.unique()))
    
    print('2222')
    print(counts_assoc)
    
    loc_assoc = associations_time_steps.loc[associations_time_steps.association_no.isin(counts_assoc.association_no.unique()) & associations_time_steps.time_association.isin(list_times)]
    loc_assoc['time_association'] /= 3600.
    
    assoc_max = counts_assoc.loc[(counts_assoc.time == counts_assoc.time.max()), 'association_no'].values[0]
    list_waveforms = associations_time_steps.loc[(associations_time_steps.association_no == assoc_max), ['satellite', 'station', 'time-corrected']]
    if list_waveforms.station.unique().size < 3:
        list_waveforms = associations_time_steps.groupby(['satellite', 'station']).first().reset_index()[['satellite', 'station', 'time-corrected']].values[:3]
    else:
        list_waveforms = associations_time_steps.loc[(associations_time_steps.association_no == assoc_max), ['satellite', 'station', 'time-corrected']].values
    print(list_waveforms)
    
    print('3333')
    
    ## Setup Figure
    fig = plt.figure()
    gs = fig.add_gridspec(6, 2)
    
    alphabet = string.ascii_lowercase
        
    if plot_time:
    
        idx = loc + 1
        
        ax = fig.add_subplot(gs[0:3, 1])
        last_step = loc_assoc.loc[loc_assoc.time_association==loc_assoc.time_association.max()]
        ax.scatter(loc_assoc.lon, loc_assoc.lat, c=loc_assoc.association_no, cmap=cmap, s=10); 
    
    else:
        done = []
        for loc in range(3):
            
            ax = fig.add_subplot(gs[loc, 0])
            print('-->', list_waveforms[loc])
            try:
                station = "{:04d}".format(list_waveforms[loc][1])
            except:
                station = list_waveforms[loc][1]
            satellite, time = list_waveforms[loc][0], list_waveforms[loc][2]
            if [satellite, station] in done:
                continue
            
            waveform = tec_data.loc[(tec_data.station==station) & (tec_data.satellite==satellite)]
            ax.axvline(time/3600., color='tab:red', linestyle='--', alpha=0.5)
            ax.plot(waveform.UT, waveform.vTEC)
            ax.set_xlim([time/3600.-500./3600., time/3600.+500./3600.])
            ax.text(0.5, 0.9, satellite + ' - ' + station, fontsize=fontsize, ha='center', va='top', transform=ax.transAxes)
            
            if loc == 0:
                ax.xaxis.tick_top()
                ax.set_xlabel('Time (UT)')
                ax.xaxis.set_label_position("top")
                ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            
            else:
                ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            
            ax.text(0., 0.95, alphabet[loc] + ')', ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize+4., fontweight='bold')
        
    idx = loc + 1
        
    ax = fig.add_subplot(gs[0:3, 1])
    last_step = loc_assoc.loc[loc_assoc.time_association==loc_assoc.time_association.max()]
    ax.scatter(loc_assoc.lon, loc_assoc.lat, c=loc_assoc.association_no, cmap=cmap, s=10); 
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.set_ylabel('Latitude', fontsize=fontsize)
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.xaxis.set_label_position("top")
    ax.text(0., 0.95, alphabet[idx] + ')', ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize+4., fontweight='bold')
    
    idx += 1
    ax = fig.add_subplot(gs[3:, :])
    sns.histplot(data=loc_assoc, x="time_association", hue="association_no", multiple="stack", palette=cmap, legend=False, bins=list_times.size, ax=ax)
    ax.set_xlabel('Time (UT)', fontsize=fontsize)
    ax.set_ylabel('Number of detections\nper class', fontsize=fontsize)
    ax.text(0., 0.95, alphabet[idx] + ')', ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize+4., fontweight='bold')
    
    plt.savefig(options['DIR_FIGURES'] + file_name)
  
def find_all_detections_nosignalday(folder, event, est, est_picker, data, options, 
                                    nb_CPU=16, nb_picks=4, max_waveform=10, load=False,
                                    return_only_tec_data=False, sampling_threshold=31.):

    """
    Compute arrival times for a list of events/satellites/stations
    """
    
    ## If only TEC data requested and if csv exists we return it right away
    observations_all, probas_all = pd.DataFrame(), pd.DataFrame()
    if (return_only_tec_data or load) and os.path.exists(folder + 'tec_data.csv'):
        dtype = {'epoch': int, 'UT': float, 'LOS': float, 'az': float, 'lat': float, 'lon': float, 'sTEC': float, 'vTEC': float, 
             'time_s': float, 'station': str, 'satellite': str, 'doy': int, 'year': int, 'event': str, 'sampling': float}
        tec_data = pd.read_csv(folder + 'tec_data.csv', header=[0], dtype=dtype)
    
    if return_only_tec_data and os.path.exists(folder + 'tec_data.csv'):
        return observations_all, probas_all, tec_data
        
    if load:
        observations_all = pd.read_csv(folder + 'observations_long_day.csv', header=[0])
        probas_all = pd.read_csv(folder + 'probas_long_day.csv', header=[0])
    
    else:
        
        ## Collect all files to process
        all_files = []
        nb_waveforms = 0
        for  subdir, dirs, files in os.walk(folder):
            for file in files:
            
                nb_waveforms += 1
                
                if not 'rtTEC' in file or nb_waveforms > max_waveform:
                    continue
                
                station = file.split('_')[1]
                satellite = file.split('_')[2]
                
                filepath_rtTEC = subdir +  file
                all_files.append( filepath_rtTEC )
        all_files = np.array(all_files)
        nb_files = all_files.size
        
        ## Create interface
        find_subset_detections_nosignalday_partial = \
            partial(find_subset_detections_nosignalday, event, est, est_picker, data, options, nb_picks, return_only_tec_data, sampling_threshold)

        N = min(nb_CPU, nb_files)
        ## If one CPU requested, no need for deployment
        if N == 1:
            observations_all, probas_all, tec_data = find_subset_detections_nosignalday_partial(all_files)

        ## Otherwise, we pool the processes
        else:
            
            step_idx =  nb_files//N
            list_of_lists = []
            for i in range(N):
                idx = np.arange(i*step_idx, (i+1)*step_idx)
                if i == N-1:
                    idx = np.arange(i*step_idx, nb_files)
                list_of_lists.append( all_files[idx] )
            
            with get_context("spawn").Pool(processes = N) as p:
                results = p.map(find_subset_detections_nosignalday_partial, list_of_lists)

            observations_all = pd.DataFrame()
            probas_all       = pd.DataFrame()
            tec_data = pd.DataFrame()
            for result in results:
                observations_all = observations_all.append( result[0] )
                probas_all       = probas_all.append( result[1] )
                tec_data         = tec_data.append( result[2] )
            
        if not return_only_tec_data:
            observations_all.reset_index(drop=True, inplace=True)
            observations_all.to_csv(folder + 'observations_long_day.csv', header=True, index=False)
            
            probas_all.reset_index(drop=True, inplace=True)
            probas_all.to_csv(folder + 'probas_long_day.csv', header=True, index=False)
        
        tec_data.reset_index(drop=True, inplace=True)
        tec_data.to_csv(folder + 'tec_data.csv', header=True, index=False)
    
    return observations_all, probas_all, tec_data
  
def find_subset_detections_nosignalday(event, est, est_picker, data, options, nb_picks, return_only_tec_data, sampling_threshold, all_files):

    """
    Find all detections over waveforms where there is no signal
    """

    ## Features used for training
    input_columns  = [key for key in train_est.data_without_info_columns(data)]

    observations_all = pd.DataFrame()
    probas_all = pd.DataFrame()
    tec_data   = pd.DataFrame()
    for filepath_rtTEC in all_files:
    
        #nb_waveforms += 1
        
        #if not 'rtTEC' in file or nb_waveforms > max_waveform:
        #    continue
        
        file = filepath_rtTEC.split('/')[-1]
        station = file.split('_')[1]
        satellite = file.split('_')[2]
        
        #filepath_rtTEC = subdir + os.sep + file
        
        ## Collect long-waveform data
        waveform = read_data.load_one_rTEC(filepath_rtTEC)
        
        if waveform.sampling.iloc[0] > sampling_threshold:
            continue
        
        tec_data = tec_data.append( waveform )
        
        ## If only tec_data required (and not detections) we skip the rest of the loop
        if return_only_tec_data:
            continue
        
        waveform_param = pd.DataFrame()
        first_row      = waveform.iloc[0]
        event, satellite, station = first_row.event, first_row.satellite, first_row.station
        time_end = waveform.time_s.max()
        
        #print(time_end, waveform_param, event, satellite, station, input_columns)
        if not(satellite == 'R12' and station == 'lesv'):
            continue
        
        ## Get probas    
        observations_dict = train_est.process_timeseries_with_forest(time_end, est, waveform, waveform_param, 
                                                 event, satellite, station, 
                                                 input_columns, options, plot_probas=False, 
                                                 type='features', est_picker=est_picker,
                                                 use_STA_LTA_for_picking=False,
                                                 return_all_waveforms_used=False,
                                                 nb_picks=nb_picks, figsize=(),
                                                 determine_elapsed_time=False,zscore_threshold = 1e10)
            
        observations = observations_dict['detections']
        observations['event'] = event
        observations['station'] = station
        observations['satellite'] = satellite
        observations_all = observations_all.append(observations)
        
        probas = observations_dict['probas']
        probas['event'] = event
        probas['station'] = station
        probas['satellite'] = satellite
        probas_all = probas_all.append(probas)
                    
    tec_data.reset_index(drop=True, inplace=True)
    
    return observations_all, probas_all, tec_data
  
def plot_time_cost_per_station(time_elapsed, waveform, window, options, fontsize=15., time_max=0.6, plot_waveform=False, axs=[], label_offset=0):

    """
    Plot time evolution of computational cost of detection per station
    """
    
    time_elapsed.reset_index(inplace=True, drop=True)
    event = '_'.join(waveform['event'].iloc[0].split('_')[:-1])
    satellite = waveform['satellite'].iloc[0]
    station = waveform['station'].iloc[0]
    list_events = get_list_events()
    time = list_events[event][-1]
    event_time = UTCDateTime(time)
    
    event_time -= UTCDateTime(event_time.year, event_time.month, event_time.day)
    event_time /= 60.
    tmin, tmax = waveform.time_s.min()/60.-event_time, waveform.time_s.max()/60.-event_time
    
    ## Setup figure
    new_figure = False
    iax = 0
    if not axs:
        new_figure = True
        fig = plt.figure()
        gs = fig.add_gridspec(3, 1)
    
        ax = fig.add_subplot(gs[0, :])
    
    alphabet = string.ascii_lowercase
    if plot_waveform:
        if not new_figure:
            ax = axs[iax]
            iax += 1
        
        ax.set_title(event, pad=20, fontsize=fontsize)
        ax.text(0.5, 1.02, ' satellite ' + satellite + ' - station '+ station, fontsize=fontsize-3., ha='center', va='bottom', transform = ax.transAxes)
        
        ax.plot(waveform.time_s/60.-event_time, waveform.vTEC)
        ax.set_xlim([tmin, tmax])
        ax.set_ylabel('vTEC', fontsize=fontsize)
        ax.set_xticklabels([])
        ax.text(0.98, 0.95, alphabet[label_offset] + ')', ha='right', va='bottom', 
                    transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w', pad=0.1), 
                    fontsize=15., fontweight='bold')
        label_offset += 1
        
    cmap = sns.color_palette("Set2")
    if not new_figure:
        print(iax, window, event_time)
        ax = axs[iax]
    else:
        ax = fig.add_subplot(gs[1:, :])
    
    ax.stackplot((time_elapsed.time.values+window-30.)/60.-event_time,
                 time_elapsed.feature.values, 
                 time_elapsed.classification.values, 
                 time_elapsed['time-picking'].values, 
                 time_elapsed.validation.values,
                 labels=time_elapsed.loc[:, ~time_elapsed.columns.isin(['time'])].columns.tolist(),
                 colors=cmap,
                 baseline='zero')
    ax.text(0.01, 0.95, 'Cost (s) single station', fontsize=fontsize-3., ha='left', va='top', transform = ax.transAxes)
    
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([0., time_max])
    ax.set_xlabel('Time since earthquake (min)', fontsize=fontsize)
    #ax.set_ylabel('Cost (s)', fontsize=fontsize)
    ax.legend(loc='lower right', bbox_to_anchor=(1., 1.05), ncol=2, frameon=False, labelspacing=0.04, handlelength=0.4, handletextpad=0.2)
    #ax.text(-0.1, 1., alphabet[label_offset] + ')', ha='right', va='bottom', 
    #            transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w', pad=0.1), 
    #            fontsize=15., fontweight='bold')
                
                
    if new_figure:
        fig.subplots_adjust(hspace=0.05)
        plt.savefig(options['DIR_FIGURES'] + 'computational_cost.pdf')
    
def plot_time_cost_associations(time_elapsed, waveform, window, options, fontsize=15., time_max=10., axs=[]):

    """
    Plot time evolution of computational cost to associate arrivals together
    """
    
    time_elapsed.reset_index(inplace=True, drop=True)
    event = '_'.join(waveform['event'].iloc[0].split('_')[:-1])
    satellite = waveform['satellite'].iloc[0]
    
    station = waveform['station'].iloc[0]
    
        
    list_events = get_list_events()
    time = list_events[event][-1]
    event_time = UTCDateTime(time)
    
    event_time -= UTCDateTime(event_time.year, event_time.month, event_time.day)
    event_time /= 60.
    tmin, tmax = (time_elapsed.time.values/60.-event_time).min(), waveform.time_s.max()/60.-event_time
    
    
    ## Setup figure
    new_figure = False
    if not axs:
        new_figure = True
        fig = plt.figure()
        axs = []
        gs = fig.add_gridspec(4, 1)
        ax = fig.add_subplot(gs[0, :])
        axs.append( ax )
    else:
        ax = axs[0]
    
    ax.set_title(event, pad=10, fontsize=fontsize)
    #ax.text(0.5, 1.02, ' satellite ' + satellite + ' - station '+ station, fontsize=fontsize-3., ha='center', va='bottom', transform = ax.transAxes)
    ax.text(0.01, 0.95, 'satellite ' + satellite + ' - station '+ station, fontsize=fontsize-3., ha='left', va='top', transform = ax.transAxes)
    
    ax.plot(waveform.time_s/60.-event_time, waveform.vTEC)
    ax.set_xlim([tmin, tmax])
    ax.set_ylabel('vTEC', fontsize=fontsize)
    ax.set_xticklabels([])
    
    cmap_line = sns.color_palette("dark")
    cmap_face = sns.color_palette("pastel")
    
    id_color = 3
    if new_figure:
        ax = fig.add_subplot(gs[1, :])
        axs.append( ax )
    else:
        ax = axs[1]
    ax.fill_between(time_elapsed.time.values/60.-event_time, time_elapsed.cost.values, facecolor=cmap_face[id_color], edgecolors=cmap_line[id_color])
    ax.text(0.01, 0.95, 'Cost (s) across network', fontsize=fontsize-3., ha='left', va='top', transform = ax.transAxes)
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([0., time_max])
    #ax.set_ylabel('Cost (s)', fontsize=fontsize)
    ax.set_xticklabels([])
    
    id_color = 4
    if new_figure:
        ax = fig.add_subplot(gs[2, :])
        axs.append( ax )
    else:
        ax = axs[2]
    ax.fill_between(time_elapsed.time.values/60.-event_time, time_elapsed.detections_in_time_new.values, facecolor=cmap_face[id_color], edgecolors=cmap_line[id_color])
    ax.set_xlim([tmin, tmax])
    #ax.set_ylabel('Nb of new\ndetections', fontsize=fontsize)
    ax.text(0.01, 0.95, 'Nb of new detections', fontsize=fontsize-3., ha='left', va='top', transform = ax.transAxes)
    ax.set_xticklabels([])
    
    id_color = 5
    if new_figure:
        ax = fig.add_subplot(gs[3, :])
        axs.append( ax )
    else:
        ax = axs[3]
    ax.fill_between(time_elapsed.time.values/60.-event_time, time_elapsed.nb_detections.values, facecolor=cmap_face[id_color], edgecolors=cmap_line[id_color])
    ax.set_xlabel('Time since earthquake (min)', fontsize=fontsize)
    #ax.set_ylabel('Nb of associated\ndetections', fontsize=fontsize)
    ax.text(0.01, 0.95, 'Nb of associated detections', fontsize=fontsize-3., ha='left', va='top', transform = ax.transAxes)
    ax.set_xlim([tmin, tmax])
    
    if new_figure:
        alphabet = string.ascii_lowercase
        for iax, ax in enumerate(axs):
            ax.text(1., 0.95, alphabet[iax] + ')', ha='left', va='bottom', 
                    transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w', pad=0.1), 
                    fontsize=15., fontweight='bold')
    
        fig.align_ylabels(axs[:])
        fig.subplots_adjust(hspace=0.15, right=0.95, left=0.1)
        
        plt.savefig(options['DIR_FIGURES'] + 'computational_cost_association.pdf')
      
def detection_summary_long_waveform(est, data, filepath_rtTEC, filepath_param, options,
                                    est_picker=None, nb_picks=4, figsize=(10, 4)):

    """
    Plot long waveform and detection probabilities for paper
    """
    
    ## Features used for training
    input_columns  = [key for key in train_est.data_without_info_columns(data)]
    
    ## Collect long-waveform data
    waveform       = read_data.load_one_rTEC(filepath_rtTEC)
    waveform_param = read_data.load_one_param(filepath_param)
    waveform_param = waveform_param.iloc[0]
    first_row      = waveform.iloc[0]
    event, satellite, station = first_row.event, first_row.satellite, first_row.station
    time_end = waveform.time_s.max()
    
    ## Plot timeseries and proba
    observations = train_est.process_timeseries_with_forest(time_end, est, waveform, waveform_param, 
                                                 event, satellite, station, 
                                                 input_columns, options, plot_probas=True, 
                                                 determine_elapsed_time=True,
                                                 type='features', est_picker=est_picker,
                                                 use_STA_LTA_for_picking=False,
                                                 return_all_waveforms_used=False,
                                                 nb_picks=nb_picks, figsize=figsize,
                                                 add_label='a)', add_inset=True)

    return observations

def process_one_waveform(est, data, waveform_in, waveform_param, options, shift=1000.):

    input_columns  = [key for key in train_est.data_without_info_columns(data)]

    arrival_time = waveform_param['arrival-time'].iloc[0]
    waveform = waveform_in
    info = waveform.iloc[0]
    event, satellite, station = info.event, info.satellite, info.station
    waveform = waveform.loc[(waveform.time_s >= arrival_time-shift) & (waveform.time_s <= arrival_time+shift), :]
    time_end = waveform.time_s.max()
    
    observations = train_est.process_timeseries_with_forest(time_end, est, waveform, waveform_param, 
                                                            event, satellite, station, 
                                                            input_columns, options, plot_probas=True, 
                                                            type='features', figsize=(10,4), add_label='e)')
    
    return observations
    
def detection_summary_volcano(est, data, filepath_rtTEC, options, est_picker=None, 
                              nb_picks=4, figsize=(10, 4), determine_elapsed_time=False):

    """
    Plot volcano data and detection probabilities for paper
    """
    
    ## Features used for training
    input_columns  = [key for key in train_est.data_without_info_columns(data)]
    
    ## Collect long-waveform data
    waveform       = read_data.load_one_rTEC(filepath_rtTEC)
    waveform_param = pd.DataFrame()
    first_row      = waveform.iloc[0]
    event, satellite, station = first_row.event, first_row.satellite, first_row.station
    time_end = waveform.time_s.max()
    
    ## Plot timeseries and proba                            
    results = train_est.process_timeseries_with_forest(time_end, est, waveform, waveform_param, 
                                                 event, satellite, station, 
                                                 input_columns, options, plot_probas=True, 
                                                 type='features', est_picker=est_picker,
                                                 use_STA_LTA_for_picking=False,
                                                 return_all_waveforms_used=False,
                                                 nb_picks=nb_picks, figsize=figsize,
                                                 determine_elapsed_time=determine_elapsed_time)
    
    if determine_elapsed_time:
        return results[0], results[1]
    else:
        return results
 
def change_event_name(x):
    
    """
    Remove sampling in event name
    """
    
    x['event_corrected'] = '_'.join(x['event'].split('_')[:-1])
    return x
    
def correct_arrival_times(detections, offset, nb_pts_picker=-1, quantile_threshold=0.5):

    """
    Correct predicted arrival times based on probabilities
    """

    grouped_detections = detections.groupby(['event', 'satellite', 'station', 'arrival_class'])
    for group, detection in grouped_detections:
    
        times  = detection.time.values
        if detection['predicted-time'].iloc[0] > -1:
            predicted_times = detection['predicted-time'].values
            if nb_pts_picker > -1:
                predicted_times = predicted_times[:nb_pts_picker]
            mean_time = np.quantile(predicted_times, q=quantile_threshold)
            detections.loc[detections.index.isin(detection.index), 'time-corrected'] = mean_time
            
        else:
            times += offset
            probas = detection.proba.values
            diff_probas = np.diff(probas)
            
            diff_probas_pos = np.where(diff_probas>=0)[0]
            first_cluster   = []
            for idiff, diff in enumerate(np.diff(diff_probas_pos)):
                if diff <= 2:
                    first_cluster.append( diff_probas_pos[idiff] )
            
            if len(first_cluster) > 0:
                
                #diff_probas_pos = diff_probas_pos[first_cluster]
                arrival_time = np.mean(times[first_cluster])
                detections.loc[detections.index.isin(detection.index), 'time-corrected'] = arrival_time
     
            else:
                detections.loc[detections.index.isin(detection.index), 'time-corrected'] = times[0] + offset
 
def get_first_detections(detections, offset=100., nb_pts_picker=-1, quantile_threshold=0.5):

    """
    Collect arrival time for each event/satellite/station and compute error between true and predicted
    """

    ## Compute right arrival time
    #detections['time-corrected'] = detections['time'] + offset
    correct_arrival_times(detections, offset, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
    
    ## Select best arrival for each wavetrain
    detections['error'] = detections['true-arrival-time'] - detections['time-corrected']
    detections['error-abs'] = abs(detections['error'])
    first_detections = detections.loc[detections.groupby(['event', 'satellite', 'station'])['error-abs'].idxmin()].reset_index(drop=True)
    #grouped_first_detections = first_detections.groupby(['event', 'satellite', 'station'])['error']
    #mask = grouped_first_detections.transform(lambda x: abs(x) == abs(x).min()).astype(bool)
    #first_detections = first_detections.loc[mask].reset_index(drop=True)
    
    ## Change event name
    first_detections = first_detections.apply(change_event_name, axis=1)
    
    return first_detections
 
def create_arrival_time_plot(detections_in, options, offset=360., ax=None, fsz=20., fsz_labels=18., 
                             nb_pts_picker=-1, quantile_threshold=0.5):

    """
    Box plot for error in arrival time picking betwen RF and true arrivals
    """

    """
    l_detect=first_detections.loc[first_detections.event=='Tohoku_1s', :].sort_values(by='error')
    item=l_detect.iloc[1]
    aa=tec_data.loc[(tec_data.station == item['station']) & (tec_data.event==item['event']), :]
    proba=probas.loc[(probas.station == item['station']) & (probas.event==item['event']), :]
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].plot(aa.time_s, aa.vTEC); 
    axs[0].axvline(item['true-arrival-time'], color='red'); 
    axs[0].axvline(item['time-corrected']); 
    axs[1].scatter(proba.time, proba.proba)
    plt.show()
    """
    
    detections = detections_in.copy()
    detections.reset_index(inplace=True, drop=True)
    first_detections = get_first_detections(detections, offset=offset, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
    
    ## Setup figure
    new_figure = False
    if ax == None:
        new_figure = True
        fig = plt.figure()
        ax  = fig.add_subplot(111)
    
    sns.boxplot(x="error", y="event_corrected", data=first_detections, ax=ax, orient='h', 
                showmeans=True, meanprops={'markerfacecolor':'tab:red', 'markeredgecolor':'black'}, 
                showfliers=False, palette='flare')
    ax.set_xlabel('Arrival-time error (s) $t_{true} - t_{pred}$', fontsize=fsz)
    ax.set_ylabel('Event name', fontsize=fsz)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsz_labels)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsz_labels)

    ax.axvline(0, color='black', linewidth=3, linestyle=':', alpha=0.8)
    ax.text(-0.29, 1.05, 'b)', ha='right', va='bottom', transform=ax.transAxes, 
            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz+4, fontweight='bold')

    ax.set_xlim([-300, 300])

    if new_figure:
        fig.subplots_adjust(left=0.3, right=0.95, bottom=0.16)
        fig.savefig(options['DIR_FIGURES'] + 'boxplot_arrival_time_error.pdf')
 
    return first_detections
 
def create_figure_nb_points_picker(detections, l_nb_pts_picker, sampling, options, detections_all=pd.DataFrame(), fsz=22., fsz_labels=18.,                                   quantile_threshold=0.5, plot_event_lines=False):
                                      
    """
    Create figure showing average arrival time error vs number of time steps waited to pick arrival
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    
    ## Compute errors as a function of nb_pts_picker
    offset = 500. # dummy value since we use a picker
    if detections_all.size == 0:
        for nb_pts_picker in l_nb_pts_picker:
            first_detections = get_first_detections(detections, offset=offset, nb_pts_picker=nb_pts_picker, quantile_threshold=quantile_threshold)
            first_detections['nb_pts_picker'] = nb_pts_picker
            detections_all = detections_all.append( first_detections )
        detections_all['nb_pts_picker'] = (detections_all['nb_pts_picker'] - 1) * sampling
    
    ## Plot error bars
    if plot_event_lines:
        sns.lineplot(data=detections_all, x="nb_pts_picker", y="error", hue='event_corrected', palette='flare', ci=None, ax=ax, alpha=0.3)
    sns.lineplot(data=detections_all, x="nb_pts_picker", y="error", color='tab:red', ax=ax, ci=None, linewidth=3.5)
    bounds = detections_all.groupby('nb_pts_picker')['error'].quantile((0.25,0.75)).unstack()
    ax.fill_between(x=bounds.index,y1=bounds.iloc[:,0],y2=bounds.iloc[:,1],alpha=0.1, color='tab:red')
    
    ax.set_xlabel('Time since first detection (s)', fontsize=fsz)
    ax.set_ylabel('Arrival-time error (s)\n$t_{true} - t_{pred}$', fontsize=fsz)
    ax.set_xlim([detections_all['nb_pts_picker'].min(), detections_all['nb_pts_picker'].max()])
    ax.axhline(0, color='black', linewidth=3, linestyle=':', alpha=0.8)
    
    if plot_event_lines:
        ax.legend(handles=ax.legend_.legendHandles, 
              labels=[t.get_text() for t in ax.legend_.texts],
              title=ax.legend_.get_title().get_text(),
              fontsize=fsz_labels-2, labelspacing=0.1,
              frameon=False, ncol=2)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsz_labels)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsz_labels)
    
    ax.text(-0.22, 1.05, 'c)', ha='right', va='bottom', transform=ax.transAxes, 
            bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=fsz+4, fontweight='bold')
    
    ax.set_ylim([-300, 300])
    if not plot_event_lines:
        ax.set_ylim([-50, 300])
        
    fig.subplots_adjust(left=0.25, bottom=0.16, right=0.95)
    
    fig.savefig(options['DIR_FIGURES'] + 'sensitivity_error_picker_delay.pdf')
    
    return detections_all
 
# utils_paper.measure_sensitivity_nb_points_empirical(tec_data_param, probas, [3,4,5], [3], options, window=720.)
def measure_sensitivity_nb_points_empirical(tec_data_param, tec_data, probas, l_nb_for_class, l_nb_for_end_class, \
                                            options, window=720., sampling=30.):

    """
    Compute list of detections based for a given probability list 
    and varying heuristic parameters nb_for_class, nb_for_end_clas
    """

    grouped_probas = probas.groupby(['event', 'satellite', 'station'])
    
    results = pd.DataFrame()
    for nb_for_class in l_nb_for_class:
        for nb_for_end_class in l_nb_for_end_class:
            for group, proba in grouped_probas:
                event, satellite, station = group
                
                print(group)
                
                waveform = tec_data.loc[ (tec_data['station'] == station) 
                        & (tec_data['satellite'] == satellite) 
                        & (tec_data['event'] == event), : ]
                
                detection_param = tec_data_param.loc[ (tec_data_param['station'] == station) 
                        & (tec_data_param['satellite'] == satellite) 
                        & (tec_data_param['event'] == event), : ]
                   
                ## Find arrival time if there is one
                true_arrival = -1
                duration     = -1
                if detection_param.size > 0:
                    true_arrival = detection_param.iloc[0]['arrival-time'] 
                    duration     = options['signal_duration'][event]
                
                detection = \
                    train_est.compute_arrival_time(proba, window, nb_for_class=nb_for_class, 
                                                  nb_for_end_class=nb_for_end_class)
                
                if detection.size == 0:
                    detection['arrival_class'] = []
                
                t0, tend = waveform.time_s.min(), waveform.time_s.max()
                times = np.arange(t0, tend - window + sampling, sampling)
                result_RF = detector.compute_metric_FTPN_detector(times, window, detection, true_arrival, duration)
                result_RF['arrival-time'] = true_arrival
                
                result_RF['event'] = event                                 
                result_RF['satellite'] = satellite
                result_RF['station'] = station
                result_RF['nb_for_class'] = nb_for_class
                result_RF['nb_for_end_class'] = nb_for_end_class
                
                results = results.append( result_RF.copy() )
                
    bp()
               
    return results
 
#def create_figure_nb_points_empirical(detections, probas, l_nb_for_class, l_nb_for_end_class, \
#                                      options, window=720.):
def create_figure_nb_points_empirical(detections_all, options, window=720., offset=360.):
                                      
    #detections_all = measure_sensitivity_nb_points_empirical(probas, l_nb_for_class, l_nb_for_end_class, \
    #                                        options, window=720.)
    
    N_parameters = detections_all['nb_for_class'].unique().size
    
    fig, axs = plt.subplots(nrows=1, ncols=N_parameters, sharex=True, sharey=True)
    
    grouped_detections = detections_all.groupby(['nb_for_class'])
    for igroup, (group, detections) in enumerate(grouped_detections):
        create_arrival_time_plot(detections, options, offset=offset, ax=axs[igroup])
    
    plt.show()
 
def plot_error_wave_picker(reports, options, axs=None, return_fig=False, metric='RMSE', fontsize=15.):

    """
    Plot a specific metric for wave picker over testing and training datasets
    """

    ## Plot results
    new_figure = False
    if axs == None and not return_fig:
        new_figure = True
        fig = plt.figure(figsize=(10,4))
        axs = []
        axs.append( fig.add_subplot(121) )
        axs.append( fig.add_subplot(122) )
        fig.subplots_adjust(bottom=0.1)
    
    #list_datasets = ['train', 'test']
    #for idataset, dataset in enumerate(list_datasets):
    list_balanced = [True, False]
    if len(axs) == 1 or return_fig:
        list_balanced = [True]
        
    for idataset, balanced in enumerate(list_balanced):
        add_ax = {}
        if not return_fig:
            add_ax = {'ax': axs[idataset]}
        scores_plot = reports.loc[(reports['balanced_classes'] == balanced) & (reports['dataset'] == 'test')]
        scores_plot = scores_plot.pivot("window", "max_deviation", metric)
        xticklabels = scores_plot.columns.tolist()
        yticklabels = scores_plot.index.tolist()
        yticklabels = [int(tick) for tick in yticklabels]
        add_ax['xticklabels'] = xticklabels
        add_ax['yticklabels'] = yticklabels
        metric_plot = sns.heatmap(scores_plot, annot=True, cbar=False, **add_ax)
    
    axs[0].set_title('Arrival-time picker error ' + metric)
    
    ## Save Figure
    if new_figure and not return_fig:
        fig.savefig(options['DIR_FIGURES'] + 'metrics_picker_'+metric+'.pdf')
 
    return metric_plot
 
def compute_error_wave_picker_one_report(est_picker, data_picker, dataset_name, window, max_deviation, balanced_classes):
    
    """
    Return one error metrics report for a given dataset
    """

    report = {
        'dataset': dataset_name,
        'window': window,
        'max_deviation': max_deviation,
        'R2': est_picker.oob_score_,
        'MSE': np.mean(data_picker['error_scaled'].values**2),
        'RMSE': np.sqrt(np.mean(data_picker['error_scaled'].values**2)),
        'MAE': np.mean(abs(data_picker['error_scaled'].values)),
        'balanced_classes': balanced_classes,
    }
    
    return report
 
def compute_error_sensitivity_wave_picker(tec_data, tec_data_param, options, l_window, l_max_deviation, nb_trees=1000):
    
    """
    Compute various error metrics for the wave picker using different window and overlap sizes
    """
    
    reports = pd.DataFrame()
    bandpass = [] # dummy value that is not used when you load features
    for max_deviation in l_max_deviation:
        for window in l_window:
            for balanced_classes in ['True', 'False']:
                options['load']['features-picker'] = options['DIR_DATA'] + 'features_picker_w'+str(window)+'_d'+str(max_deviation)+'.csv'
                est_picker, data_picker = \
                    train_wave_picker.create_arrival_picker(tec_data, tec_data_param, bandpass, options, 
                                                           sampling=30., window=window, nb_picks=5, max_deviation=max_deviation, 
                                                           min_distance_time=30., split=0.9, seed=1, nb_trees=nb_trees, 
                                                           type_ML = 'forest', balanced_classes=balanced_classes, save_est=False,
                                                           plot_error_distribution=False)
 
                data_picker['error_scaled'] = data_picker['error'] / window
                data_picker_test  = data_picker.loc[data_picker['type'] == 'test', :]
                data_picker_train = data_picker.loc[data_picker['type'] == 'train', :]
                
                report = compute_error_wave_picker_one_report(est_picker, data_picker, 'all', window, max_deviation, balanced_classes)
                reports = reports.append( [report] )
                report = compute_error_wave_picker_one_report(est_picker, data_picker_test, 'test', window, max_deviation, balanced_classes)
                reports = reports.append( [report] )
                report = compute_error_wave_picker_one_report(est_picker, data_picker_train, 'train', window, max_deviation, balanced_classes)
                reports = reports.append( [report] )
 
    reports.to_csv(options['DIR_FIGURES'] + 'reports_picker.csv', header=True, index=False)
 
    return reports
 
def setup_iono_map(ax, lon_source, lat_source, offset_source_lat, offset_source_lon, no_plot, labels_bottom=False):

    """
    Setup a basemap for ionospheric point plotting
    """

    dimension = {
        'lon_0': lon_source,
        'lat_0': lat_source,
        'llcrnrlon': lon_source - offset_source_lon, 
        'llcrnrlat': lat_source - offset_source_lat, 
        'urcrnrlon': lon_source + offset_source_lon, 
        'urcrnrlat': lat_source + offset_source_lat,
    }
    
    m = Basemap(projection='mill', resolution='h', ax=ax, **dimension)
           
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='white', lake_color='cyan')
    m.drawcoastlines()
    m.drawcountries()

    if no_plot == 0:
        lat_ticks = np.arange(-90., 90., 2.)
        lon_ticks = np.arange(-180., 180., 4.)
        lat_ticks = lat_ticks[(lat_ticks >= dimension['llcrnrlat']) & (lat_ticks <= dimension['urcrnrlat'])]
        lon_ticks = lon_ticks[(lon_ticks >= dimension['llcrnrlon']) & (lon_ticks <= dimension['urcrnrlon'])]
        lats = m.drawparallels(lat_ticks, labels=[True,False,False,False], linewidth=0.)
        labels=[False,False,True,False]
        if labels_bottom:
            labels=[False,False,False,True]
        lons = m.drawmeridians(lon_ticks, labels=labels, linewidth=0., rotation=45)
    
    return m

def get_list_events():

    """
    Return location and time of each event in the dataset
    """

    list_events = {
            'Tohoku': (38.3, 142.37, '2011-03-11T05:46:23'),
            'Sanriku': (38.44, 142.84, '2011-03-09T02:45:20'),
            'Kaikoura': (-42.757, 173.077, '2016-11-13T11:02:56'),
            'Kii': (33.1, 136.6, '2004-09-05T10:07:07'),
            'Sumatra_1': (2.35, 92.86, '2012-04-11T08:38:37'),
            'Sumatra_2': (0.90, 92.31, '2012-04-11T10:43:09'),
            'Macquarie': (-49.91, 161.25, '2004-12-23T14:59:03'),
            'Fiordland': (-45.75, 166.58, '2009-07-15T09:22:29'),
            'Tokachi': (41.78, 143.90, '2003-09-25T19:50:06'),
            'Chuetsu': (37.54, 138.45, '2007-07-16T01:12:22'),
            'Illapel': (-31.57, -71.61, '2015-09-16T22:54:33'),
            'Iquique': (-19.61, -70.77, '2014-04-01T23:46:47'),
        }
    return list_events

def plot_stations_one_event(m, associations_event_max_time_in, associations_event_end_in, 
                            event, event_time, cmap, plot_only_grey_stations=False,
                            vmin=7., vmax=11.):
    
    ## Copy to local instance for modification
    associations_event_max_time = associations_event_max_time_in.copy()
    associations_event_end = associations_event_end_in.copy()

    ## Normalize colors
    associations_event_max_time['time-corrected'] /= 60.
    associations_event_max_time['time-corrected'] -= event_time
    associations_event_end['time-corrected'] /= 60.
    associations_event_end['time-corrected'] -= event_time
    #if not plot_only_grey_stations:
    #    vmin = associations_event_end['time-corrected'].quantile(q=0.025) - 0.5
    #    vmax = associations_event_end['time-corrected'].quantile(q=0.9) + 0.5
        #bounds = np.linspace(vmin, vmax, int((vmax-vmin)//0.1))
        #norm   = mpl.colors.BoundaryNorm(bounds, cmap.N)

    print('loop over stations for plotting')
        
    
    if plot_only_grey_stations:
        args_color = {'c': 'grey', 'alpha': 0.5}
        lon = associations_event_end.lon.values
        lat = associations_event_end.lat.values
    else:
        #vmin = 7.#associations_event_end['time-corrected'].quantile(q=0.1) - 0.5
        #vmax = 11.#associations_event_end['time-corrected'].quantile(q=0.5) + 0.5   
        args_color = {
            'c': associations_event_max_time['time-corrected'].values, 
            'cmap': cmap, 
            'vmin': vmin,
            'vmax': vmax,
            #'norm': norm,
        }
        lon = associations_event_max_time.lon.values
        lat = associations_event_max_time.lat.values
    
    #print('c:', args_color['c'])
    #print('lat:', lat)
    
    sc = m.scatter(lon, lat, marker='o', s=15, zorder=10, latlon=True, **args_color)
        
    return sc

def plot_time_dependence_image_iono(associations_time_steps, 
                                    list_time_proportion, options, 
                                    use_time_since_quake_instead=True, 
                                    window=720., overlap=0.7, 
                                    offset_source_lat=8., offset_source_lon=8.,
                                    offset=500., nb_pts_picker_max=10, quantile_threshold=0.8,
                                    vmin=7., vmax=11.):

    """
    Plot ionospheric images of RF prediction at various times
    """
    
    ## Initialize color map for arrival times
    cmap = sns.color_palette("rocket", as_cmap=True)
    
    ## Get list of all events in dataset
    list_events = get_list_events()
    
    ## Loop over each event since each event will be plotted in a different figure
    associations_grouped = associations_time_steps.groupby('event')
    for event, associations_event in associations_grouped:
    
        associations_event = associations_event.groupby(['time_association', 'satellite', 'station', 'arrival_class']).first().reset_index()
        associations_event_end = associations_event.loc[associations_event.time_association \
                                                        == associations_event.time_association.max()]
        
        print('event', event, associations_event.shape, associations_event_end.shape)
        
        ## Event characteristics
        event_name = '_'.join(event.split('_')[:-1])
        lat_source, lon_source, time = list_events[event_name]
        event_time = UTCDateTime(time)
        event_time -= UTCDateTime(event_time.year, event_time.month, event_time.day)
        event_time /= 60.
        
        #if not event_name in ['Tohoku', 'Sanriku']:
        #    continue
        
        ## Find all detections
        #first_detections_all = \
        #        get_first_detections(detections_event, offset, nb_pts_picker=nb_pts_picker_max, quantile_threshold=quantile_threshold)
        
        #print('Collected first detections')
        
        ## Setup figure
        fig, axs = plt.subplots(nrows=1, ncols=len(list_time_proportion), figsize=(10,5))
        
        ## Find all detections for a given max time
        #first_detections = []
        #ms = []
        for iax, (time_proportion, ax) in enumerate(zip(list_time_proportion, axs)):
        
            ## Collect detections up to current time
            #min_time = first_detections_all.time.min() + window
            #max_time = first_detections_all.time.max() + window #+ overlap * window
            #time_since_event = min_time + time_proportion * (max_time - min_time)
            if use_time_since_quake_instead:
                time_since_event = (event_time + time_proportion)*60.
            else:
                time_since_event = (associations_event.time_association).quantile(q=time_proportion)
                
            associations_event_max_time = \
                associations_event.loc[associations_event.time_association <= time_since_event, :]
            associations_event_max_time = \
                associations_event_max_time.loc[associations_event_max_time.time_association \
                                                == associations_event_max_time.time_association.max()]
            
            print('time', time_since_event, associations_event_max_time.shape)
            
            ## Find arrival times for each detection
            #first_detections = get_first_detections(detections_event_max_time, offset, nb_pts_picker=nb_pts_picker_max, quantile_threshold=quantile_threshold) 

            ## Setup map
            m = setup_iono_map(ax, lon_source, lat_source, offset_source_lat, offset_source_lon, iax, labels_bottom=True)
            
            ## Plot source
            m.scatter(lon_source, lat_source, marker='*', s=200, c='yellow', edgecolors='black', zorder=5, latlon=True)
        
            ## Plot current time as title
            time_str = str(np.round((time_since_event/60.-event_time), 2))
            ax.set_title('Minute since event ' + time_str)
            
            ## Plot all stations as grey dots
            print('plot stations grey')
            _ = plot_stations_one_event(m, associations_event_max_time, associations_event_end, event, event_time, cmap, plot_only_grey_stations=True)
            #_ = plot_stations_one_event(m, tec_data, tec_data_param, first_detections_all, first_detections_all, event, event_time, cmap, plot_only_grey_stations=True)
            
            ## Plot all stations that have a detection
            print('plot stations')
            sc = plot_stations_one_event(m, associations_event_max_time, associations_event_end, event, event_time, cmap, 
                                        plot_only_grey_stations=False, vmin=vmin, vmax=vmax)
            #sc = plot_stations_one_event(m, tec_data, tec_data_param, first_detections, first_detections_all, event, event_time, cmap, plot_only_grey_stations=False)
            
            ## Add labels
            alphabet = string.ascii_lowercase
            offset = 0
            for iax, ax in enumerate(axs):
                ax.text(-0.1, 1.1, alphabet[iax+offset] + ')', ha='right', va='bottom', transform=ax.transAxes, 
                     bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=17, fontweight='bold')
          
        axins_ = inset_axes(axs[-1], width="4%", height="100%", loc='lower left', 
                            bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[-1].transAxes, borderpad=0)
        axins_.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        cbar = plt.colorbar(sc, cax=axins_, extend='both')
        cbar.ax.set_ylabel('Arrival time (min since event)', rotation=270, labelpad=12) 
        
        fig.subplots_adjust(wspace=0.1, right=0.85)
    
        fig.savefig(options['DIR_FIGURES'] + 'iono_map_event_time_'+event+'.svg')
        
    bp()

def read_fault_complete(event, main_dir, unknown='slip'):

    """
    Read fault complete inversion downloaded from IRIS .param files
    """

    filename = main_dir + 'surface_displacement/{event}_faults_complete.txt'
    filename_dict = {'event': event}
    file = filename.format(**filename_dict)
    all_faults = pd.read_csv(file, header=None, comment='%', delim_whitespace=True)
    all_faults.columns = ['Lat', 'Lon', 'X', 'y', 'Z', 'slip', 'rake', 'trup', 'rise', 'sf_moment']
    
    return all_faults

def read_fault_params(event, main_dir, unknown='slip'):

    """
    Read fault parameters downloaded from IRIS .param files
    """

    filename = main_dir + 'surface_displacement/{event}_faults.txt'
    filename_dict = {'event': event}
    with open(filename.format(**filename_dict)) as f:
        content = f.readlines()

    line_begin = '#Lat. Lon. depth slip rake strike dip t_rup t_ris t_fal mo'
    line_end = '#Fault_segment'

    ## Read line by line
    all_faults = pd.DataFrame()
    read_fault = False
    cols = ['Lat', 'Lon', 'depth', 'slip', 'rake', 'strike', 'dip', 't_rup', 't_ris', 't_fal', 'mo']
    new_fault = pd.DataFrame()
    for line in content:
        
        if line_end in line:
            if new_fault.size > 0:
                all_faults = all_faults.append( new_fault )
            read_fault = False
        
        if read_fault:
            loc_df = pd.DataFrame(data=[line.split()], columns=cols, dtype=float)
            new_fault = new_fault.append( loc_df )
    
        if line_begin in line:
            new_fault = pd.DataFrame(columns = cols)
            read_fault = True
    
    all_faults.reset_index(drop=True, inplace=True)
    
    return all_faults
    #sns.scatterplot(data=all_faults, x="Lon", y="Lat", hue=unknown, marker=('s', 0, 45))
    #bp()

def plot_error_LOS_RF(tec_data, first_detections, options):

    """
    Show arrival time errors vs LOS to investigate the influence of satellite geometry
    """

    if not 'LOS' in first_detections.columns:
        first_detections['LOS'] = -200
        first_detections_grouped = first_detections.groupby(['event', 'satellite', 'station'])
        for group, detection in first_detections_grouped:
        
            event, satellite, station = group
            
            one_detection = detection.iloc[0]
            tec_data_station = tec_data.loc[(tec_data['station'] == station) 
                                            & (tec_data['satellite'] == satellite) 
                                            & (tec_data['event'] == event), :]
            
            predicted_time = one_detection['time-corrected']
            idxmax = abs(tec_data_station.time_s-predicted_time).idxmin()
            LOS = tec_data_station['LOS'].loc[tec_data_station.index == idxmax].iloc[0]
            first_detections.loc[first_detections.index==one_detection.name, 'LOS'] = LOS
        
        first_detections.to_csv('first_detections_all_events_LOS.csv', sep=',', index=False, header=True)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(data=first_detections, x="error", y="LOS", thresh=0.05, fill=True, ax=ax); 
    ax.set_xlim([-100., 100.])
    ax.set_ylim([0., 90.])
    ax.set_xlabel('Arrival-time error (s)')
    ax.set_ylabel('Line Of Sight (LOS, degrees)')
    fig.savefig('/staff/quentin/Documents/Projects/ML_TEC/figures/error_LOS.pdf')
    
    bp()
 
def correct_position_points_iono(x, tec_data_hion, hion_dict):

    """
    Correct latitude and longitude of ionospheric points depending on Hion tec_data database
    """
    
    if x['satellite'] not in hion_dict:
        print('Information about which Hion to use is not provided for satellite ', x['satellite'])
        return x

    try:
        selected_station = "{:04d}".format(x['station'])
    except:
        pass
    tec_data_station = tec_data_hion.loc[(tec_data_hion['station'] == selected_station) 
                                    & (tec_data_hion['satellite'] == x['satellite']) 
                                    & (tec_data_hion['event'] == x['event'])
                                    & (abs(tec_data_hion['time_s'] - x['time']) < 0.5)
                                    & (tec_data_hion['Hion'] == hion_dict[x['satellite']]), :]
    
    #tec_data_station = tec_data_hion.loc[(tec_data_hion['station'] == selected_station) & (tec_data_hion['satellite'] == x['satellite']) & (tec_data_hion['event'] == x['event'])& (tec_data_hion['time_s'] == x['time'])& (tec_data_hion['Hion'] == hion_dict[x['satellite']]), :]
    
    if tec_data_station.size == 0:
        print('Can not correct position for ', x['event'], x['satellite'], x['station'])
        return x
    
    x['lat'] = tec_data_station.iloc[0].lat
    x['lon'] = tec_data_station.iloc[0].lon
    
    return x
 
def plot_image_iono(tec_data, tec_data_param, first_detections,
                    options, offset_source_lat=8., offset_source_lon=8.,
                    associations=pd.DataFrame(), tec_data_hion=pd.DataFrame(),
                    hion_dict={'G26': 180., 'G05': 180., 'G27': 180.},
                    # add_new_waveform={'satellite': 'G27', 'station': '0027'},
                    add_new_waveform={'satellite': 'G27', 'station': '0167'},
                    #add_new_waveform_class={'satellite': 'G05', 'station': '0033'},
                    add_new_waveform_class={'satellite': 'G05', 'station': '0155'},
                    add_fault=True, add_inset_fault=True, unknown='slip', rotation=24.5, size=20.,
                    vmin=9., vmax=15., first_label='f', remove_first_arrival=1000., ext_name='',
                    list_events=[]):

    """
    Plot ionospheric images of RF prediction vs true arrivals
    """
    
    ## Recast new waveforms
    if add_new_waveform and associations.station.dtypes=='int64':
        try:
            add_new_waveform['station'] = int(add_new_waveform['station'])
        except:
            pass
    
    if add_new_waveform_class and associations.station.dtypes=='int64':
        try:
            add_new_waveform_class['station'] = int(add_new_waveform_class['station'])
        except:
            pass
    
    ## If an association dataframe is provided, we select only the first time window of each associated detection
    if associations.size > 0:
        associations = associations.groupby(['satellite', 'station', 'arrival_class']).first().reset_index()
        
        ## If addition Hion dependent positions are provided for the iono points we store them
        if tec_data_hion.size > 0:
            associations = associations.apply(correct_position_points_iono, args=[tec_data_hion, hion_dict], axis=1)
        
        ## We save the largest association class in a separate dataframe for plotting
        id_best_assoc = associations.groupby('association_no')['station'].count().idxmax()
        associations_best = associations.loc[associations.association_no==id_best_assoc]
    
    if not list_events:
        list_events = get_list_events()
    
    print('AA')
    
    first_detections_grouped = first_detections.groupby('event')
    for event, first_detections_event in first_detections_grouped:
    
        ## Event characteristics
        if len(event.split('_')) > 1:
            event_name = '_'.join(event.split('_')[:-1])
        else:
            event_name = event
        print(event_name)
        lat_source, lon_source, time = list_events[event_name]
        event_time = UTCDateTime(time)
        event_time -= UTCDateTime(event_time.year, event_time.month, event_time.day)
        event_time /= 60.
        
        print('BB')
        
        #if not event_name in ['Tohoku', 'Sanriku']:
        #    continue
        
        ## Setup figure
        nb_figures = 2
        if associations.size > 0:
            nb_figures += 1
        fig, axs = plt.subplots(nrows=1, ncols=nb_figures, figsize=(10,5))
        
        ms = []
        for iax, ax in enumerate(axs):
        
            print('CC',iax)
            
            ## Setup map
            m = setup_iono_map(ax, lon_source, lat_source, offset_source_lat, offset_source_lon, iax)
            
            ## Plot source
            m.scatter(lon_source, lat_source, marker='*', s=200, c='yellow', edgecolors='black', zorder=5, latlon=True)
        
            ms.append( m )
        
        axs[0].set_title('True\narrivals')
        axs[1].set_title(event_name + '\nRF predictions')
        axs[2].set_title('Associated arrivals')
        
        print('DD')
        
        ## Plot ionospheric points
        first_detections_event_grouped = first_detections_event.groupby(['satellite', 'station'])
        
        ## Find range true arrivals
        tec_data_param_event = tec_data_param.loc[(tec_data_param['event'] == event), :]
        
        print('EE')
        
        ## Normalize colors
        first_detections_event['time-corrected'] /= 60.
        first_detections_event['time-corrected'] -= event_time
        #vmin = first_detections_event['time-corrected'].quantile(q=0.2)
        #vmax = first_detections_event['time-corrected'].quantile(q=0.8)
        if tec_data_param_event.size > 0:
            vmin = tec_data_param_event['arrival-time'].min()/60. - event_time - 0.5
            vmax = tec_data_param_event['arrival-time'].max()/60. - event_time + 0.5
        cmap = sns.color_palette("rocket", as_cmap=True)
        bounds = np.linspace(vmin, vmax, min(256, int((vmax-vmin)//0.1)))
        norm   = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        print('FF')

        if associations.size == 0:
            for (satellite, station), first_detections_station in first_detections_event_grouped:
            
                tec_data_param_station = tec_data_param_event.loc[(tec_data_param_event['station'] == station) 
                                        & (tec_data_param_event['satellite'] == satellite), :]
                                        
                if tec_data_param_station.size == 0:
                    continue
                    
                tec_data_param_station = tec_data_param_station.iloc[0]
                                        
                tec_data_station = tec_data.loc[(tec_data['station'] == station) 
                                        & (tec_data['satellite'] == satellite) 
                                        & (tec_data['event'] == event), :]
                
                arrival_time = tec_data_param_station['arrival-time']
                predicted_time = first_detections_station.iloc[0]['time-corrected']
                tmax = arrival_time + tec_data_param_station['t-ampmax-TEC']
                idxmax = abs(tec_data_station.time_s-tmax).idxmin()
                lat = tec_data_station['lat'].loc[tec_data_station.index == idxmax].iloc[0]
                lon = tec_data_station['lon'].loc[tec_data_station.index == idxmax].iloc[0]
                
                sc = ms[0].scatter(lon, lat, marker='o', s=50, c=arrival_time/60.-event_time, cmap=cmap, norm=norm, zorder=10, latlon=True)
                sc = ms[1].scatter(lon, lat, marker='o', s=50, c=predicted_time, cmap=cmap, norm=norm, zorder=10, latlon=True)
        
        else:
        
            associations_t = associations_best.loc[associations_best['true-arrival-time'] > -1]
            sc = ms[0].scatter(associations_t.lon.values, associations_t.lat.values, marker='o', s=20, 
                               c=associations_t['true-arrival-time'].values/60.-event_time, cmap=cmap, 
                               norm=norm, zorder=10, latlon=True)
                      
            if associations_t.shape[0] == 0:
                axs[0].set_title('')
                      
            associations_p = associations.loc[associations['time-corrected'] > -1]
            associations_p['keep'] = associations_p['time-corrected'] >= associations_p['time-begin-waveform']+remove_first_arrival
            associations_p = associations_p.loc[associations_p.keep]
            vmin_classes, vmax_classes = associations_p.association_no.values.min(), associations_p.association_no.values.max()
            sc = ms[1].scatter(associations_p.lon.values, associations_p.lat.values, marker='o', s=20, 
                               c=associations_p['time-corrected'].values/60.-event_time, cmap=cmap, 
                               norm=norm, zorder=10, latlon=True)
            
            _ = ms[1].scatter(associations_best.lon.values, associations_best.lat.values, marker='o', s=20, 
                               c=associations_best['time-corrected'].values/60.-event_time, cmap=cmap, 
                               norm=norm, zorder=10, latlon=True)
            
            ## Plot close up on fault
            if add_inset_fault:
            
                print('JJ')
            
                axins_inset_faults = inset_axes(axs[0], width="40%", height="40%", loc='lower left', 
                                bbox_to_anchor=(0., 0., 1, 1.), 
                                bbox_transform=axs[0].transAxes, borderpad=0)
            
                ## Setup map
                m_inset_fault = setup_iono_map(axins_inset_faults, lon_source, lat_source, 1., 1., 0)
                axins_inset_faults.set_xlabel('')
                axins_inset_faults.set_ylabel('')
                axins_inset_faults.set_xticklabels([])
                axins_inset_faults.set_yticklabels([])
                
                ## Plot source
                m_inset_fault.scatter(lon_source, lat_source, marker='*', s=200, c='yellow', edgecolors='black', zorder=5, latlon=True)
                
            ## Plot true slip at the surface
            if add_fault:
                print('KK')
                all_faults = read_fault_complete(event_name, options['DIR_DATA'] + '../', unknown='slip')
                
                cmap_fault = sns.color_palette("viridis", as_cmap=True)
                cmap_fault_ = cmap_fault(np.arange(cmap_fault.N))
                cmap_fault_[:,-1] = np.linspace(0, 1, cmap_fault.N)
                cmap_fault = ListedColormap(cmap_fault_)
                cmap_fault.set_under(alpha=0.)
                
                norm_fault = plt.Normalize(vmin=all_faults[unknown].quantile(0.4), vmax=all_faults[unknown].quantile(1.))
                sc_faults = ms[0].scatter(all_faults.Lon.values, all_faults.Lat.values, c=all_faults[unknown].values, 
                              marker=(4, 0, rotation), s=size, norm=norm_fault, cmap=cmap_fault, latlon=True, zorder=9)

                print('LL')

                axins_ = inset_axes(axs[0], width="100%", height="4%", loc='lower left', 
                            bbox_to_anchor=(0., -0.1, 1, 1.), bbox_transform=axs[0].transAxes, borderpad=0)
                axins_.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False, labelrotation=90.)
                cbar = plt.colorbar(sc_faults, cax=axins_, extend='both', orientation="horizontal")  
                cbar.ax.xaxis.set_ticks_position('bottom') 
                cbar.ax.xaxis.set_label_position("bottom")
                cbar.ax.xaxis.tick_bottom()
                cbar.ax.set_xlabel('Slip (m)', rotation=0., labelpad=0) 

                ## Plot close up on fault
                if add_inset_fault:
                    m_inset_fault.scatter(all_faults.Lon.values, all_faults.Lat.values, c=all_faults[unknown].values, 
                              marker=(4, 0, rotation), s=size, norm=norm_fault, cmap=cmap_fault, latlon=True, zorder=9)
                    
            print('LLMM')
                    
            ## Add inset for new waveform
            if add_new_waveform:
                
                print('MM', add_new_waveform)
                
                id_correct_assoc = associations.groupby('association_no')['station'].count().idxmax()
                correct_assoc = associations.loc[(associations.association_no == id_correct_assoc) & (associations['true-arrival-time'] == -1)]
                correct_assoc = correct_assoc.groupby(['satellite', 'station', 'arrival_class']).first().reset_index()
                
                print('NN', add_new_waveform, correct_assoc)
                
                #selected_detections = correct_assoc.loc[(correct_assoc.lon <= lon_source)]
                #selected_detections = selected_detections[selected_detections.lon == selected_detections.lon.max()]
                selected_detections = correct_assoc.loc[(correct_assoc.satellite==add_new_waveform['satellite']) 
                                                       & (correct_assoc.station==add_new_waveform['station'])]
                if selected_detections.shape[0] == 0:
                    try:
                        selected_detections = correct_assoc.loc[(correct_assoc.satellite==add_new_waveform['satellite']) 
                                                       & (correct_assoc.station==int(add_new_waveform['station']))]
                    except:
                        pass
                
                print('OO', selected_detections)
                
                selected_detection = selected_detections.iloc[0]
                selected_event, selected_satellite, selected_station = \
                        selected_detection.event, selected_detection.satellite, selected_detection.station
                try:
                    selected_station = "{:04d}".format(selected_station)
                except:
                    pass
                
                waveform = tec_data.loc[(tec_data['station'] == selected_station) 
                                        & (tec_data['satellite'] == selected_satellite) 
                                        & (tec_data['event'] == selected_event)
                                        & (tec_data['time_s'] >= selected_detection['time-corrected']-150.)
                                        & (tec_data['time_s'] <= selected_detection['time-corrected']+800.), :]
                                        
                print('PP', waveform)
                print('uuu', selected_detection)
                                        
                axins = inset_axes(axs[1], width="65%", height="28%", loc='lower left', 
                                bbox_to_anchor=(0., 0., 1, 1.), 
                                bbox_transform=axs[1].transAxes, borderpad=0)
                axins.plot(waveform.time_s, waveform.vTEC, linewidth=2.)
                axins.axvline(selected_detection['time-corrected'], color='tab:red', alpha=0.5)
                axins.set_xlim([selected_detection['time-corrected']-150., selected_detection['time-corrected']+800.])
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                
                print(selected_detections['time-corrected'].values/60.-event_time, waveform.iloc[0])
                ms[1].scatter(selected_detections.lon.values, selected_detections.lat.values, marker='o', s=20, 
                               c=selected_detections['time-corrected'].values/60.-event_time, cmap=cmap, 
                               norm=norm, zorder=10, latlon=True, edgecolors='tab:blue')
                               
                axs[1].text(0.66, 0.01, selected_satellite + '\n' + selected_station, ha='left', va='bottom', transform=axs[1].transAxes)
                
                scalebar = AnchoredSizeBar(axins.transData,
                           120., '2 min', 'lower left', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1)

                axins.add_artist(scalebar)
                
                axs[1].annotate("",
                    xy=ms[1](selected_detection.lon, selected_detection.lat), xycoords=axs[1].transData,
                    xytext=(0.25, 0.28), textcoords=axs[1].transAxes,
                    arrowprops=dict(arrowstyle="<-", color="tab:blue",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle='angle,angleA=90,angleB=10,rad=5',
                                    ),
                    )
            
            
            print('Ww')
            
            sc_classes = ms[2].scatter(associations_p.lon.values, associations_p.lat.values, marker='o', s=20, 
                               c=associations_p.association_no.values, cmap='tab20b', zorder=10, latlon=True, 
                               vmin=vmin_classes, vmax=vmax_classes)
            _ = ms[2].scatter(associations_best.lon.values, associations_best.lat.values, marker='o', s=20, 
                               c=associations_best.association_no.values, cmap='tab20b', zorder=10, latlon=True,
                               vmin=vmin_classes, vmax=vmax_classes)
            axins_ = inset_axes(axs[2], width="100%", height="4%", loc='lower left', 
                            bbox_to_anchor=(0., -0.1, 1, 1.), bbox_transform=axs[2].transAxes, borderpad=0)
            axins_.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            cbar = plt.colorbar(sc_classes, cax=axins_, extend='both', orientation="horizontal")  
            cbar.ax.xaxis.set_ticks_position('bottom') 
            cbar.ax.xaxis.set_label_position("bottom")
            cbar.ax.xaxis.tick_bottom()
            cbar.ax.set_xlabel('Class number', rotation=0., labelpad=0) 
            
            print('XX')
            
            ## Add inset for classification waveform
            if add_new_waveform_class:
                #id_correct_assoc = associations.groupby('association_no')['station'].count().idxmax()
                #correct_assoc = associations.loc[(associations['true-arrival-time'] == -1)]
                correct_assoc = associations.groupby(['satellite', 'station', 'arrival_class']).first().reset_index()
                #selected_detections = correct_assoc.loc[(correct_assoc.lon <= lon_source)]
                #selected_detections = selected_detections[selected_detections.lon == selected_detections.lon.max()]
                selected_detections = correct_assoc.loc[(correct_assoc.satellite==add_new_waveform_class['satellite']) 
                                                       & (correct_assoc.station==add_new_waveform_class['station'])]
                
                if selected_detections.shape[0] == 0:
                    try:
                        selected_detections = correct_assoc.loc[(correct_assoc.satellite==add_new_waveform_class['satellite']) 
                                                       & (correct_assoc.station==int(add_new_waveform_class['station']))]
                    except:
                        pass
                
                print(selected_detections)
                
                #print('YY')
                
                axins = inset_axes(axs[2], width="65%", height="28%", loc='lower left', 
                                bbox_to_anchor=(0., 0., 1, 1.), 
                                bbox_transform=axs[2].transAxes, borderpad=0)
                
                ## Collect wavefrom from first detection
                selected_event, selected_satellite, selected_station = \
                            selected_detections.iloc[0].event, selected_detections.iloc[0].satellite, selected_detections.iloc[0].station
                
                try:
                    selected_station = "{:04d}".format(selected_station)
                except:
                    pass
                
                waveform = tec_data.loc[(tec_data['station'] == selected_station) 
                                            & (tec_data['satellite'] == selected_satellite) 
                                            & (tec_data['event'] == selected_event), :]
                
                #tec_data.loc[(tec_data['station'] == selected_station) & (tec_data['satellite'] == selected_satellite) & (tec_data['event'] == selected_event), :]
                print('----', waveform, selected_event, selected_satellite, selected_station)
                waveform = waveform.loc[(waveform['time_s'] <= waveform['time_s'].max()-700.)
                                            & (waveform['time_s'] >= waveform['time_s'].min()+600.), :]
                                            
                for iselect, selected_detection in selected_detections.iterrows():
                    selected_event, selected_satellite, selected_station = \
                            selected_detection.event, selected_detection.satellite, selected_detection.station
                    try:
                        selected_station = "{:04d}".format(selected_station)
                    except:
                        pass
                    
                    #print(selected_detection['time-corrected'])
                    axins.axvline(selected_detection['time-corrected'], color=sc_classes.to_rgba(selected_detection['association_no']), alpha=0.95)
                    
                print('ZZ', selected_event, selected_satellite, selected_station, tec_data)
                
                axins.plot(waveform.time_s, waveform.vTEC, linewidth=2.)
                axins.set_xlim([waveform.time_s.min(), waveform.time_s.max()])
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                    
                #print(selected_detections['time-corrected'].values/60.-event_time, waveform.iloc[0])
                ms[2].scatter(selected_detections.lon.values, selected_detections.lat.values, marker='o', s=20, 
                               c=selected_detections['association_no'].values, cmap='tab20b', 
                               vmin=vmin_classes, vmax=vmax_classes, zorder=10, latlon=True, edgecolors='tab:blue')
                print(selected_detections)
                               
                axs[2].text(0.66, 0.01, selected_satellite + '\n' + selected_station, ha='left', va='bottom', transform=axs[2].transAxes)
                
                scalebar = AnchoredSizeBar(axins.transData,
                           600., '10 min', 'lower right', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0.1)

                axins.add_artist(scalebar)
                
                axs[2].annotate("",
                    xy=ms[2](selected_detections.iloc[0].lon, selected_detections.iloc[0].lat), xycoords=axs[2].transData,
                    xytext=(0.25, 0.28), textcoords=axs[2].transAxes,
                    arrowprops=dict(arrowstyle="<-", color="tab:blue",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle='angle,angleA=90,angleB=10,rad=5',
                                    ),
                    )
        
                print('ZZAA')
        
        alphabet = string.ascii_lowercase
        l_alphabet = [str for str in alphabet]
        offset = alphabet.index(first_label)
        for iax, ax in enumerate(axs):
            ax.text(-0.1, 1.1, alphabet[iax+offset] + ')', ha='right', va='bottom', transform=ax.transAxes, 
                 bbox=dict(facecolor='w', edgecolor='w', pad=0.1), fontsize=17, fontweight='bold')
        
        axins_ = inset_axes(axs[1], width="100%", height="4%", loc='lower left', 
                            bbox_to_anchor=(0., -0.1, 1, 1.), bbox_transform=axs[1].transAxes, borderpad=0)
        axins_.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False, labelrotation=90.)
        cbar = plt.colorbar(sc, cax=axins_, extend='both', orientation="horizontal")  
        cbar.ax.xaxis.set_ticks_position('bottom') 
        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.xaxis.tick_bottom()
        #cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        cbar.ax.set_xlabel('Minutes since event', rotation=0., labelpad=0) 
        
        fig.subplots_adjust(wspace=0.1, right=0.85, top=0.95)
    
        fig.savefig(options['DIR_FIGURES'] + 'iono_map_event_'+event+ext_name+'.svg')
        
def update_scatter_association(associations_time_steps, times, axs, xlim, ylim, i):

    """
    Update scatter plot of the spatial distribution of associated arrivals at new time
    """
    
    data = associations_time_steps.loc[associations_time_steps.time_association == times[i]]
    sns.scatterplot(data=associations_time_steps, x="lon", y="lat", ax=axs[0], alpha=0.5, color='grey', legend=False)
    sns.scatterplot(data=data, x="lon", y="lat", hue="association_no", ax=axs[0], palette='Paired', legend=False)
    sns.scatterplot(data=associations_time_steps, x="lon", y="lat", ax=axs[1], alpha=0.5, color='grey', legend=False)
    sns.scatterplot(data=data, x="lon", y="lat", hue="time-corrected", ax=axs[1], palette='rocket', legend=False)
    axs[0].set_title('Time ' + str(times[i]))
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
 
def create_animation_association(associations_time_steps_all, options, subsample=4, interval=500):
    
    """
    Create a GIF animation for the spatial distribution of associated arrivals at different times
    """
    
    ## Create one figure for each event
    associations_time_steps_grouped = associations_time_steps_all.groupby(['event'])
    for event, associations_time_steps_event in associations_time_steps_grouped:
    
        associations_time_steps = associations_time_steps_event.groupby(['time_association', 'satellite', 'station', 'arrival_class']).first().reset_index()
        times = associations_time_steps.time_association.unique()[::subsample]
        
        xlim = [associations_time_steps.lon.min()-1., associations_time_steps.lon.max()+1.]
        ylim = [associations_time_steps.lat.min()-1., associations_time_steps.lat.max()+1.]
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        update_scatter_association_partial = partial(update_scatter_association, associations_time_steps, times, axs, xlim, ylim)
        
        num_frames = len(times)        
        anim = animation.FuncAnimation(fig, update_scatter_association_partial, frames=num_frames, interval=interval)
        anim.save(options['DIR_FIGURES'] + 'association_'+event+'.gif')
 
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
 
def resize_seaborn_plots(fig, l_seaborn_fig, subplot_spec=111):
    
    """
    Resize jointplots to fit them in subplot figure
    """
    
    ## Input size
    w_offset = 0.15
    w_offset_bw_fig = 0.05
    h_offset = 0.1
    h_max = 0.4
    h_distrib = 1./8.
    w_distrib = 1./8.
    
    ## Build sizes
    w_kde = 1. - 2*w_offset - w_offset_bw_fig*(len(l_seaborn_fig) - 1) #- w_distrib*w_kde*len(l_seaborn_fig) - (len(l_seaborn_fig)-1)*w_kde
    w_kde /= len(l_seaborn_fig) * (1. + w_distrib)
    h_kde = h_max - h_offset
    sizes = [
        (w_offset, h_offset, w_kde, h_kde),
        (w_offset, h_offset+h_kde, w_kde, h_distrib*h_kde),
        (w_offset+w_kde, h_offset, w_distrib*w_kde, h_kde),
    ]
    
    for iJ, J in enumerate(l_seaborn_fig):
        for is_, (s, ax) in enumerate(zip(sizes, J.fig.axes)):
            
            ax.figure = fig
            
            #fig._axstack.add(fig._make_key(A), A)
            fig.axes.append(ax)
            # twice, I don't know why...
            fig.add_axes(ax)
            
            s_local = list(s)
            s_local[0] += iJ*(w_kde+w_distrib*w_kde+w_offset_bw_fig)
            ax.set_position(s_local)
    
def plot_performance_picker(data_picker, reports, options, metric='RMSE', 
                            xlim=[-400., 400.], ylim=[-300., 300.],
                            name_ext='', fontsize=15.):

    """
    Figure showing error vs time shift, R2 vs window/deviation
    """
    
    ## Create figure grid
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2, 2)
    
    alphabet = string.ascii_lowercase
    for imetric, metric in enumerate(['RMSE', 'R2']):
        axs = [fig.add_subplot(gs[imetric])]
        #fig = plt.figure(figsize=(13,8)); gs = gridspec.GridSpec(2, 2); axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        gmetric = plot_error_wave_picker(reports, options, metric=metric, return_fig=False, axs=axs)
        axs[0].text(-0.1, 1.05, alphabet[imetric] + ')', ha='right', va='bottom', 
                transform=axs[0].transAxes, bbox=dict(facecolor='w', edgecolor='w', pad=0.1), 
                fontsize=15., fontweight='bold')
        axs[0].set_xticklabels(axs[0].get_xmajorticklabels(), fontsize = fontsize-2.)
        axs[0].set_yticklabels(axs[0].get_ymajorticklabels(), fontsize = fontsize-2.)
        if imetric == 0:
            axs[0].set_ylabel('Window (s)', fontsize = fontsize)
            axs[0].set_xlabel('Deviation', fontsize = fontsize)
        else:
            axs[0].set_ylabel('')
            axs[0].set_xlabel('')
        
    gtest, gtrain = train_wave_picker.plot_error_distribution(data_picker, options, xlim=xlim, ylim=ylim, name_ext=name_ext, fontsize=fontsize, create_figures=False)
                         
    ## Move complex jointplot around
    l_seaborn_fig = [gtest, gtrain]
    resize_seaborn_plots(fig, l_seaborn_fig)
    #mg0 = SeabornFig2Grid(gmetric, fig, gs[0])
    #mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])
    #mg2 = SeabornFig2Grid(gtest, fig, gs[2])
    #mg3 = SeabornFig2Grid(gtrain, fig, gs[3])
    
    fig.savefig(options['DIR_FIGURES'] + 'performance_picker_figure_paper.pdf')
    bp()

def read_tec_data_Hion(dir):

    """
    Read tec data Hion file from a given directory
    """
    
    dtype = {'epoch': int, 'UT': float, 'LOS': float, 'az': float, 'lat': float, 'lon': float, 'sTEC': float, 'vTEC': float, 
             'time_s': float, 'station': str, 'satellite': str, 'doy': int, 'year': int, 'event': str, 'sampling': float, 'Hion': float}
    tec_data_hion = pd.read_csv('/staff/quentin/Documents/Projects/ML_TEC/data_Hion/tec_data_hion.csv', header=[0], dtype=dtype)

    return tec_data_hion

def create_tec_data_for_Hion(dir):

    """
    Create TEC data file only for latitude and longitude computed with different Hion
    """
    
    file_tec_data       = 'tec_data.csv'
    files_to_exclude = [file_tec_data]
        
    tec_data = pd.DataFrame()
    print('Start reading data')
    for  subdir, dirs, files in os.walk(dir):
        
        for file in files:
        
            filepath = subdir + os.sep + file
            
            ## Skip csv tec_data files
            if file in files_to_exclude:
                continue
            
            if 'rtTEC' in file:
            
                tec_data_ = read_data.load_one_rTEC(filepath)
                Hion = float(file.split('_')[-1].replace('km', ''))
                tec_data_['Hion'] = Hion
                #if tec_data_.station.iloc[0] == '0001':
                #    print(file, Hion)
                tec_data = tec_data.append( tec_data_ )
                
    tec_data.reset_index(drop=True, inplace=True)
        
    ## Post processing to add default values if no wignal duration provided
    tec_data.drop_duplicates(inplace=True, subset=['event', 'satellite', 'station', 'time_s', 'Hion'], keep='last')
    tec_data.reset_index(drop=True, inplace=True)
    tec_data.to_csv(dir + 'tec_data_hion.csv', header=True, index=False)
    
def create_latex_table_features(options):

    """
    Create latex table to list input features
    """

    feature_name = compute_params_waveform.setup_name_parameters(options)
    
    template_table_attribute = """
    \\begin{{table}}
    \\caption{{List of attributes.}}
    \\begin{{tabular}}{{l c c}} 
    \\hline
    Short name & Type & Description \\\\
    \\hline\\hline
    {content}
    \\end{{tabular}}
    \\label{{tab:list_attributes}}
    \\end{{table}}
    """
    
    template = '{short_name} & {type} & {description} \\\\\n'
    types    = {'F': 'Spectro.', 'W': 'Timeseries', 'S': 'Spectrum'}
    content  = ''
    for key in feature_name:
        type  = types[key[0]]
        entry = {'short_name': key, 'type': type, 'description': feature_name[key]}
        content += template.format(**entry)
        
    table_attribute = template_table_attribute.format(content=content)
     
    file_name = options['DIR_FIGURES'] + 'table_atributes.tex'
    with open(file_name, 'w') as w_file:
        w_file.write(table_attribute)
    
