import pandas as pd
import re
import os
import sys
from os.path import join, basename, dirname, isfile, isdir, abspath
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.signal import argrelextrema
from scipy.signal import butter, lfilter, freqz, find_peaks_cwt
import scipy.signal as signal
from scipy.integrate import simps
import scipy.stats
from numpy import trapz
import seaborn as sns

import itertools
import operator

# area under the curve
import numpy as np
from scipy.integrate import simps
from numpy import trapz

def interpolate_df(df, gap):
    f = interpolate.interp1d(df['time[us]'], df['volt(fsr)[v]'])
    y = f(np.arange(df['time[us]'].min(), df['time[us]'].max(), gap))

    f_signal = interpolate.interp1d(df['time[us]'], df['signal'], kind='nearest')
    y_signal = f_signal(np.arange(df['time[us]'].min(), df['time[us]'].max(), gap))
    
    data_response_interp = pd.DataFrame({'time[us]':np.arange(df['time[us]'].min(), 
                                                             df['time[us]'].max(), gap),
                                        'volt(fsr)[v]':y,
                                        'signal':y_signal})
    return data_response_interp

def find_nearest(array,value):
    '''
    Return the nearest item from the array to the value
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def get_nearest_indices(df, indices):
    '''
    Return list of indices that match nearest indices in the df
    '''
    matched_indices = []
    for i in indices:
        nearest_point = find_nearest(df.index, i)
        matched_indices.append(nearest_point)
    return matched_indices


def remove_partial_peaks(df, threshold):
    cut_index_thr = np.diff(df[df['volt(fsr)[v]'] > threshold].index).argmax()
    cut_index = df[df['volt(fsr)[v]'] > threshold].index.values[cut_index_thr]

    
    # select longer part of the response
    first_df_tmp = df.loc[:cut_index]
    second_df_tmp = df.loc[cut_index:]
    
    if np.any(first_df_tmp['volt(fsr)[v]'] > threshold * 1.5) & np.any(second_df_tmp['volt(fsr)[v]'] > threshold * 1.5):
        new = df
    else:
        get_longer_df = lambda x,y : x if len(x[x['volt(fsr)[v]']>threshold]) > len(y[y['volt(fsr)[v]']>threshold]) else y
        new = get_longer_df(first_df_tmp, second_df_tmp)
        new = new.reindex_like(df)
        new['time[us]'] = df['time[us]']
        new['signal'] = df['signal']
        new.loc[new.index[new.isnull()['volt(fsr)[v]']], 'volt(fsr)[v]'] = 0    
    
    return new

def plot_epoch_graph(csvLocation, name, df, num, first_touch_index, last_touch_index, first_peak_index, first_peak_value, first_peak_time, second_peak_index, second_peak_value, second_peak_time, max_volt, title=''):
    # Plot graphs
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='white'
    fig, axes = plt.subplots(ncols=1, figsize=(5,5))

    # raw data
    axes.plot(df['time[us]'], 
              df['volt(fsr)[v]'], label='Response data')
    try:
        axes.axvspan(df[df['signal']!=0]['time[us]'].values[0], 
                     df[df['signal']!=0]['time[us]'].values[-1], 
                     color='r', alpha=0.3, label = 'Sound signal')
    except:
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        axes.text(right, top, 'No sound',
                  horizontalalignment='left',
                  verticalalignment='top',
                  transform=axes.transAxes)
    # labels
    try:
        axes.plot(second_peak_time, second_peak_value, 'ro', label='Second peak')
        axes.plot(first_peak_time, first_peak_value, 'bo', label='First peak')
        axes.axvline(df.loc[first_touch_index, 'time[us]'], alpha=0.2, label = 'First touch')
        axes.axvline(df.loc[last_touch_index, 'time[us]'], alpha=0.2, color='green', label = 'last touch')
    except:
        pass

    # plot settings
    axes.set_xlim(df['time[us]'].values[0], df['time[us]'].values[-1])
    axes.set_xlabel('Time [us]')
    axes.set_ylabel('Volt (fsr) [v]')
    
    # check here later
    axes.set_ylim(0, max_volt)
    axes.set_title(title+'Epoch : {}'.format(num))
    axes.set_xticklabels(axes.get_xticks().astype('int'), rotation=30)

    figloc = os.path.join(csvLocation, title+'{}_epoch_{}.png'.format(name, str(num).zfill(3)))
    plt.savefig(figloc, bbox_inches = 'tight')
    plt.close()


def get_sound_variation_epoch_nums(dfs):
    # Select only the middle 
    try:
        first_sound_indices = [df_tmp[df_tmp['signal']!=0].index.values[0] for df_tmp in dfs]
        sound_diff = np.diff(first_sound_indices)
        sound_diff_zscore = scipy.stats.zscore(sound_diff)

        # sound variation index
        sound_var_index = np.where(sound_diff_zscore > 3)[0] + 1
        sound_var_windows = [np.arange(x-4, x+5) for x in sound_var_index]

    except:
        sound_var_windows = []
        print('\tThere is no sound variation in the dfs')
    
    return sound_var_windows


def remove_noise(df, fsr_threshold, window_threshold=5):
    over_fsr_threshold_indices = df[df['volt(fsr)[v]'] > fsr_threshold].index
    over_fsr_threshold_indices_diff = np.diff(over_fsr_threshold_indices) - 1
    over_fsr_threshold_indices_diff_group = np.split(over_fsr_threshold_indices, 
                                                     np.nonzero(over_fsr_threshold_indices_diff)[0]+1)
    fsr_over_threshold_groups_long = [x for x in over_fsr_threshold_indices_diff_group if len(x) > 10]    
    df.loc[~df.index.isin(itertools.chain.from_iterable(fsr_over_threshold_groups_long)), 'volt(fsr)[v]'] = 0
    
    return df


def temporal_recalibration(csvLoc, 
                           fsr_threshold=40, 
                           max_volt=2500,
                           minimum_gap=0, first_window_index=20, second_window_index=50, 
                           fsr_to_volt_constant=3.3/4096,
                           interpolation_gap = 360):
    csvDirname = dirname(csvLoc)
    csvFilename = basename(csvLoc).split('.')[0]
    data = pd.read_csv(csvLoc, sep=',')
    print('='*80)
    print(csvFilename)
    #check here why it was deleted
#     data['volt(fsr)[v]'] = (3.3/4096) * data['volt(fsr)[v]']

    # Plot whole data
    fig, ax = plt.subplots(ncols=1, figsize=(30,10))
    ax.plot(data['volt(fsr)[v]'], 'k-')
    sound_on_indicies = data[data['signal']==1].index
    sound_on_firsts = sound_on_indicies[1:][np.diff(sound_on_indicies) > 10]
    for first_sound_index in sound_on_firsts:
        ax.axvline(first_sound_index, color='b', alpha=0.5)
    fig.suptitle(csvFilename, fontsize=30)
    fig.savefig(join(csvDirname, 'Whole_run.png'))
    plt.close()
    
    #interpolation
    # check here later why it's 360
    data_response_interp = interpolate_df(data, interpolation_gap)
            
    #data split
    if re.search('ph(\d00)', csvFilename):
        sound_interval = int(re.search('ph(\d00)', csvFilename).group(1))
        print('\tUsing sond interval of {} ms'.format(sound_interval))
        data_split = split_df(csvFilename, data_response_interp, interpolation_gap, sound_interval)
    else:        
        data_split = split_df(csvFilename, data_response_interp, interpolation_gap)
    
    missing_epochs = []
    error_epochs = []
    response_only_df = pd.DataFrame()
    timing_df = pd.DataFrame()
    figlocs = []
    
    # functionalize below scripts
    mainfig, mainax = plt.subplots(ncols=1, figsize=(10,5))
    for num, df_tmp in enumerate(data_split, 1): # looping from 1       
        df_tmp = remove_noise(df_tmp, fsr_threshold)
        
        if len(df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold]) < 10:
            missing_epochs.append(num)
            print('\tEpoch {} has no fsr response greater than the fsr_threshold ({})'.format(num, fsr_threshold))
            plot_epoch_graph(csvDirname, csvFilename, df_tmp, num, 
                 0, 0, 
                 0, 0, 0, 
                 0, 0, 0, max_volt)
            continue
        
        # if there are more than two responses
        elif np.any(np.diff(df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index)!=1):
            fig, ax = plt.subplots(ncols=1, figsize=(10,5))
            ax.plot(df_tmp['volt(fsr)[v]'], 'k--', alpha=0.8, label='Before')
            df_tmp = remove_partial_peaks(df_tmp, fsr_threshold)
            ax.plot(df_tmp['volt(fsr)[v]'], 'r-', alpha=0.8, label='After')
            fig.suptitle('Remove peaks')
            fig.savefig(join(csvDirname, '{}_removal_info_{}.png'.format(csvFilename, num)))
            plt.close()
        
#         try:
        # first and last contact
#         over_threshold_fsr_indices = df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index
#         over_threshold_fsr_indices_continuous = over_threshold_fsr_indices[np.diff(over_threshold_fsr_indices) > 10]
        first_touch_index = df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index.values[0]
        last_touch_index = df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index.values[-1]

        first_touch_time = df_tmp.loc[first_touch_index, 'time[us]']
        last_touch_time = df_tmp.loc[last_touch_index, 'time[us]']

        # Peak information
        first_peak_index, first_peak_value, first_peak_time, second_peak_index, second_peak_value, second_peak_time = get_peak_info(df_tmp, minimum_gap, first_window_index, second_window_index)

        try:
            # Sound information
            first_sound_index = df_tmp[df_tmp['signal']!=0].index.values[0]
            first_sound_time = df_tmp.loc[first_sound_index, 'time[us]']
        except:
            if np.any(df_tmp[df_tmp['signal'] != 0].index) == False:
                first_sound_index = df_tmp.index.values[0]
                first_sound_time = df_tmp.loc[first_sound_index, 'time[us]']
            else:
                os.error('Sound signal information')

        # Estimate area under curve
        area_under_curve_trapz = trapz(df_tmp.loc[first_touch_index:last_touch_index, 'volt(fsr)[v]'],
                                       df_tmp.loc[first_touch_index:last_touch_index, 'time[us]'])
        area_under_curve_simps = simps(df_tmp.loc[first_touch_index:last_touch_index, 'volt(fsr)[v]'],
                                       df_tmp.loc[first_touch_index:last_touch_index, 'time[us]'])

        # Graph
        plot_epoch_graph(csvDirname, csvFilename, df_tmp, num, 
                         first_touch_index, last_touch_index, 
                         first_peak_index, first_peak_value, first_peak_time, 
                         second_peak_index, second_peak_value, second_peak_time, max_volt)

        figlocs.append('nonmusicians/con01/con01_ph_{}.png'.format(str(num).zfill(3)))

        # Error detection
        # If first peak is far from the first touch
#         if (first_peak_index - first_touch_index) > 25:
#             pass
# #             error_epochs.append(num)

#         else:
        timing_df_tmp = pd.DataFrame({'epoch':[num],
                                      'sound_onset_time(SO)':first_sound_time,
                                      'first_touch_time(FT)':first_touch_time,
                                      'first_peak_time(FP)':first_peak_time,
                                      'second_peak_time(SP)':second_peak_time,
                                      'last_touch_time(LT)':last_touch_time,
                                      'first_peak_voltage(FP_volt)':fsr_to_volt_constant*first_peak_value,
                                      'second_peak_voltage(SP_volt)':fsr_to_volt_constant*second_peak_value,
                                      'area_under_curve_trapz':area_under_curve_trapz,
                                      'area_under_curve_simps':area_under_curve_simps,
                                      'FP - FT':first_peak_time-first_touch_time,
                                      'SP - FP':second_peak_time-first_peak_time,
                                      'LT - FT':last_touch_time-first_touch_time,
                                      'SO - FP':first_sound_time-first_peak_time,
                                      'SO - SP':first_sound_time-second_peak_time,
                                      'SO - FT':first_sound_time-first_touch_time,
                                      'SO - LT':first_sound_time-last_touch_time,
                                      'FP_volt - SP_volt':fsr_to_volt_constant*first_peak_value - fsr_to_volt_constant*second_peak_value
                                     })

        timing_df = pd.concat([timing_df, timing_df_tmp.set_index('epoch')])

        df_tmp = df_tmp[df_tmp['volt(fsr)[v]'] != 0].reset_index() # what is this line doing? --> moving the first touch part to have index of 0
        response_only_df = pd.concat([response_only_df, df_tmp['volt(fsr)[v]']], axis=1)

        mainax.plot(df_tmp.reset_index().index * interpolation_gap, df_tmp['volt(fsr)[v]'])
#     mainfig.show()
    
    dirName_filename_wo_ext = join(csvDirname, csvFilename)
    
    # Make GIF --> install image magick on windows
    cmd = 'convert -delay 30 -loop 0 {} {}'.format(' '.join(figlocs), dirName_filename_wo_ext+'.gif')
    os.popen(cmd).read()
    
    try:
        with open(dirName_filename_wo_ext+'.gif', 'rb') as f:
            display(Image(data=f.read()), format="gif")
    except:
        print('\tGIF file is not created. Install imageMagick and try again')
        cmd = 'magick convert -delay 30 -loop 0 {} {}'.format(' '.join(figlocs), dirName_filename_wo_ext+'.gif')
        os.popen(cmd).read()
        try:
            with open(dirName_filename_wo_ext+'.gif', 'rb') as f:
                display(Image(data=f.read()), format="gif")
        except:
            print('\tImageMagick error --> check images manually')
        
    # Tsplot
    response_only_df.columns = np.arange(1, len(response_only_df.columns)+1)
    response_only_df = response_only_df.stack().reset_index()
    response_only_df.columns = ['time', 'epoch', 'response']
    
    # check here, why multiply 360?
    response_only_df['time'] = response_only_df['time']*interpolation_gap 

    fig, axes = plt.subplots(ncols=1, figsize=(10,5))
    sns.tsplot(data=response_only_df, 
               time='time', 
               unit='epoch', 
               value='response')
    axes.set_xlabel('Time in us')
    axes.set_ylabel('Volt (fsr) in v')
    fig.show()

    # IRI estimation
    IRI_df = timing_df.diff().describe().loc[['count', 'mean', 'std', 'min', 'max']].T
    print('Epochs used in estimating IRI : {}'.format(', '.join([str(x) for x in timing_df.diff().index.values])))

    print('Error epochs : ', ', '.join([str(x) for x in error_epochs]))
    print('Missing epochs : ', ', '.join([str(x) for x in missing_epochs]))

    # save Dfs
    IRI_df.to_csv(dirName_filename_wo_ext+'_IRI.csv')    
    timing_df.to_csv(dirName_filename_wo_ext+'_timing.csv')
    
    variation_epoch_nums = get_sound_variation_epoch_nums(data_split)
    if variation_epoch_nums != []:
        var_dfs_list = []
        for variation_epoch_num in variation_epoch_nums:
            sub_df = timing_df.loc[list(variation_epoch_num)].T
            sub_df.columns = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            var_dfs_list.append(sub_df)
        var_out_excel = pd.ExcelWriter(dirName_filename_wo_ext+'_timing_variation.xlsx')
        for variable in timing_df.columns:
            var_df_tmp = pd.DataFrame([x.loc[variable] for x in var_dfs_list])#
            var_df_tmp.index = ['_'.join([str(y) for y in x]) for x in variation_epoch_nums]
            var_df_tmp.to_excel(var_out_excel, variable)
        var_out_excel.save()
            
    else:
        var_dfs_list = None
    
    return timing_df


def split_df(csvFilename, df, interpolation_gap, sound_interval=300):
    '''
    Splits the fsr data frame
    Default sound_interval : 300ms
    '''
    # Width of the window
    # - made into the unit of the post-interpolation
    cut_window_div = (sound_interval*1000) / interpolation_gap
    
    # Difference between consecutive sound
    pre_post_signal_diff = np.ediff1d(df['signal'])
    # where the difference is +1 : Signal turning on (0 --> 1)
    # where the difference is -1 : Signal turnning off (1 --> 0)

    # 1 has been added to catch the index of 'on' sound
    # 0 0 0 1 1 1 
    #     *       : pre_post_signal_dff = 0
    #       *     : 'on' sound
    signal_onset_index = np.where(pre_post_signal_diff==1)[0] + 1
    signal_onset_time = df.loc[signal_onset_index, 'time[us]'].values
    
    # signal index difference (between the first signal and the second)
    sounds_interval_index = (signal_onset_index[0] - signal_onset_index[1])
    
    # Index between the signal onsets
    half_point_between_signal = signal_onset_index + sounds_interval_index/2
    half_index_between_signal = get_nearest_indices(df, half_point_between_signal)
            
    # Split the first peak info from the rest
    df_splited_list = np.split(df, half_index_between_signal)[1:]

    # if the lengths of the epochs splited based on the sound onset is too variable
    # using zscore distribution
    # --> for the runs with half signal + half no signal
    z_score_threshold = 5
    df_splited_length_list = [len(x) for x in df_splited_list]
    df_splited_length_zscore = scipy.stats.zscore(df_splited_length_list)


    nameSearch = re.search('_ph', csvFilename)
    if np.any(df_splited_length_zscore > z_score_threshold) and len(df_splited_list) < 40:
        print('****split by zeros*****')
        no_sound_df_number = np.argmax(df_splited_length_zscore)
        df_no_sound = pd.concat(df_splited_list[no_sound_df_number:])

        zeros_index_array = df_no_sound[df_no_sound['volt(fsr)[v]'] == 0].index # indices of zero fsr
        zeros_index_diff_array = np.diff(zeros_index_array) # consecutive difference between the indices

        # change below later
        # indices of zero fsrs that have greater than 10 gaps to the next index
        zeros_index_diff_gt_10 = zeros_index_array[np.hstack([zeros_index_diff_array > 100, False])]

        # indices of zero fsrs that have greater than 300 gpas to the next index
#         zeros_index_diff_gt_10 = zeros_index_diff_gt_10[np.diff(zeros_index_diff_gt_10) > 300]

        # get index of between the zero fsr intervals
        zeros_index_half_points = np.diff(zeros_index_diff_gt_10)/2
        zeros_index_half_points_plus_last = np.hstack((zeros_index_half_points, zeros_index_half_points[-1]))
        zeros_index_diff_gt_10_middle = zeros_index_diff_gt_10 + zeros_index_half_points_plus_last
        zeros_index_diff_gt_10_middle_nearest = get_nearest_indices(df_no_sound, zeros_index_diff_gt_10_middle)

        # split the unsplited part of the df
        # reindex of the df_no_sound is required as np.split works on the index order
        df_splited_list_zero_cut = np.split(df_no_sound.reindex_like(df), zeros_index_diff_gt_10_middle_nearest)
        df_splited_first_zero_cut = df_splited_list_zero_cut[0].loc[df_no_sound.index[0]:]

        # append the splited epochs at the end of the part splited by the sound
        df_splited_list = df_splited_list[:no_sound_df_number] +                           [df_splited_first_zero_cut] +                           df_splited_list_zero_cut[1:]
                
    return df_splited_list


def get_peak_info(df, minimum_gap, first_window_index, second_window_index):
    # smoothing
    df['svolt(fsr)[v]'] = scipy.ndimage.filters.gaussian_filter1d(df['volt(fsr)[v]'], sigma=2)

    #local minima
    get_index_from_order = lambda df, order : df.iloc[order].index.values
    
    local_minima = argrelextrema(df['svolt(fsr)[v]'].values, np.less)[0]
    local_minima_index = get_index_from_order(df, local_minima)
    local_maxima = argrelextrema(df['svolt(fsr)[v]'].values, np.greater)[0]
    local_maxima_index = get_index_from_order(df, local_maxima)
    
    
    non_zero_index = df[df['volt(fsr)[v]'] != 0].index
    non_zero_index_diff = np.diff(non_zero_index) - 1
    non_zero_index_diff_group = np.split(non_zero_index,
                                         np.nonzero(non_zero_index_diff)[0]+1)

    # Simple one peak
    if len(non_zero_index_diff_group) == 1:
        # No local minima
        # Change here later
        if len(local_minima) == 0:  
            df['svolt_grad'] = np.gradient(df['svolt(fsr)[v]'])
            local_minima_grad = argrelextrema(df['svolt_grad'].values, np.less)[0]
            svolt_grad_max_index = df['svolt_grad'].idxmax()
            split_point_index = get_index_from_order(df, local_minima_grad)[0]

        # If more than one local minima
        else:
            split_point_index = local_minima_index[0]      
    # More than one separate peaks
    else:
         # split point : start point of the response part2
        # non_zero_index_diff_group[1] --> non_zero group (response part2)
        # non_zero_index_diff_group[1][0] --> the order of the first non-zero value in response part2
        split_point_index = non_zero_index_diff_group[1][0]

    first_peak_index = df.loc[:split_point_index, 'svolt(fsr)[v]'].idxmax()
    second_peak_index = df.loc[split_point_index:, 'svolt(fsr)[v]'].idxmax()
     
    # peak information
    first_peak_value = df.loc[first_peak_index, 'volt(fsr)[v]']
    first_peak_time = df.loc[first_peak_index, 'time[us]']
    
    second_peak_value = df.loc[second_peak_index, 'volt(fsr)[v]']
    second_peak_time = df.loc[second_peak_index, 'time[us]']     
    
    return first_peak_index, first_peak_value, first_peak_time, second_peak_index, second_peak_value, second_peak_time


def return_dict_to_remove(timing_files, subject_remove_dict):
    '''
    subject_remove_dict = {'mus07':[0,3,7,9]}
    '''
    dict_to_remove = {}
    for timing_file in timing_files:
        for subject_num, epochs_to_remove in subject_remove_dict.items():
            if subject_num in timing_file:
                dict_to_remove[timing_file] = epochs_to_remove
    
    for timing_file in timing_files:
        if timing_file not in dict_to_remove.keys():
            dict_to_remove[timing_file] = []


timing_files = []
for root, dirs, files in os.walk('due1_base/musicians'):
    for file in files:
        if re.search('_1_exp_2_dataLog_timing.csv', file):
            timing_files.append(join(root, file))



def grand_average_from_raw_data(csvLoc, 
                                epochs_to_remove=[1,2,3,4,5,6,7,8,9,10,130,131],
                                fsr_threshold=40, 
                                max_volt=2500,
                                minimum_gap=0, first_window_index=20, second_window_index=50, 
                                fsr_to_volt_constant=3.3/4096,
                                interpolation_gap = 360):
    csvDirname = dirname(csvLoc)
    csvFilename = basename(csvLoc).split('.')[0]
    data = pd.read_csv(csvLoc, sep=',')
    sound_on_indicies = data[data['signal']==1].index
    sound_on_firsts = sound_on_indicies[1:][np.diff(sound_on_indicies) > 10]
   
    #interpolation
    # check here later why it's 360
    data_response_interp = interpolate_df(data, interpolation_gap)
            
#     print(np.unique(np.diff(data_response_interp['time[us]'])))
    #data split
    if re.search('ph(\d00)', csvFilename):
        sound_interval = int(re.search('ph(\d00)', csvFilename).group(1))
        data_split = split_df(csvFilename, data_response_interp, interpolation_gap, sound_interval)
    else:        
        data_split = split_df(csvFilename, data_response_interp, interpolation_gap)
    
    missing_epochs = []

    # functionalize below scripts
    final_dfs = []
    for num, df_tmp in enumerate(data_split, 1): # looping from 1
        if num not in epochs_to_remove:
            df_tmp = remove_noise(df_tmp, fsr_threshold)

            if len(df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold]) < 10:
                missing_epochs.append(num)
                continue

            # if there are more than two responses
            elif np.any(np.diff(df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index)!=1):
                df_tmp = remove_partial_peaks(df_tmp, fsr_threshold)
        

            #make first touch their 0 index
            first_touch_index = df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index.values[0]
            last_touch_index = df_tmp[df_tmp['volt(fsr)[v]'] > fsr_threshold].index.values[-1]

            first_touch_time = df_tmp.loc[first_touch_index, 'time[us]']
            last_touch_time = df_tmp.loc[last_touch_index, 'time[us]']
            
            df_tmp = df_tmp.loc[first_touch_index:]#.reindex(np.arange(2000))
#             df_tmp['volt(fsr)[v]'] = df_tmp['volt(fsr)[v]'].fillna(0)
            final_dfs.append(df_tmp)
    return final_dfs


def get_subject_average(subject_dfs):
    subject_dfs_reindexed = [x.reset_index() for x in subject_dfs]
    
    # align sound signal
    sound_onsets = [x[x['signal']==1].index.values[0] for x in subject_dfs_reindexed]
    unique_sound_onsets = set(sound_onsets)

    new_subject_dfs = []
    sound_onsets = []
    
#     for subject_df in subject_dfs_reindexed:
#         subject_sound_onset = subject_df[subject_df['signal']==1].index.values[0]
#         to_shift = 900-subject_sound_onset
#         subject_df = subject_df.shift(to_shift).reset_index()
#         new_subject_dfs.append(subject_df)
    
    # epoch volt(fsr)[v] average
    all_epoch_concat = pd.concat([x['volt(fsr)[v]'] for x in subject_dfs_reindexed], axis=1)
    all_epoch_concat = all_epoch_concat.fillna(0)

    # time 
    all_sound_concat = pd.concat([x['signal'] for x in subject_dfs_reindexed], axis=1)
    all_sound_concat = all_sound_concat.fillna(0)

    subject_average_df = pd.concat([all_epoch_concat.mean(axis=1), 
                                    all_sound_concat.mean(axis=1)], axis=1)
    
    subject_average_df.columns = ['volt(fsr)[v]', 'signal']
    return subject_average_df



def get_group_average(subject_average_dfs):
    group_df_concat_fsr = pd.concat([x['volt(fsr)[v]'] for x in subject_average_dfs], axis=1)
    group_df_concat_fsr_average = group_df_concat_fsr.fillna(0).mean(axis=1)
    group_df_concat_signal = pd.concat([x['signal'] for x in subject_average_dfs], axis=1)
    group_df_concat_signal_average = group_df_concat_signal.fillna(0).mean(axis=1)
    
    group_df = pd.concat([group_df_concat_fsr_average,
                          group_df_concat_signal_average],
                         axis=1)
    return group_df


if __name__ == '__main__':
    temporal_recalibration(sys.argv[1])
