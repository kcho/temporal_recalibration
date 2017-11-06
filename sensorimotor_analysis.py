import os
from os.path import join
import argparse
import textwrap
import sys
import time
import progressbar

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from scipy.signal import butter, filtfilt
from scipy.integrate import simps
from numpy import trapz

import matplotlib.pyplot as plt
import seaborn as sns

# kcho, Monday, November 06, 2017

def interpolate_df(df, gap):
    f = interpolate.interp1d(df['time[us]'], df['volt(fsr)[v]'])
    y = f(np.arange(df['time[us]'].min(), df['time[us]'].max(), gap))

    f_signal = interpolate.interp1d(df['time[us]'], df['signal'], kind='nearest')
    y_signal = f_signal(np.arange(df['time[us]'].min(), df['time[us]'].max(), gap))
    
    data_reponse_interp = pd.DataFrame({'time[us]':np.arange(df['time[us]'].min(), 
                                                             df['time[us]'].max(), gap),
                                        'volt(fsr)[v]':y,
                                        'signal':y_signal})
    return data_reponse_interp


def split_df(df):
    # split the dataframe
    # for the variation project, the timing of the signal should lie on the center of x-axis

    # difference between consecutive element
    pre_post_signal_diff = np.ediff1d(df['signal'])

    if len(pre_post_signal_diff) > 20:
        # where the difference is +1 : Signal turning on (0 --> 1)
        # 1 has been added to return the index of first 'on'

        signal_first_ones_index = np.where(pre_post_signal_diff==1)[0] + 1
        signal_first_ones_time = df.loc[signal_first_ones_index, 'time[us]'].values

        signal_last_ones_index = np.where(pre_post_signal_diff==-1)[0]
        signal_last_ones_time = df.loc[signal_last_ones_index, 'time[us]'].values


        # Estimate the time difference between each signal
        # [1] has been appended at the end in order to make the length of the matrices equal
        #cut_window_div_index = np.append(np.ediff1d(signal_first_ones_index), [1])
        cut_window_div_index = np.ediff1d(signal_first_ones_index)
        #cut_window_div_index = signal_first_ones_index - 5000

        # divided by 2 in order to shift the index by half
        # this will make the sound signal to be near the center
        #cut_time = signal_first_ones_index - (cut_window_div_index)/2
        cut_time = signal_first_ones_index[:-1] + (cut_window_div_index)/2

    # no sound signal analysis should be added here

    #data_split = np.split(df, cut_time.astype('int'))[1:]
    data_split = np.split(df, cut_time.astype('int'))
    return data_split

    
def remove_partial_peaks(df, touch_threshold):
    cut_index_thr = np.diff(df[df['volt(fsr)[v]'] > touch_threshold].index).argmax()
    cut_index = df[df['volt(fsr)[v]'] > touch_threshold].index.values[cut_index_thr]

    # select longer part of the response
    first_df_tmp = df.ix[:cut_index]
    second_df_tmp = df.ix[cut_index:]
    get_longer_df = lambda x,y : x if len(x[x['volt(fsr)[v]']>touch_threshold]) > len(y[y['volt(fsr)[v]']>touch_threshold]) else y
    new = get_longer_df(first_df_tmp, second_df_tmp)
    new = new.reindex_like(df)
    new['time[us]'] = df['time[us]']
    new['signal'] = df['signal']
    new.ix[new.index[new.isnull()['volt(fsr)[v]']], 'volt(fsr)[v]'] = 0    
    
    return new

def get_peak_info(df):
    # Estimating second peak
    # Smooth data : strong smoothing
    b, a = butter(4, 0.01)
    df['svolt(fsr)[v]'] = filtfilt(b, a, df['volt(fsr)[v]'])

    max_response_index_from_strong_smoothing = df['svolt(fsr)[v]'].idxmax()
    second_window_index = 50
    higher_window_bound_2nd_peak = max_response_index_from_strong_smoothing + second_window_index

    # Second peak
    # Always on the right of the max_response_index_from _strong_smoothing
    second_peak_index = df['volt(fsr)[v]'].ix[max_response_index_from_strong_smoothing:higher_window_bound_2nd_peak].idxmax()
    second_peak_value = df.ix[second_peak_index, 'volt(fsr)[v]']
    second_peak_time = df.ix[second_peak_index, 'time[us]']

    # First peak
    # must appear before the second peak
    # max diff peak
    max_diff_peak_index = (df['volt(fsr)[v]'] - df['svolt(fsr)[v]']).ix[:second_peak_index].idxmax()
    first_window_index = 20

    first_peak_index = df['volt(fsr)[v]'].ix[max_diff_peak_index-first_window_index:max_diff_peak_index+first_window_index].idxmax()
    first_peak_value = df.ix[first_peak_index, 'volt(fsr)[v]']
    first_peak_time = df.ix[first_peak_index, 'time[us]']

    return first_peak_index, first_peak_value, first_peak_time, second_peak_index, second_peak_value, second_peak_time

def plot_epoch_graph(df, num, first_touch_index, last_touch_index, first_peak_index, first_peak_value, first_peak_time, second_peak_index, second_peak_value, second_peak_time):
    # Plot graphs
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='white'
    fig, axes = plt.subplots(ncols=1, figsize=(5,3))

    # raw data
    axes.plot(df['time[us]'], 
              df['volt(fsr)[v]'], label='Response data')
    axes.axvspan(df[df['signal']!=0]['time[us]'].values[0], 
                 df[df['signal']!=0]['time[us]'].values[-1], 
                 color='r', alpha=0.3, label = 'Sound signal')

    # labels
    axes.plot(second_peak_time, second_peak_value, 'ro', label='Second peak')
    axes.plot(first_peak_time, first_peak_value, 'bo', label='First peak')
    axes.axvline(df.ix[first_touch_index, 'time[us]'], alpha=0.2, label = 'First touch')
    axes.axvline(df.ix[last_touch_index, 'time[us]'], alpha=0.2, color='green', label = 'last touch')

    # plot settings
    axes.set_xlim(df['time[us]'].values[0], df['time[us]'].values[-1])
    axes.set_xlabel('Time [us]')
    axes.set_ylabel('Volt (fsr) [v]')
    #axes.set_ylim(0, 2500)
    axes.set_ylim(0, (3.3/4096) * 2500)
    axes.set_title('Epoch : {}'.format(num))
    axes.set_xticklabels(axes.get_xticks().astype('int'), rotation=30)
    figloc = join(os.getcwd(), 'temporal_{}.png'.format(str(num).zfill(3)))
    plt.savefig(figloc, bbox_inches = 'tight')
    plt.close()
    
    
def sensorimotor_asynchrony(csvLoc):
    data = pd.read_csv(csvLoc, sep=',')

    data['volt(fsr)[v]'] = (3.3/4096) * data['volt(fsr)[v]']

    # Interpolation
    data_reponse_interp_360 = interpolate_df(data, 360)
    
    # Data split
    data_split = split_df(data_reponse_interp_360)
    print(len(data_split), 'data_split length')
    
    missing_epochs = []
    error_epochs = []
    response_only_df = pd.DataFrame()
    timing_df = pd.DataFrame()
    figlocs = []

    # Progressive bar
    bar = progressbar.ProgressBar(max_value=len(data_split), redirect_stdout=True)

    #touch_threshold = 80
    touch_threshold = (3.3/4096) * 80
    # Iterate each epochs
    for num, df_tmp in enumerate(data_split, 1):
        # If there is no touch response above the threshold,
        # add this epoch to the missing epoch list
        if len(df_tmp[df_tmp['volt(fsr)[v]'] > touch_threshold]) == 0:
            missing_epochs.append(num)
            continue

        # If there are more than two responses
        elif np.any(np.diff(df_tmp[df_tmp['volt(fsr)[v]'] > touch_threshold].index)!=1):
            df_tmp = remove_partial_peaks(df_tmp, touch_threshold)
            
        # first and last contact
        first_touch_index, last_touch_index = df_tmp[df_tmp['volt(fsr)[v]'] > touch_threshold].index.values[[0, -1]]
        first_touch_time = df_tmp.ix[first_touch_index, 'time[us]']
        last_touch_time = df_tmp.ix[last_touch_index, 'time[us]']
        
        # Peak information
        first_peak_index, first_peak_value, first_peak_time, second_peak_index, second_peak_value, second_peak_time = get_peak_info(df_tmp)

        # Sound information
        first_sound_index = df_tmp[df_tmp['signal']!=0].index.values[0]
        first_sound_time = df_tmp.ix[first_sound_index, 'time[us]']
        
        # Estimate area under curve
        area_under_curve_trapz = trapz(df_tmp.ix[first_touch_index:last_touch_index, 'volt(fsr)[v]'],
                                       df_tmp.ix[first_touch_index:last_touch_index, 'time[us]'])
        area_under_curve_simps = simps(df_tmp.ix[first_touch_index:last_touch_index, 'volt(fsr)[v]'],
                                       df_tmp.ix[first_touch_index:last_touch_index, 'time[us]'])

        # Graph
        plot_epoch_graph(df_tmp, num, 
                         first_touch_index, last_touch_index, 
                         first_peak_index, first_peak_value, first_peak_time, 
                         second_peak_index, second_peak_value, second_peak_time)

        figlocs.append(join(os.getcwd(), 'temporal_{}.png'.format(str(num).zfill(3))))
        
        # Error detection
        # If first peak is far from the first touch
        if (first_peak_index - first_touch_index) > 25:
            pass
#             error_epochs.append(num)
        else:
            timing_df_tmp = pd.DataFrame({'epoch':[num],
                                          'sound_onset_time':first_sound_time,
                                          'first_touch_time(FT)':first_touch_time,
                                          'first_peak_time(FP)':first_peak_time,
                                          'second_peak_time(SP)':second_peak_time,
                                          'last_touch_time(LT)':last_touch_time,
                                          'first_peak_voltage(FP_volt)':first_peak_value,
                                          'second_peak_voltage(SP_volt)':second_peak_value,
                                          'area_under_curve_trapz':area_under_curve_trapz,
                                          'area_under_curve_simps':area_under_curve_simps,
                                          'SP - FP':second_peak_time-first_peak_time,
                                          'LT - FT':last_touch_time-first_touch_time,
                                          'SO - FP':first_sound_time-first_peak_time,
                                          'SO - SP':first_sound_time-second_peak_time,
                                          'SO - FT':first_sound_time-first_touch_time,
                                          'SO - LT':first_sound_time-last_touch_time,
                                          'FP_volt - SP_volt':first_peak_value - second_peak_value
                                              })
            timing_df = pd.concat([timing_df, timing_df_tmp.set_index('epoch')])
        
            df_tmp = df_tmp[df_tmp['volt(fsr)[v]'] != 0].reset_index()
            response_only_df = pd.concat([response_only_df, df_tmp['volt(fsr)[v]']], axis=1)
            plt.plot(df_tmp.reset_index().index * 360, df_tmp['volt(fsr)[v]'])

        # Update progress bar
        bar.update(num)

    dirName = os.path.dirname(csvLoc)
    filename_wo_extention = os.path.basename(csvLoc).split('.')[0]
    dirName_filename_wo_ext = os.path.join(dirName, filename_wo_extention)
    plt.savefig(dirName_filename_wo_ext+'_all_plots.png', bbox_inches = 'tight')

    # Make GIF
    cmd = 'convert -delay 30 -loop 0 {} {}'.format(' '.join(figlocs), dirName_filename_wo_ext+'.gif')
    os.popen(cmd).read()
    for pngImg in figlocs:
        os.remove(pngImg)

    # Tsplot
    response_only_df.columns = np.arange(1, len(response_only_df.columns)+1)
    response_only_df = response_only_df.stack().reset_index()
    response_only_df.columns = ['time', 'epoch', 'response']
    response_only_df['time'] = response_only_df['time']*360

    fig, axes = plt.subplots(ncols=1, figsize=(10,5))
    sns.tsplot(data=response_only_df, 
               time='time', 
               unit='epoch', 
               value='response')
    axes.set_xlabel('Time in us')
    axes.set_ylabel('Volt (fsr) in v')
    plt.savefig(dirName_filename_wo_ext+'_tsplot.png', bbox_inches = 'tight')

    # IRI estimation
    IRI_df = timing_df.diff().describe().ix[['count', 'mean', 'std', 'min', 'max']].T
    print('Epochs used in estimating IRI : {}'.format(', '.join([str(x) for x in timing_df.diff().index.values])))
    print('Error epochs : ', ', '.join([str(x) for x in error_epochs]))
    print('Missing epochs : ', ', '.join([str(x) for x in missing_epochs]))

    # save Dfs
    IRI_df.to_csv(dirName_filename_wo_ext+'_IRI.csv')    
    timing_df.to_csv(dirName_filename_wo_ext+'_timing.csv')
    
    return timing_df

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            {codeName} : Return timing information of the tapping experiment
            ========================================
            eg) {codeName} -d /Users/kevin/NOR04_CKI
                Analyse all the text files within the directory
            eg) {codeName} -i /Users/kevin/NOR04_CKI/ha.txt
                Analyse ha.txt
            '''.format(codeName=os.path.basename(__file__))))
    parser.add_argument(
        '-d', '--directory',
        help='Data directory location')
    parser.add_argument(
        '-i', '--inputFile',
        help='Lodation of the data log textfile',
        nargs='+')

    args = parser.parse_args()

    if args.directory:
        textFiles = [join(args.directory, x) for x in os.listdir(args.directory) if x.endswith('txt')]
        for textFile in textFiles:
            print(textFile)
            try:
                sensorimotor_asynchrony(textFile)
            except:
                pass
    if args.inputFile:
        for textFile in args.inputFile:
            print(textFile)
            sensorimotor_asynchrony(textFile)
    else:
        sys.exit('Please read how to use the script')

