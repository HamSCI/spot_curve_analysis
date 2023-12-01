# Curve edge autoplot
# Authors:   W. Engelke, N. Callahan, University of Alabama
# Enhancements from N. Frissell, University of Scranton
# Dec. 2023
# Reads in a csv file containing spot counts for a given band/day/continent
# Outputs heat map of data, calculates/plots lower edge (Minimum Useful Range)
# using gradient method plus Butterworth filter or Lowess smoothing
# Will do a single csv file or process en entire directory of csv files

import datetime
from   datetime import datetime,timedelta, timezone
import pytz
import numpy as np
import pandas as pd

from   scipy.interpolate import interp1d
from   scipy.signal import filtfilt, butter, lfilter
from   scipy import signal

from   PIL import Image, ImageColor, ImageFont, ImageDraw

import matplotlib as mpl
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import glob
import traceback
import os
import csv
import bottleneck as bn
import seaborn as sns
import statsmodels.api as sm

# set some defaults

showPlots = True  # set this to False to suppress showing of plots
showfinal = True
FirstLine = True # to cause a header line to be included in the summary.csv file
first_pass = True
use_Butterworth = True

conti = "NA"   # North America
theBand = '20' # band in meters

# point this to directory containing the input csv spot files
inputDirectory = ".//data_files//*.csv"

# to save the generated plots, set this to True and set save directory
savePlots = True
saveDirectory = './/curve_comboplots'

saveSummary = False # if True, save summary file at end of processing
typeList = 'T0'


# Matplotlib settings to make the plots look a little nicer.
plt.rcParams['font.size']      = 12
#plt.rcParams['font.weight']    = 'bold'
plt.rcParams['axes.grid']      = True
plt.rcParams['axes.xmargin']   = 0
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['figure.figsize'] = (17,10)

def gradient_edge(img):
    new_img = np.gradient(img, axis=1)
    move_img = bn.move_mean(new_img, 35, axis=1)
    move_img = np.nan_to_num(move_img, nan=0.0)
    edge = np.argmax(move_img, axis=1)
    return edge

def lowess_smooth(edge, window_size=40, x=None):
    if x is None:
        x = np.linspace(0, len(edge), len(edge))
    smooth_edge = sm.nonparametric.lowess(edge, x, frac=window_size/len(arr))[:,1]    
    return smooth_edge

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



################ Section for FFT ######################################################



# FFT code goes here ###################





 
################## Section to write results of above analysis to summary csv file ###########    
    plt.close()

    return # temporary until the Sxx FFT data is working

    if saveSummary == True:
    
        with open(outputDirectory  + 'summary.csv', 'a') as fd:
          if FirstLine == True:
              fd.write('Date,Minutes,Raw Max,Raw Sum,Noise Sum,rejected,No_LSTID,LSTID,\n')
              FirstLine = False
          fd.write(basename + ',' + str(data_length) + ',' + '{0:1.1f}'.format(Sxx_1.max())+','+'{0:1.1f}'.format(Sxx_1.sum()) + 
                    ',' + noise_measure)
          if rejected == True:
              fd.write(',' + '{0:1.1f}'.format(Sxx_1.sum()))
          if No_LSTID == True:
              fd.write(',,' + '{0:1.1f}'.format(Sxx_1.sum()))
          if LSTID == True:
              fd.write(',,,' + '{0:1.1f}'.format(Sxx_1.sum()))
          if current_month != last_month:
            fd.write("," + last_year + "-" + last_month + "," + '{0:1.1f}'.format(Sxx_1_max_month_total)+','+'{0:1.1f}'.format(Sxx_1_int_month_total))
            monthlyResults = [(last_year + "-" + last_month), ('{0:1.1f}'.format(Sxx_1_max_month_total)), ('{0:1.1f}'.format(Sxx_1_int_month_total))]
            summaryResults.append(monthlyResults)

            Sxx_1_max_month_total = Sxx_1.max()
            Sxx_1_int_month_total = Sxx_1.sum()
            last_month = current_month
            last_year = current_year
          else:
            if LSTID == True:
                Sxx_1_max_month_total = Sxx_1_max_month_total + Sxx_1.max()
                Sxx_1_int_month_total = Sxx_1_int_month_total + Sxx_1.sum()
          fd.write("\n")

            
    plt.close('all')



def plot_raw(arr, title='', xlabel='Time (min)', ylabel='Height (km)', ax=None):
    """ Simple way to display the image in matplotlib """
    arr = arr.transpose()[::-1,:]
    if ax is None:
        
        plt.xlim((720,1440))
        plt.title(title)
        sns.heatmap(arr, cmap='jet', robust=True, cbar=False) # cmaps was magma
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((720,1440))
       
        y_locs, y_texts = plt.yticks()
        yticks = np.asarray([float(str(a.get_text())) for a in y_texts[::2]])
        plt.yticks(300 - y_locs[::2], yticks / 100)

        x_locs, x_texts = plt.xticks()
        plt.xticks(x_locs[::2], x_texts[::2])
        plt.xlim((720,1440))
   
        plt.show()
    else:
        
        ax.set_title(title)
        ax.set_xlim((720,1440))
        sns.heatmap(arr, cmap='jet', robust=True, cbar=False, ax=ax) # cmap was magma
        ax.set_yticks(np.arange(0,400,100), labels=['300','200','100','0'])
        ax.set_ylabel("Range (10 km)")
        ax.set_xticks(np.arange(0,1441, 120), labels=['00:00','02:00','04:00','06:00','08:00','10:00','12:00' , \
                                                '14:00','16:00','18:00','20:00','22:00','24:00' ], rotation=45)
        
        ax.set_xlabel('Time (UTC)')
    return


def plot_images_on_date(img,theDate):
    fix, ax = plt.subplots(figsize=(7.5,10))
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    plot_raw(img, title='Initial Data - ' + theDate,ax=ax1) # FIRST PLOT, raw data

    new_img = np.gradient(img, axis=1)

    ax1.set_xlim((720,1440))

  #  plt.show()
    move_img = bn.move_mean(new_img, 35, axis=1)
    move_img = np.nan_to_num(move_img, nan=0.0)
    # SECOND PLOT, gradient
    plot_raw(move_img, title='Columnwise Gradient Moving Average (.5km)',ax=ax2)
    ax2.set_xlim((720,1440))
    plt.tight_layout()
   # plt.show()
    assert np.isfinite(move_img.ravel()).all()

    noisy_arr = np.argmax(move_img, axis=1)
 #   plt.figure(figsize=(15,5))
    ax3.scatter(np.arange(0, noisy_arr.shape[0], 1), noisy_arr, label='Gradient Points')

    if use_Butterworth == False:

   # produce smoothed curve using Lowess algorithm
        smooth_arr = lowess_smooth(noisy_arr)
        ax3.plot(smooth_arr, color='red', label='Smoothed')
        ax3.set_title('Argmax Averaged, Lowess smoothing')

    else:

    # Produce smoothed curve using Butterworth filter
    # FILTERBREAK is # half-cycles in a minute; a 60-min. period wave has (1/60) cycles/min, or
    # (1/120) half-cycles per min. Sampling period is 1 min. See doc.: scipy.signal.butter
        FILTERBREAK = (1./120.)*2. # 2 times no. of half-cycles per sample period (1 min.)
        FILTERORDER = 5
        b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='lowpass', fs=1.)
        smooth_arr = filtfilt(b, a, noisy_arr) # plotedge is the filtered signal     
        ax3.plot(smooth_arr, color='red', label='Smoothed')
        ax3.set_title('Argmax Averaged, Butterworth filter')
       
    ax3.set_ylabel('Range (10 km)')
    ax3.set_xlabel('Time (minutes)')
   
    ax3.set_ylim([0,300])
    ax3.set_xlim((720,1440))
    ax3.legend()
    outputFilename = theDate + "_" + conti + "_" + theBand + "m.png"
    if savePlots:
        outfile = os.path.join(saveDirectory, outputFilename)
        plt.savefig(outfile)
        print("Saved file:", outputFilename)
    if showPlots:
        plt.show()

    plt.close('all')

    return # this is not required, but shows intentional end of function




############################# MAIN #####################################################

#set up array to hold spot counts
dist = [0] * 300         # this is a column (x) of distance values in 10s of km range
minute = [dist] * 1440   # this is an array of columns, one per minute
spotcount = np.array(minute, dtype=np.uint32) # 1440 columns of 300 ranges
arr = spotcount  # to accommodate the lowess implementation
startdate = ''
enddate = ''

fileList = glob.glob(inputDirectory)

jobtype = '0'
while jobtype != '1' and jobtype != '2':
    print("Job type: Enter 1 to process a single date, or 2 to process a range of dates")
    jobtype = input()

if jobtype == '1':
    print("Date to process (YYYY-MM-DD)?")
    startdate = input()
    startdate_obj =  datetime.strptime(startdate, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    enddate_obj = startdate_obj
else:
    print("Start date? (YYYY-MM-DD); or Enter to start at beginning and process entire directory")
    startdate = input()
    if startdate != '':
        startdate_obj =  datetime.strptime(startdate, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        print("End date? (YYYY-MM-DD); Enter to start at start date and process all later files")
        enddate = input()
        if enddate != '':
            enddate_obj =  datetime.strptime(enddate, '%Y-%m-%d').replace(tzinfo=timezone.utc)


startdate_ts = datetime.timestamp(startdate_obj)
enddate_ts   = datetime.timestamp(enddate_obj)

# Build array of columns (minutes), where each entry contains the number of th rang bin
# where the max gradient of the column appears

for datafile in fileList:
    f_dat = datafile.rsplit('/') # extract the date from the file path
#    print("datafile:",datafile)
 #   print("f_dat=",f_dat)
    theDate = f_dat[3][6:16]
    theDate_obj = datetime.strptime(theDate, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    theDate_ts = datetime.timestamp(theDate_obj)
    if jobtype == '1': # are we processing a single date
        if startdate != theDate:
            continue
    else:
        if theDate_ts < startdate_ts or  \
           theDate_ts > enddate_ts:
            continue
    print("Processing ",theDate)


    with open(datafile,newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=",")
        x_counter = 0
       # maxcount = 0
        for row in reader:
            np_row = np.array(list(row),dtype=np.uint32)
            spotcount[x_counter] = np_row
            x_counter += 1
        maxcount = np.max(spotcount)

        plot_images_on_date(spotcount,theDate)



print("end of processing")
if saveSummary:
    with open(outputDirectory  + 'summary.csv', 'a') as fd:
      fd.write("\n")
      for month in summaryResults:
        for value in month:
            fd.write(value + ",")
        fd.write('\n')


        
    
