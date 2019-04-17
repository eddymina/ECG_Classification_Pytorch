import time 
import pandas as pd 
import numpy as np
from collections import Counter
from scipy import signal
from scipy.signal import find_peaks, resample
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
from os import listdir
from os.path import isfile, join
import sys
import warnings 

classes_further= {'N':'Normal beat','L':'Left bundle branch block beat','R':'Right bundle branch block beat',
'A':'Atrial premature beat','a':'Aberrated atrial premature beat','J':'Nodal (junctional) premature beat',
'S':'Supraventricular premature beat','V':'Premature ventricular contraction','F':'Fusion of ventricular and normal beat',
'[':'Start of ventricular flutter/fibrillation','!':'Ventricular flutter wave',']':'End of ventricular flutter/fibrillation','e':'Atrial escape beat',
'j':'Nodal (junctional) escape beat','E':'Ventricular escape beat','/':'Paced beat',
'f':'Fusion of paced and normal beat','x':'Non-conducted P-wave (blocked APB)','Q':'Unclassifiable beat',
'|':'Isolated QRS-like artifact'}

#https://arxiv.org/abs/1805.00794 (original paper)

def longest(list1):
    """
    Returns index for length of longest list for a list of lists 
    """
        
    return max(enumerate(list1), key = lambda tup: len(tup[1]))[0]
    
def moving_average(x,window):
    """
    Numpy based moving average function.
    Input is a signal and window size 
    Output is average 
    """
    return np.convolve(x, np.ones((window,))/window, mode='valid')

def amplitude_ratio(ecg_signal):
    """
    From signal get Ratio of 
    Postive Signal to Negative 
    """
    return ecg_signal[np.where(ecg_signal>0)].mean()/abs(ecg_signal[np.where(ecg_signal<0)].mean())

def distribution_bar(patients,classes,classes_reducer=None):
    """
    Generate simple bar plot graphic
    of the condition distributions 
    across all patients and per patient

    Returns a patient information dictionary 
    that has a counter list of each examined class
    patient_dic['101']= [('N', 1860), ('S', 3), ('V', 0), ('F', 0), ('Q', 2)]

    """
    print('Generating_plot(s)...')
    patient_dic= {}
    for patient in patients:
        sig,ecg_notes= get_patient_data(patient)
        patient_list= []
        for c in classes.values():
            summed=0
            if classes_reducer != None: 
                for i in classes_reducer[c]:
                    summed += Counter(ecg_notes.type)[i]
                patient_list.append((c, summed)) 
            else:
                patient_list.append((c, Counter(ecg_notes.type)[c]))  
        patient_dic[patient]= patient_list    

    r_len = len(patients)
    barWidth = 0.25
    bars={}
    r={}
    for i in range(len(classes)):
        bars[i] = [x[i][1] for x in list(patient_dic.values())]
        if i == 0:
            r[0] = np.arange(r_len).tolist()
        else:
            r[i] = [x + 1/8 for x in r[i-1]]


    plt.figure(figsize=(25,20))
    # Make the plot
    plt.subplot(211)
    condition_count={}
    for condition,count in bars.items():
        condition_count[condition] = sum(count)
            
    s = pd.Series(
        list(condition_count.values()),
        index = [classes[i] for i in list(condition_count.keys())])
    s.plot(kind='bar')
    plt.ylabel('Count')
    plt.xlabel('Condition')
    
    plt.subplot(212)
    for i in range(0,len(classes)):
        plt.bar(r[i], bars[i], width=barWidth, edgecolor='white', label=classes[i])
    n=r[0]
    plt.xticks([n + barWidth for n in range(r_len)],patients)
    plt.ylabel('Count', fontweight='bold')
    plt.xlabel('Patients ({})'.format(r_len), fontweight='bold')
    plt.legend()
    plt.show()
    return patient_dic

def update_progress(progress,barLength=30):
    """
    Simple Progess Bar. Used in each iteration for 
    each patient that is processed. Shows percent 
    completion and an arrow 
    
    """
    s=time.time()
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    t=time.time()-s;
    block = int(round(barLength*progress))
    text = "\rPercent: [{}] {:.2f}% {}".format( ">"*block + "-"*(barLength-block), progress*100,status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def zero_pad(lst):
    """
    Import a list of lists [[1 x n],[1 x c],[1 x m]]
    Create np array size: with the number of columns 
    being the length of longest list within list of lists.
    All shorter other links are zero padded. 
    
    Ex.
    Given [[1 x n],[1 x c],[1 x m]] and m > n > c
    return array [3,m]
    
    [[1 x 1],[1 x 2],[1 x 3]] --> [a1 0 0],
                                  [b1 b2 0],
                                  [m1 m2m m3]
                                  
    Input: [[list],[list],[list],...,]
    Output: Zero Padded Array [len(list) X len_longest_list]
    
    """
    pad = len(max(lst, key=len))
    return np.array([i + [0]*(pad-len(i)) for i in lst])

def get_HR(peaklist,fs):
    """
    Takes in List of Sample Peaks and sampling freq.
    Computes average distance between peaks w/r time.
    Returns BPM
    
    Inputs: 
        peaklist:: list of HR R peaks 
        fs:: sampling rate 
    Output: 
        HR (float)
    """
    RR_list = []
    for beat in range(len(peaklist)-1):
        RR_interval = (peaklist[beat+1] - peaklist[beat]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs)) #Convert sample distances to s distances
        RR_list.append(ms_dist)
    bpm =60 / np.mean(RR_list) # (1 minute) / average R-R interval of signal
    return bpm

def all_patients():
    """
    Assumes that all folder called mit_data is next folder 
    in current directory. This function can be bypassed if
    user acquires list of patients (str or int). 
    
    Inputs: None
    
    Outputs: List of Patients 
    
    """
    
    onlyfiles = [f for f in listdir(os.getcwd()+'/mit_data') if isfile(join(os.getcwd()+'/mit_data', f))]
    return np.unique([file[0:3] for file in onlyfiles])[1:].tolist()

def get_patient_data(patient,norm=True, sample_plot=False):
    """
    Assumes that all folder called mit_data is next folder 
    in current directory. Can change this function internally 
    or write your own personalized one. 
    
    Input: 
        patient:: Patient Number [Str or Int]
        norm:: (optional) =True --> Normalize Data 
        sample_plot:: (optional) Show Patient ECG Signal [True or False]
    Output: 
        Normalized Signal Data, Ecg Notes 
            Ecg_Notes:: Labeled Sample Peaks and Heart Conditions 
            Ecg_Data:: np.array of signal
    """
    widths= [4,8,11,6,3,5,5,8]
    
    patient=str(patient)
    ecg_notes= pd.read_fwf('mit_data/{}annotations.txt'.format(patient),widths=widths).drop(['Unnamed: 0'],axis=1)
    ecg_data= pd.read_csv('mit_data/{}.csv'.format(patient))
    ecg_data.columns= ['samp_num','signal','V']
    ecg_notes=ecg_notes[['Sample #','Type','Aux']]
    ecg_notes.columns=['sample_num','type','aux']
    if norm == True:
        ecg_data.signal= z_norm(ecg_data.signal)
    if sample_plot == True:
        peaklist= ecg_notes.sample_num.astype(int).values
        plt.figure()
        b=np.random.choice(len(ecg_data.signal))
        plt.plot(ecg_data.signal)
        plt.xlim(b,b+5000)
        plt.plot(peaklist, ecg_data.signal[peaklist], "x")
        plt.title('  Sample Patient {} data'.format(patient))
        return None
        
    return ecg_data.signal,ecg_notes

def z_norm(result):
    """
    Normalize Data. This fits
    all values between 0 and 1. 
    """    
    result = (result-min(result))/(max(result)-min(result))
    return result

def hr_sample_len(HR,fs=360):
    """
    Convert a HR to sample len
    
    """
    return int((fs*60)/HR)# 60 seconds * samples/sec 

def isolate_patient_data(patients,classes,classes_further,classes_reducer=None, min_HR= 40,max_HR= 140,fs=360,verbose=False,plot_figs=True):
    """
    Isolation Model. Examines Patients, Normalizes Signal,
    and creates a python array with a length of the number of heat
    beates by the lenght of the longest heart rate signal. Signals
    that are smaller this are zero padded. These represent the X 
    values that used for training. They data includes patient number,
    patient heart rate, and class of condition the heart beat corres-
    ponds too. 
    
    
    Input: 
        patients:: Patient Numbers list of Patient numbers [list]
        classes:: classes to be examined {dic}
        classes_further:: expansion of previous classes with names {dic}
        classes_reducer:: optional dictionary to reduce classes 
        min_HR:: (optional) minimum HR to consider (longer HR Sample Rate)
        max_HR:: (optional) max HR to consider (longer HR Sample Rate)
        fs:: (optional) sampling frequency --> 360 for this database
        verbose:: (optional) prints out some information per patient if true [boolean]
        plot_figs:: (optional) prints out HR and Heat Beat distributions 
        
    Output: 
        X,y np arrays 
        Isolated beat:: list of lists of each patient ecg data (unpadded)
    """   
    isolated_beat= []
    start=time.time()
    print('Examining {} patients...'.format(len(patients)))

    for i,patient in enumerate(patients):
        ecg_signal,ecg_notes= get_patient_data(patient)
        peaklist= ecg_notes.sample_num.astype(int).values 
        for c in classes.values():
            class_loc=[]
            if classes_reducer != None:
                for rc in classes_reducer[c]:
                    class_loc.extend(ecg_notes.loc[ecg_notes.type == rc]['sample_num'].values.tolist())     
            else:
                class_loc= ecg_notes.loc[ecg_notes.type == c]['sample_num'].values 
            for n in range(1,len(peaklist)-1):
    
                if peaklist[n] in set(class_loc):
                    delta1= int(np.round((peaklist[n+1]-peaklist[n])/2))
                    delta2= int(np.round((peaklist[n]-peaklist[n-1])/2))
                    peak_data= ecg_signal[peaklist[n]-delta2:peaklist[n]+delta1] 
                    if hr_sample_len(max_HR) <= len(peak_data) <= hr_sample_len(min_HR):
                        isolated_beat.append([patient,get_HR(peaklist,fs=fs),c]+peak_data.tolist())

        if verbose == True:
            print('\nPatient {}...'.format(patient))
            print('Normalizing --> [0 1]')
            print('Patient HR',get_HR(peaklist,fs=fs))
        if len(patients) <=1:
            update_progress(i/(len(patients)))
        else:
            update_progress(i/(len(patients)-1))

    print('\nPadding...\n')
    isolated_beats= zero_pad(isolated_beat) 
    X=isolated_beats[:,3:].astype(float)
    y=isolated_beats[:,:3]
    avg_samp=np.array([len(l) for l in isolated_beat]).mean()
    print('\nAverage HR Sample Len: {:.2f} samples ({:.2f}s per beat)'.format(avg_samp,avg_samp/360))
    print('Average HR: {:.2f} bpm'.format(y[:,1].astype(float).mean()))
    
    if plot_figs== True:
        if len(patients)==1:
            print('\n*****Error Will Arise with Plot because only single sample used*****\n')
        print('Plotting...\n')
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        x=[len(elem) for elem in isolated_beat]
        warnings.filterwarnings("ignore")
        sns.distplot(x,rug=True)
        plt.title('Heart Rate RR Width')
        plt.xlabel('HR Sample Interval Length')
        plt.subplot(122)
        sns.distplot(y[:,1].astype(float),rug=True)
        plt.title('Heart Rate Distribution')
        plt.xlabel('HR [bpm]')
        plt.show()
        warnings.resetwarnings()
    print('Data Loaded | Shape:{}\n'.format(isolated_beats.shape))
    for label,count in Counter(y[:,2].tolist()).items():
        print('    {} cases of {}\n'.format(count,classes_further[label]))
    print('{:.2f}min Runtime'.format((time.time()-start)/60))
    return X,y,isolated_beat

def show_sample_plots(X,y,classes,classes_further,num_sigs=5,fs=360,plot_xlim=1,dims=[2,4]):
    """
    Sample Plot Generator Function. Show user selected amount of 
    signals per class to show on plt subplots. (One per class --> 2x4)

    Input: 
        X:: Isolated Patient HR [nd array]
        y:: Grouping of (Patient, HR, Class) [nd array]
        classes:: classes to be examined {dic}
        classes_further:: expansion of previous classes with names {dic}
        fs:: (optional) sampling frequency --> 360 for this database
        plot_xlim:: xlim dimenstions
    Output: 
        Sample signal plots per class
    """
    time= np.arange(0, X.shape[1])/fs
    print("MAX HB TIME:",(X.shape[1])/fs)
    colors=['m','c','b','g','r']
    plt.figure(figsize=(25,15))
    for c in range(len(classes)):
        samples = np.argwhere(y[:,2] == classes[c]).flatten()
        rand=np.random.choice(len(samples))
        samples=samples[0+rand:num_sigs+rand]
        if len(samples)>0:
            subplot_val=dims[0]*100+dims[1]*10+1+c
            plt.subplot(subplot_val)
            label=classes[c]
            plt.title(classes_further[label]+': '+classes[c])
            for samp in samples:
                plt.plot(time,X[samp])
            plt.xlim(0,plot_xlim)
            plt.xlabel('time [s]')
            plt.ylabel('Voltage [-1,1]')

def most_common_conditions(patients,top_k):
    """
    Illustration of the top k classes 
    for the input number of patients
    
    """
    allp=[] #all patient notes 
    for p in patients:
        sig,notes=get_patient_data(p)
        allp.extend(notes.type.values.tolist())
        
    return Counter(allp).most_common(top_k)

def resample_vals(X,samp_len=187):
    """
    Signal resampling function 
    """

    X_resamp= np.zeros((X.shape[0],samp_len))
    for i,x in enumerate(X):
        X_resamp[i]=resample(x,samp_len)
    return X_resamp

##################MAIN#########################     
if __name__ == '__main__':
    classes= {0:'N',1:'L',2:'R',3:'V',4:'/',5:'A',6:'f',7:'F'}
    patients = all_patients()
    patients=[101]
    X,y,isolated_beat=isolate_patient_data(patients=patients, classes=classes,classes_further=classes_further,
                             fs=360,verbose=False,plot_figs=True)