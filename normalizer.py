
import heartbeat as hb
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import fftpack
from scipy.signal import butter, lfilter, freqz
import math

def z_norm_01(result):
    """
    Normalize Data. This creats a mean of 0, and normalizes 
    all values between 0 and 1 by shifting negative values 
    upward
    """
    result_mean = result.mean()
    result -= result_mean
    if min(result)<0.0:
        result-=min(result)
    result /= max(abs(result))
    return result

def z_norm_b(result):
    """
    Normalize Data. This creats a mean of 0, and a
    std of 1
    """
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result

def z_norm(result):
    """
    Normalize Data. This fits
    all values between 0 and 1. 
    """
    result = (result-min(result))/(max(result)-min(result))
    return result

def z_norm2(result):
    """
    Normalize Data. This creats a mean of 0, and fits
    all values between -1 and 1. 
    """
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= max(abs(result))
    return result

def dynamic_threshold(signal, fs):
    mvg_avg_percent= np.arange(5,100,5).tolist() #[0 100]
    valid_percent = []
    interval_SD=[]
  
    for percent in mvg_avg_percent: #Detect peaks with all percentages, append results to list 'rrsd'
        mov_avg, peaklist= peak_finder(signal, percent, fs)
        RR_list, bpm = get_HR(peaklist)
        print(percent,bpm)
        RR_list_STD= np.std(RR_list)
        
        interval_SD.append([RR_list_STD, bpm, percent])
    for interval,HR,perc in interval_SD: #Test list entries and select valid measures
        if ((interval > 1) and ((HR > 30) and (HR < 130))):
            valid_percent.append([interval, perc])

    mov_avg, peaklist= peak_finder(signal, min(valid_percent, key = lambda t: t[0])[1], fs) #Detect peaks with 'ma_perc' that goes with lowest rrsd
    return mov_avg, peaklist

def peak_finder(signal,mvg_perc, fs):
    """
    1. Get moving average of the data
    2. Replace NaN's with moving average
    3. 5% Amplification to prevent heart rate interference
    4. for each point in signal
        current_mean = point in moving avg
        if point < current mean and len(window) =0 #NO R Complex Activity 
            get new mean avg point 
        else if point > current mean: #region of interest  
            include point in window 
            get new mean avg point
        else 
            peak = position of the point on the X-axis
            get new mean avg point

    """
    width = fs*.1
    mvg_perc= (1+mvg_perc/100)
    mov_avg = signal.rolling(int(width)).mean() #Calculate moving average
    
    #Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
    avg_hr = np.mean(signal)
    mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
    mov_avg = [x*mvg_perc for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering, in part 2 we will do this dynamically
    
    window = []
    peaklist = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in signal:
        rollingmean = mov_avg[listpos] #Get local mean
        if (datapoint <= rollingmean) and (len(window) <= 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            
            beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
            peaklist.append(beatposition) #Add detected peak to list
            window = [] #Clear marked ROI
            listpos += 1
    
    return mov_avg, peaklist 
  
class filt: 

    def fft_plot(self,sig,label,color):
        """
        Generates a FFT or fast fourier transform 
        plot for the input signal.

        Input: Signal, Label (filtered/not), Color of Figure
        Output: Histogram of Frequencies 

        """
        f_s=360
        X = fftpack.fft(sig)
        freqs = fftpack.fftfreq(len(sig)) * f_s
        plt.plot(freqs, np.abs(X))
        plt.xlabel('Frequency in Hertz [Hz]')
        plt.ylabel('Frequency Domain (Spectrum) Magnitude')
        plt.xlim(0, f_s / 2)
        plt.ylim(-5, 110)

    def butter_lowpass(self,cutoff, fs, order=5):
        """
        Butterworth lowpass filter:: Allows only signal below 
        certain cutoff to pass through. Order is a constant value
        and is part of the filter arguments. 

        Return b,a inputs for filter 

        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self,data, cutoff, fs=360, order=5):
        """
        Low pass filter for the signal data w/r to a particular
        cut off frequency. 

        """

        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def low_pass_filter_plot(self,patient,cutoff,fs=360,order=5):
        """
        Generates a 3 figure subplot
        FFT, Illustration of Frequency Response, 
        and a plot of signal vs unfilt signal 
        for that patient. 

        Inputs are patient, cuttoff freq

        Outputs are the plots. 


        """

        plt.figure(figsize=(20,10))
        sig,notes=hb.get_patient_data(patient,norm=True)

        #FFT PLOT
        plt.subplot(221)
        self.fft_plot(sig,label='Orig Sig',color='b')
        self.fft_plot(self.butter_lowpass_filter(sig,cutoff),label='Filt Sig',color='r')
        plt.title('FFT Patient: {}'.format(patient))
        plt.xlabel('Frequency [Hz]')
        plt.xlim(0,cutoff+30)

        #Low pass Filter Frequency Response 
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, sig)
        w, h = freqz(b, a, worN=8000)
        plt.subplot(222)
        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, .1*fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        #Signal Plot 
        plt.subplot(212)
        plt.plot(sig, 'b-', label='data')
        plt.plot(self.butter_lowpass_filter(sig,cutoff), 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.xlim(0,1000)
        plt.ylim(-.5,1.1)
        plt.grid()
        plt.legend()
