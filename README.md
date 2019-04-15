# ECG_Classification_Pytorch
### Application of deep learning and convolutional networks for ECG classification


#### Make sure heartbeat.py and normalizer.py are in the same directory
###### `heartbeat.py`  is a small written signal processing library and `normalizer.py` has several other util functions

#### There also must be a folder called mit_data (attached in zip file) where all data is stored

The proposed idea for this project is inspired by the Apple Watch Series 4 and modern-day ECG machines. ECG’s or electrocardiograms are heart monitoring devices that measure the hearts’ normal sinus rhythm. Biologically speaking, cardiac muscles generate a wave pattern through depolarization and repolarization of each beat. ECG measurements are recorded by the placement of up to 12 electrodes on the body that can measure the overall magnitude of the hearts electric potential. A single heart beat voltage signal can be examined with a P wave which represents the depolarization of the atria; the QRS complex which illustrates ventricle depolarization of the atria, and finally the T wave, which indicates ventricle repolarization. The image below shows a common signal repeat unit of this waveform. 
​
												<img src="heartbeat.png">
 
The primary objective of this project is to use a 1D Convolutional Network paired with a Multilayer perceptron that finds unhealthy signal in a continous heart beat. All code is public and can be manipulated for use. 
​
​
The data used will be from the MIT-BIH Arrhythmia Database. The following database, collecting data from as far as 1975 has 48 instances of 30 minute (360 samples/sec) ECG records from 47 patients at Beth-Israel Hospital. “Twenty-three recordings were chosen at random from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed population of inpatients (about 60%) and outpatients (about 40%). The remaining 25 recordings were selected from the same set to include less common but clinically significant arrhythmias that would not be well-represented in a small random sample.” All data exists per patients in the form of csv files and instances of annotated by associate txt files.

Database:: [https://physionet.org/physiobank/database/html/mitdbdir/intro.htm]






