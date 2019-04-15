# ECG_Classification_Pytorch
### Application of deep learning and convolutional networks for ECG classification


#### Make sure heartbeat.py and normalizer.py are in the same directory
###### `heartbeat.py`  is a small written signal processing library and `normalizer.py` has several other util functions

#### There also must be a folder called mit_data (attached in zip file) where all data is stored

---

Electrocardiograms or ECG are signals that assess an individual’s cardiac rhythm as signal. This is a result of the depolarization and repolarization of the hearts’ chambers that can be interpreted voltage over time. The premise of this paper is the application of supervised deep learning to identify illustrations of labeled rhythmic aberrations. The proposed technique uses a series of single dimensional convolutions paired with a multilayered perceptron to classify five common arrhythmias. The model was trained with 75% of the data which was sampled with equal class counts per batch but tested on the data’s natural distribution. After implementation, the accuracy of the model proved to be 97.5 + .0044 % over five iterations with per class metrics higher than 85% across all classes. Future improvements include different processing techniques as well as slight adjustment to model architecture. 
​
<p align="center"> 
<img src="labelled_beat.png">
</p>

---
 
The primary objective of this project is to use a 1D Convolutional Network paired with a Multilayer perceptron that finds unhealthy signal in a continous heart beat. All code is public and can be manipulated for use. 
​
<p align="center"> 
<img src="comparison_of_beats.png">
</p>

---
​
The data used will be from the MIT-BIH Arrhythmia Database. The following database, collecting data from as far as 1975 has 48 instances of 30 minute (360 samples/sec) ECG records from 47 patients at Beth-Israel Hospital. “Twenty-three recordings were chosen at random from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed population of inpatients (about 60%) and outpatients (about 40%). The remaining 25 recordings were selected from the same set to include less common but clinically significant arrhythmias that would not be well-represented in a small random sample.” All data exists per patients in the form of csv files and instances of annotated by associate txt files.

Database: https://physionet.org/physiobank/database/html/mitdbdir/intro.htm






