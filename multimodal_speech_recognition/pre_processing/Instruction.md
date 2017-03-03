##1.Audio
I use matlab bulit-in method **spectrogram()** to calculate the spectrogram with 882-points Hamming window, 442-points overlap and 500-points FFT. This will result 540*1 cell array which is x*50 matrix(x is the length of audio frame through processing).

##2.Video
I use matlab built_in function VideoReader() to read the video frames, Viola-Jones Algorithm to extract ROI. The Debug is Very hard, so I create a cell to save the special frames, and debugv.m to debug and tuning.


##3.Tuning
1 video frame = 4 audio frame.

Due to the mismatching of video frames and audio frames, I did some tuning of video frames and audio frames. 


##4.PCA
video: 100 principal components

audio: 50 principal components

Finally, video: 21600*100, audio: 21600*200

##5.Save as hdf5
As I want to use Torch to implement NNet, therefore hdf5 format is good choice being swap files. Matlab support hdf5 very well.