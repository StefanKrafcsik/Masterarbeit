import json
import os
import pickle
import numpy as np
from dtw import *
from fastdtw import fastdtw
#from pydub import AudioSegment
from pydub import AudioSegment
from scipy import signal
from scipy.fft import fft
import time
import librosa
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import euclidean
import threading
from madmom.features import (BeatTrackingProcessor, RNNBeatProcessor, DBNBarTrackingProcessor,RNNBarProcessor, RNNDownBeatProcessor,DBNDownBeatTrackingProcessor)
from math import pi, sqrt, exp
import copy
mpl.rcParams['agg.path.chunksize'] = 10000

def applyDTW(Yin, Yout,wholeFirst, wholeSecond, useFast= False, sr=44100, nperseg = 11):

    start_time = time.time()
    nperseg = np.power(2,nperseg)
    kernelsize = 17640
    gaussian_filter = gaussian_kernel(kernelsize/2)

    firstSongSTFT = signal_preprocessing_bars(Yin, nperseg = nperseg, gaussian_filter = gaussian_filter, sr = sr)
    secondSongSTFT = signal_preprocessing_bars(Yout, nperseg = nperseg, gaussian_filter = gaussian_filter, sr = sr)

    if(useFast):
        alignmentDistance, path = fastdtw(firstSongSTFT, secondSongSTFT, dist=euclidean)
        crossFadeTime = int(Yin.duration_seconds*1000- alignmentDistance/sr)
    else:
        alignment = dtw(firstSongSTFT, secondSongSTFT, keep_internals=True)
        crossFadeTime = int(Yin.duration_seconds*1000- alignment.distance/sr)

    print("Alignment global time : %s seconds" %(Yin.duration_seconds -crossFadeTime/1000))
    print("--- %s seconds ---" % (time.time() - start_time))

    shiftedSecondSong = shiftSecondSong(Yin,Yout, crossFadeTime)
    shiftedWholeSecondSong = shiftSecondSong(wholeFirst, wholeSecond, crossFadeTime)
    output = wholeFirst.append(wholeSecond, crossfade=crossFadeTime)
    return output, shiftedSecondSong,shiftedWholeSecondSong, crossFadeTime

def signal_preprocessing_beats(song, nperseg, gaussian_filter, sr = 44100):
    signalArr = song.set_channels(1)
    signalArr = np.asarray(signalArr.get_array_of_samples(),dtype=np.float32)
    signalArr2 =  np.array(signalArr, copy=True)
    beats = beater(signalArr)
    beats = beats.astype(int)*sr
    for beat in beats:
        if beat > 0:
            signalArr2[beat-2500:beat+2501]  *= gaussian_filter
    return signal.stft(signalArr2, sr, nperseg=nperseg)[2], signal.stft(signalArr, sr, nperseg=nperseg)[2]

def signal_preprocessing_bars(song, nperseg, gaussian_filter, sr = 44100):
    window_range = int(len(gaussian_filter)/2)
    signalArr = song.set_channels(1)
    signalArr = np.asarray(signalArr.get_array_of_samples(),dtype=np.float32)
    signalArr2 =  np.array(signalArr, copy=True)
    beats = bearer(signalArr)
    beats = beats * sr
    beats = np.around(beats).astype(int)
    for beat in beats:
        if (beat > 8820):
            signalArr2[beat-window_range:beat+window_range+1] *= gaussian_filter

    plotWaveArr(signalArr2,signalArr)
    return signal.stft(signalArr2, sr, nperseg=nperseg)[2]

def gaussian_kernel(n, scale=1):
    sigma = n/(np.sqrt(5)*2)
    r = range(-int(n/2),int(n/2)+1)
    r = [(exp(-float(x)**2/(2*sigma**2))) + 1 for x in r]
    return np.asarray(r)*scale

def plotCrossover(Yin, shiftedSecondSong, sr=44100):
    time1 = np.linspace(0, len(Yin.get_array_of_samples()) / sr, num=len(Yin.get_array_of_samples()))
    time2 = np.linspace(0, len(shiftedSecondSong.get_array_of_samples()) / sr, num=len(shiftedSecondSong.get_array_of_samples()))
    plt.plot(time1, Yin.get_array_of_samples(), color = 'c')
    plt.plot(time2, shiftedSecondSong.get_array_of_samples(), alpha = 0.75, color = 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plotSTFT(song1,sr = 44100, nperseg = 2048):
    f, t, firstSongSTFT = signal.stft(song1, sr, nperseg=nperseg)
    #f2, t2, secondSongSTFT = signal.stft(song2, sr, nperseg=nperseg)
    plt.pcolormesh(t, f, np.abs(firstSongSTFT))
    #plt.pcolormesh(t2, f2, np.abs(secondSongSTFT))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def plotWaveArr(signal1,signal2, sr = 44100):
    #create a time variable in seconds
    time = np.linspace(0, len(signal1) / sr, num=len(signal1))
    time2 = np.linspace(0, len(signal2) / sr, num=len(signal2))
    #plot amplitude (or loudness) over time
    plt.plot(time, signal1, color = 'c')
    plt.plot(time2, signal2, color = 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def exportMatchedFile(inName, outName, out, fast):
    print("Exporting...")
    if fast:
        outname = "Songs/mixedSongs/" + inName + "X" + outName+"Fast.wav"
    else:
        outname = "Songs/mixedSongs/" +inName + "X" + outName+".wav"
    out.export(out_f=outname, format="wav")
    print("[SUCCESS] Export as " + outname)
    printedWave = out.get_array_of_samples()
    printedWave = np.array(printedWave)
    
def shiftSecondSong(Yin, Yout, crossFadeTime):               
    # Pad second wave with 0 array with length of the first wave minus the crossfade time.
    # If we plot this over the first wave we should see the overlap
    paddingSize = Yin.duration_seconds*1000 - crossFadeTime
    silence = AudioSegment.silent(duration=paddingSize)
    return silence + Yout

def exportSecondSong(shiftedSecondSong, outName):
    shiftedSecondSong.export(out_f="Songs/mixedSongs/" + outName+'Shifted'+".wav", format="wav")

def beater(data):
    proc = BeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(data)
    return proc(act)

def bearer(data):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(data)
    result = proc(act)
    beats, bars = result[:,0], result[:,1]
    return beats#[bars==1]