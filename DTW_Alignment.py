#!/usr/bin/env python3
"""
Play a file continously, and exit gracefully on signal

Based on https://github.com/steveway/papagayo-ng/blob/working_vol/SoundPlayer.py

@author Guy Sheffer (GuySoft) <guysoft at gmail dot com>
"""
try: 
    import queue
except ImportError:
    import Queue as queue
import signal
import time
import os
import threading
import pyaudio
import pylab
from pydub import AudioSegment
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
import numpy as np
import threading
from helper import applyDTW, plotCrossover, plotSTFT, exportMatchedFile, exportSecondSong, beater

def soundplot(stream):
    data = stream.read(self.millisecondchunk)
    data = np.frombuffer(data, np.int16)
    line.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
def match(self, audio_file_second):
    
    # PLAYBACK LOOP
    start = sound.duration_seconds - 15       
    length = sound.duration_seconds       
    #volume = 100.0
    playchunk = sound[start*1000.0:(start+length)*1000.0] - (60 - (60 * (self.volume/100.0)))    

    #millisecondchunk = 1024 / 1000.0

    start_second = 0
    length_second = soundTwo.duration_seconds
    time_second = 0
    playchunk_two = soundTwo[start_second*1000.0:(start_second+length_second)*1000.0] - (60 - (60 * (self.volume/100.0)))
    second_counter = 0
    second_chunk = make_chunks(playchunk_two, self.millisecondchunk*1000)
    
    while self.loop:
        self.time = start
        for chunks in make_chunks(playchunk, self.millisecondchunk*1000): 
            if self.time >= start+length-self.millisecondchunk:
                break
            elif (length - self.time <= 10) and round(chunks.duration_seconds,3) == self.millisecondchunk:
                match_chunk = self.matcher.applyDTW(chunks, second_chunk[second_counter], useFast = True, secondOnly=True)
                self.time += match_chunk.duration_seconds 
                time_second += match_chunk.duration_seconds
                stream.write(chunks._data)
                time.sleep (0.02)
                stream.write(match_chunk._data)
                #self.soundplot(stream)
                second_counter += 1
            elif not self.loop:
                break 
            else:
                self.time += self.millisecondchunk
                stream.write(chunks._data)
                #self.soundplot(stream)
        self.run(audio =soundTwo, start = time_second, stream = streamTwo)

    stream.close()
    player.terminate()
    # Open an audio segment
    sound = AudioSegment.from_file(self.filepath)
    player = pyaudio.PyAudio()

    stream = player.open(format = player.get_format_from_width(sound.sample_width),
        channels = sound.channels,
        rate = sound.frame_rate,
        output = True)
    # Open an audio segment for the second file
    soundTwo = AudioSegment.from_file(audio_file_second)
    playerTwo = pyaudio.PyAudio()
    streamTwo = playerTwo.open(format = playerTwo.get_format_from_width(soundTwo.sample_width),
        channels = soundTwo.channels,
        rate = soundTwo.frame_rate,
        output = True)
    stream = player.open(format = playerTwo.get_format_from_width(soundTwo.sample_width),
        channels = soundTwo.channels,
        rate = soundTwo.frame_rate,
        output = True)

second_start = None
second_start_available = threading.Event()
class PlayerLoop(threading.Thread):
    """
    A simple class based on PyAudio and pydub to play in a loop in the backgound
    """

    def __init__(self,filepath= None,  sound = None, loop=True, start_shift= 0, crossoverTime=0, volume = 0):
        """
        Initialize `PlayerLoop` class.

        PARAM:
            -- filepath (String) : File Path to wave file.
            -- loop (boolean)    : True if you want loop playback. 
                                False otherwise.
        """
        self.volume = volume
        self.crossoverTime = crossoverTime
        self.start_shift = start_shift
        self.sound = sound
        super(PlayerLoop, self).__init__()
        self.loop = loop
        self.millisecondchunk = 1024 / 1000
        self.sr = 44100 
        if filepath is not None:       
            self.filepath = os.path.abspath(filepath)
        # Open an audio segment     
        if self.sound is None:
            if self.filepath is None:
                raise Exception('Need either silepath or soundfile to play the music')
            else:
                self.sound = AudioSegment.from_file(self.filepath)
        else:
            self.sound = self.sound
        self.player = pyaudio.PyAudio()
        self.stream = self.player.open(format = self.player.get_format_from_width(sound.sample_width),
                channels = sound.channels,
                rate = sound.frame_rate,
                output = True)

    def run(self):
        # PLAYBACK LOOP
        length = self.sound.duration_seconds        
        if(self.start_shift<0):
            start = length + self.start_shift
        else:
            start = self.start_shift
        player = self.player
        stream = self.stream
        sound = self.sound
        playchunk = sound[start*1000.0:(start+length)*1000.0] - (60 - (60 * (self.volume/100.0)))
        millisecondchunk = 1 / 1000.0

        self.time = start
        for chunks in make_chunks(playchunk, millisecondchunk*1000):
            self.time += millisecondchunk
            stream.write(chunks._data)
            if(length-self.time <= self.crossoverTime):
                global second_start
                second_start = True
                second_start_available.set()
            if self.time >= length:
                break

        stream.close()
        player.terminate()

    def play(self):
        """
        Just another name for self.start()
        """
        self.start()

    def stop(self):
        """
        Stop playback. 
        """
        self.loop = False

def play_audio_background(audio_file, audio_file2):
    """
    Play audio file in the background, accept a SIGINT or SIGTERM to stop
    """
    fast = False
    crossoverTime = 20 #in seconds
    soundFirst = AudioSegment.from_file(audio_file)
    soundSecond = AudioSegment.from_file(audio_file2)
    #matcher = DTWMatcher()
    volume = 0
    #player = PlayerLoop(sound = soundFirst, start_shift = -(crossoverTime +3), crossoverTime= crossoverTime, volume = volume)
    #player.play()
    print(os.getpid()) 
    inputData = soundFirst[-((crossoverTime+5)*1000):-5000]
    overlayData = soundSecond[:crossoverTime*1000]
    output, shiftedSecondSong, shiftedWholeSecondSong, crossFadeTime = applyDTW(inputData, overlayData, useFast=fast, wholeFirst = soundFirst, wholeSecond = soundSecond)
    inName = "Frozen"
    outName = "BlaBla"
    exportMatchedFile(inName, outName, output, fast= fast)
    exportSecondSong(shiftedWholeSecondSong, outName)
    #playerTwo = PlayerLoop(sound = shiftedSecondSong, volume = volume)
    #second_start_available.wait()    
    #playerTwo.play()
    print(os.getpid())
    return

def analyze_one(audio_file):
     """
    Play audio file in the background, accept a SIGINT or SIGTERM to stop
    """
    fast = False
    crossoverTime = 20 #in seconds
    sound = AudioSegment.from_file(audio_file)
    overlayData = sound[:crossoverTime*1000]
    output, shiftedSecondSong, shiftedWholeSecondSong, crossFadeTime = applyDTW(inputData, overlayData, useFast=fast, wholeFirst = soundFirst, wholeSecond = soundSecond)
    inName = "Frozen"
    outName = "BlaBla"
    exportMatchedFile(inName, outName, output, fast= fast)
    exportSecondSong(shiftedWholeSecondSong, outName)
    #playerTwo = PlayerLoop(sound = shiftedSecondSong, volume = volume)
    #second_start_available.wait()    
    #playerTwo.play()
    print(os.getpid())
    return
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=True, description="Play a file continously, and exit gracefully on signal")
    parser.add_argument('audio_file', type=str, help='The Path to the audio file (mp3, wav and more supported)')
    parser.add_argument('audio_file_second', type=str, help='The Path to the second audio file (mp3, wav and more supported)')
    args = parser.parse_args()

    play_audio_background(args.audio_file, args.audio_file_second)