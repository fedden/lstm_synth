
#This python script acts a function library used by the other files to the generate phases.

# coding: utf-8

# In[13]:
#inport nessessary libraries.
import os
import numpy as np
import librosa


# In[9]:

#Polar to read convertion of numpy matricies
def polar2real(mag, angle):
    #Matrix of magnitudes * the exponent of 1j(imaginary value) * the phase values (-3.14, 3.14)
    return mag * np.exp(1j*angle)


# In[10]:

#this function output the frequency increment of each bin, highest freq/bin = [0], lowest freq/bin = [n_fft/2]
def phase_angle_inc_generator(_fftFrameSize, _hop_length, sampleRate):
    #Finding the period of each bin's central frequency of my fft.

    #to stop a divide by 0.
    fft_frequencies = librosa.fft_frequencies(sr=sampleRate, n_fft=_fftFrameSize)

    # print("Phase_gen: ", fft_frequencies.shape, fft_frequencies[0], fft_frequencies[1], fft_frequencies[1024])

    fft_frequencies[0] = 1
    fft_freq_period_sample = sampleRate/fft_frequencies
    
    #Stopping the 0's element of fft_freq_period_sample from being "inf".
    #MAJOR ISSUE: 0th element is inf, 1st element is 2048, 1024th element is 2. What is the 0th element and why does this STFT return 1 more value than half the frameSize.
    #    the answer is probably the reason the phase vocoding sounds lame.
    #Going to make the 0th bin phase 0 later in this code.
    fft_freq_period_sample[0] = _fftFrameSize

    #dividing the hoplength by the period of the bins
    # to create an value to increment the phase by for each hop.
    # and scaling the the value to a number between 2*PI.
    fft_freq_period_angle_hopInc = (_hop_length / fft_freq_period_sample) * 2*np.pi
    
    fft_freq_period_angle_hopInc[0] = 0.0

    #return the array of hop incrememnts.
    return fft_freq_period_angle_hopInc


# In[11]:
#this function generates an frame of phase values for every frame of magnitude values (numFrames)
def gen_phases(numFrames, _fftFrameSize, _hop_length, sampleRate):
    #Get the ammount we are going to increment every bin's phase value by.
    phase_angle_inc = phase_angle_inc_generator(_fftFrameSize, _hop_length, sampleRate)
    
    #Initialize a random array of the number of phases in each set with a value between 0 and 2*Pi.
    #if we don't start with with random values then we get a sweeping effect when they all start 
    # from 0 at the begining of the audio.
    #this array will be incrememted in the forloop below.
    current_phase = np.random.rand(int(_fftFrameSize/2)+1) * 2*np.pi
    current_phase[0] = 0.0
    
    #this array holds 0.0 values for every phase value we need.
    #This array will be over written with the actual phase values in the forloop below. 
    new_phases = np.zeros((numFrames, int(_fftFrameSize/2)+1))
    
    #Get an integer i for every frame of phases that we need.
    for i in range(numFrames):
        #Allocate the array full of 0's with the phase values.
        new_phases[i] = current_phase
        #Increment current phase but the increment values.
        current_phase = current_phase + phase_angle_inc
        #Modulo the incremented phase by 2*pi to make sure it stays in the range of phases.
        current_phase = current_phase % 2*np.pi
    
    #Minus all the new phase values by Pi so they are a range of -Pi to Pi
    new_phases = new_phases-(np.pi)
    
    #Return an array full of frames of phase values (as many frames as numFrames)
    return new_phases


# In[12]:
#fft2samples takes in sets of magnitudes and phases and inverts them back into amplitude values.
def fft2samples(_mags, _phases, _hop_length):
    #First one must convert the magnitudes and phases in to their real and imaginary values using the polar2real function defined above.
    real = polar2real(_mags, _phases)
    #Transpose the new matricies and librosa like it the shape (fftSize, numFrames)
    real = real.T
    #Take the ifft of the data using librosa and the correct hoplength.
    return librosa.istft(real, hop_length=_hop_length)


