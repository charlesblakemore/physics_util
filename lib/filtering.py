import numpy as np

import matplotlib.pyplot as plt



def ewma(waveform, alpha):
    '''
    Returns the exponentially weighted moving average of an input 
    waveform. Taken shamelessly from a stackoverflow answer: 
    <https://stackoverflow.com/questions/42869495>

    INPUTS

        waveform - a 1D numpy array containing the waveform to be filtered

        alpha - value in the inclusive interval [0,1] to set the degree of
            averaging, 0 being an aggressive average, 1 being no average.
            Essentially parameterizes how fast the moving average 'forgets'
            about data farther away

    OUTPUTS:

        waveform_filt - an exponentially weighted moving average of the
            input waveform
    '''

    ### Make sure we're numpy
    waveform = np.array(waveform)
    n = waveform.size

    ### Create initial weight and power matrices
    w0 = np.ones(shape=(n,n)) * (1 - alpha)
    p = np.vstack([np.arange(i,i-n,-1) for i in range(n)])

    ### Build the actual weight matrix
    w = np.tril(w0**p,0)

    ### Compute and return the ewma
    return np.dot(w, waveform[::np.newaxis]) / w.sum(axis=1)